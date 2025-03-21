"""
Modal-based LLaMA.cpp Server with DeepSeek R1 Model

This module implements a high-performance LLaMA.cpp server using Modal cloud infrastructure
and the DeepSeek R1 model. It includes:

1. Model initialization and caching
2. Token-based authentication using Modal secrets
3. ASGI-based request validation and routing
4. Multi-GPU inference using H100s
5. Automatic model downloading and merging

Key Components:
- validate(): ASGI application that handles token validation and request routing
- serve(): GPU-accelerated inference endpoint
- download_model(): Downloads and caches model files
- merge_model_files(): Merges split model files
- initialize(): Coordinates model setup and validation

Authentication:
- Uses Bearer token authentication
- Tokens are stored in Modal secrets
- All requests must include a valid token in the Authorization header
- Token validation happens in the validate() ASGI application before any GPU resources are allocated

Usage:
1. Deploy the server:
   modal deploy main.py

2. Make requests:
   curl -X POST "https://[username]--[app-name]-validate.modal.run/v1/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer <your-token>" \
        -d '{
          "prompt": "Your prompt here",
          "max_tokens": 100,
          "temperature": 0.7
        }'

Environment Variables:
- MODELS_DIR: Directory for model storage
- TOKEN: Authentication token (stored in Modal secrets)

Dependencies:
- Modal
- llama-cpp-python
- huggingface_hub
"""

# DeepSeek LLM Server with llama.cpp
#
# Authors:
# - Original implementation
# - Steven Fisher (stevef1uk@gmail.com) + Cursor :-)
#
# This implementation provides a FastAPI server running DeepSeek-R1 language model
# using llama.cpp backend. It features:
#
# - GPU-accelerated inference using CUDA
# - Bearer token authentication
# - Automatic model downloading and caching
# - GGUF model file merging
# - Swagger UI documentation
#
# Key Components:
#
# 1. Infrastructure Setup:
#    - Uses Modal for serverless deployment
#    - CUDA 12.4.0 with development toolkit
#    - Python 3.12 environment
#
# 2. Model Configuration:
#    - DeepSeek-R1 model with UD-IQ1_S quantization
#    - Persistent model storage using Modal Volumes
#    - Automatic GGUF file merging for split models
#
# 3. Server Features:
#    - FastAPI-based REST API
#    - Bearer token authentication
#    - Interactive documentation at /docs endpoint
#    - Configurable context length and batch size
#    - Flash attention support
#
# Hardware Requirements:
#    - 3x NVIDIA H100-80GB GPUs
#    - Supports concurrent requests
#
# Quick Start:
# 1. Initialize (downloads and merges model, ~130GB):
#    modal run --detach main.py::initialize
#    
#    You can monitor progress with:
#    modal app logs myid-llama-cpp-server-v1
#
# 2. Deploy the server:
#    modal deploy main.py
#
# 3. Test the deployment:
#    export API_URL="https://[your-prefix]--[app-name]-llama-cpp-server-v1-serve.modal.run"
#    export API_TOKEN="your-token-from-logs"
#    
#    curl -X POST "${API_URL}/v1/completions" \
#      -H "Content-Type: application/json" \
#      -H "Authorization: Bearer ${API_TOKEN}" \
#      -d '{
#        "prompt": "What is the capital of France?",
#        "max_tokens": 100,
#        "temperature": 0.7
#      }'
#
# Authentication:
# All API endpoints (except documentation) require Bearer token authentication
# Example:
# curl -H "Authorization: Bearer your-token" \
#   https://[your-prefix]--[app-name]-llama-cpp-server-v1-serve.modal.run/v1/completions \
#   -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'
#
# Model Settings:
# - Context length (n_ctx): 4096
# - Batch size (n_batch): 128
# - Thread count (n_threads): 12
# - GPU Layers: All (-1)
# - Flash Attention: Enabled
#
# Note: 
# 1. The server includes automatic redirection from root (/) to documentation (/docs)
# 2. First run requires downloading and merging model files (~130GB)
# 3. GPU costs vary by type and count - monitor usage accordingly

from __future__ import annotations

import glob
import subprocess
import os
import secrets
import time
from datetime import datetime
from pathlib import Path
import modal
from modal import Secret
import requests
import shutil

# Constants for CUDA setup
cuda_version = "12.4.0"  # Latest stable CUDA version
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Create the Modal app
app = modal.App("myidsjf567321-llama-cpp-server-v1")  # Change "myid" to your desired prefix

# Constants
MINUTES = 60
INFERENCE_TIMEOUT = 15 * MINUTES
MERGE_TIMEOUT = 60 * MINUTES  # Increase to 1 hour to be safe
MODELS_DIR = "/deepseek"
cache_dir = "/root/.cache/deepseek"

# System memory for H100-80GB
SYSTEM_MEMORY = 131072  # 128GB of system memory

# After the constants and before any function definitions
def get_gpu_config():
    """Helper function to get the correct GPU configuration"""
    # Default configuration for H100
    gpu_type = os.environ.get("gpu_type", "H100")  # Default to H100
    gpu_count = int(os.environ.get("gpu_count", "3"))  # Default to 3
    
    gpu_map = {
        "H100": "H100",  # Changed from "H100-80GB" to "H100"
        "A100": "A100-40GB",
        "L40": "L40-48GB",
        "A10G": "A10G-24GB",
    }
    
    if gpu_type not in gpu_map:
        print(f"⚠️ Warning: Unknown GPU type {gpu_type}, falling back to H100")
        gpu_type = "H100"
    
    gpu_spec = f"{gpu_map[gpu_type]}:{gpu_count}"
    print(f"🖥️ Using GPU configuration: {gpu_spec}")
    return gpu_spec

# Move this up near the other constants and imports
TOKEN = secrets.token_urlsafe(32)
print(f"🔑 Your API token is: {TOKEN}")
print("\n To create/update the Modal secret, run this command:")
print(f"modal secret create MODAL_SECRET_LLAMA_CPP_API_KEY llamakey={TOKEN}")

# Create a temporary secret for this run
secret = Secret.from_dict({"TOKEN": TOKEN})
print(f"🔒 Created Modal secret with token")

# Create the model cache volumes
model_cache = modal.Volume.from_name("deepseek", create_if_missing=True)
merge_cache = modal.Volume.from_name("deepseek-merge", create_if_missing=True)  # New volume for merging

# Define the download image with tqdm
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface_hub[hf_transfer]==0.26.2", "tqdm", "requests")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Define the merge image
merge_image = (
    modal.Image.debian_slim(python_version="3.12")  # Simple Debian image is enough
    .pip_install("huggingface_hub")
    .run_commands(
        # Set non-interactive frontend
        "export DEBIAN_FRONTEND=noninteractive",
        # Install basic build dependencies
        "apt-get update",
        "DEBIAN_FRONTEND=noninteractive apt-get install -y git build-essential cmake curl libcurl4-openssl-dev",
        # Clone llama.cpp
        "git clone https://github.com/ggerganov/llama.cpp",
        # Build without CUDA - we only need file operations
        "cmake llama.cpp -B llama.cpp/build",
        # Build just the merge tool
        "cmake --build llama.cpp/build --config Release -j --target llama-gguf-split",
        # Copy binary to accessible location
        "cp llama.cpp/build/bin/llama-gguf-split /usr/local/bin/"
    )
)

# Combine all apt installations and system dependencies
vllm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
        "cmake",
        "curl",
        "libcurl4-openssl-dev",
        "libopenblas-dev",
        "libomp-dev",
        "clang",
    )
    .pip_install(
        [
            "fastapi",
            "sse_starlette",
            "pydantic",
            "uvicorn[standard]",
            "python-multipart",
            "starlette-context",
            "pydantic-settings",
            "ninja",
            "packaging",
            "wheel",
            "torch",
            "requests",
        ]
    )
    .run_commands(
        'CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_MMQ=on" pip install llama-cpp-python',
        gpu="H100",  # Updated to new syntax
    )
    .entrypoint([])  # remove NVIDIA base container entrypoint
)

# Create a lightweight image for token validation
validation_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi", "requests", "uvicorn[standard]")
)

# Add a flag to track if model has been downloaded
model_downloaded = False

# Global variable to store the LLM instance
_llm = None

@app.function(
    image=merge_image,
    volumes={
        MODELS_DIR: model_cache,
        "/merge_workspace": merge_cache
    },
    timeout=MERGE_TIMEOUT,  # Use the longer timeout
    memory=65536,
    retries=2
)
def merge_model_files(model_dir: str, valid_files: list[str]) -> str:
    """Merge GGUF files and return path to merged file"""
    import os
    from datetime import datetime
    import shutil
    import glob
    from subprocess import run
    
    def print_flush(*args, **kwargs):
        kwargs['flush'] = True
        print(*args, **kwargs)
        
    # Create merge workspace directory
    merge_dir = "/merge_workspace"
    os.makedirs(merge_dir, exist_ok=True)
    merged_file = f"{merge_dir}/DeepSeek-R1-UD-IQ1_S.gguf"
    temp_output = f"{merged_file}.temp"
    
    try:
        # Correct the path to include the nested directory
        actual_model_dir = os.path.join(model_dir, "DeepSeek-R1-UD-IQ1_S")
        print_flush(f"Looking for files in: {actual_model_dir}")
        
        # Use glob to find all GGUF files
        valid_files = sorted(glob.glob(f"{actual_model_dir}/*-of-*.gguf"))
        
        if not valid_files:
            print_flush("❌ No valid files found for merging")
            print_flush("Directory contents:")
            for root, dirs, files in os.walk(model_dir):
                print_flush(f"Directory: {root}")
                for f in files:
                    fpath = os.path.join(root, f)
                    size = os.path.getsize(fpath) / (1024**3)
                    print_flush(f"  - {f}: {size:.2f} GB")
            return ""
            
        print_flush(f"📁 Found files for merging:")
        total_size = 0
        
        # Verify all input files exist and are readable
        for f in valid_files:
            if not os.path.exists(f):
                print_flush(f"❌ Input file missing: {f}")
                return ""
            try:
                size_gb = os.path.getsize(f) / (1024**3)
                total_size += size_gb
                print_flush(f"   - {f}: {size_gb:.2f} GB")
            except Exception as e:
                print_flush(f"❌ Error checking input file {f}: {str(e)}")
                return ""
                
        print_flush(f"Total expected size: {total_size:.2f} GB")
            
        # Clean up any partial merges
        for f in [temp_output]:
            if os.path.exists(f):
                print_flush(f"Removing existing file: {f}")
                os.remove(f)
                
        # Verify disk space in merge volume
        disk_stats = shutil.disk_usage(merge_dir)
        free_space_gb = disk_stats.free / (1024**3)
        print_flush(f"Available disk space in merge volume: {free_space_gb:.2f} GB")
        if free_space_gb < total_size + 10:  # Add 10GB buffer
            print_flush("❌ Not enough disk space for merge!")
            return ""
            
        # Create merge directory if it doesn't exist
        os.makedirs(os.path.dirname(temp_output), exist_ok=True)
        
        # Merge files using llama-cpp's merge tool with correct command format
        print_flush("\n🔄 Starting file merge...")
        merge_cmd = [
            "llama-gguf-split",
            "--merge",
            valid_files[0],  # First file
            temp_output      # Output file
        ]
        print_flush(f"Running merge command: {' '.join(merge_cmd)}")
        result = run(merge_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print_flush(f"❌ Merge failed: {result.stderr}")
            print_flush(f"Command output: {result.stdout}")
            return ""
        
        # After successful merge, verify the file size before moving
        if os.path.exists(temp_output):
            temp_size = os.path.getsize(temp_output) / (1024**3)
            if temp_size < 100:  # We expect around 130GB
                print_flush(f"❌ Merge appears incomplete: temp file only {temp_size:.2f} GB")
                os.remove(temp_output)
                return ""
                
        # Move to final location
        print_flush(f"Moving temp file to final location: {merged_file}")
        os.rename(temp_output, merged_file)
        
        final_size = os.path.getsize(merged_file) / (1024**3)
        print_flush(f"\n✅ Merge completed successfully")
        print_flush(f"Final file: {merged_file} ({final_size:.2f} GB)")
        
        # Ensure file is fully written
        os.sync()
        
        # Verify final size before commit
        if final_size < 100:  # Sanity check
            print_flush(f"❌ Final file size too small: {final_size:.2f} GB")
            os.remove(merged_file)
            return ""
            
        # Commit the merge volume
        print_flush("💾 Committing merge volume...")
        merge_cache.commit()
        print_flush("✅ Volume committed successfully")
        
        # Final verification
        print_flush("🔍 Verifying committed file...")
        if os.path.exists(merged_file) and os.path.getsize(merged_file) / (1024**3) > 100:
            print_flush("✅ File verified successfully")
            
            # Clean up original files
            print_flush("\n🧹 Cleaning up original model files...")
            try:
                shutil.rmtree(model_dir)
                os.sync()  # Force flush to disk
                print_flush("✅ Original model files deleted successfully")
                
                # Commit the changes to the volume
                print_flush("💾 Committing changes to model cache volume...")
                model_cache.commit()
                print_flush("✅ Volume committed successfully")
                
            except Exception as e:
                print_flush(f"⚠️ Warning: Could not delete original files: {str(e)}")
            
            return merged_file
            
    except Exception as e:
        print_flush(f"❌ Merge failed with error: {str(e)}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return ""

@app.function(
    image=download_image,
    volumes={MODELS_DIR: model_cache},
    timeout=30 * MINUTES,
    container_idle_timeout=300,
)
def download_model(repo_id: str = None, patterns: list[str] = None, revision: str = None) -> bool:
    """Download model files from Hugging Face"""
    from huggingface_hub import snapshot_download
    import os
    import time
    import shutil
    
    org_name = "unsloth"
    model_name = "DeepSeek-R1"
    quant = "UD-IQ1_S"
    repo_id = f"{org_name}/{model_name}-GGUF"
    
    print(f"🦙 Starting model download from {repo_id}")
    download_dir = f"{MODELS_DIR}/{model_name}-{quant}"
    
    try:
        # Clean up any existing downloads
        if os.path.exists(download_dir):
            print(f"🧹 Cleaning up existing directory: {download_dir}")
            shutil.rmtree(download_dir)
        os.makedirs(download_dir, exist_ok=True)
        
        print("\n⏳ Starting download...")
        start_time = time.time()
        
        # Download the entire folder with the specific quantization
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=download_dir,
            allow_patterns=[f"DeepSeek-R1-{quant}/*"],
            ignore_patterns=["*.lock", "*.json", "*.md"]
        )
        
        print(f"📥 Files downloaded to: {downloaded_path}")
        
        # Find all downloaded GGUF files
        valid_files = []
        total_size = 0
        print("\n🔍 Verifying downloaded files:")
        for root, _, files in os.walk(downloaded_path):
            for file in files:
                if file.endswith('.gguf') and 'incomplete' not in file:
                    file_path = os.path.join(root, file)
                    size = os.path.getsize(file_path) / (1024**3)
                    print(f"  - {file}: {size:.2f} GB")
                    total_size += size
                    valid_files.append(file_path)
        
        print(f"\n📦 Total size: {total_size:.2f} GB")
        
        if total_size < 130:  # Expected size is ~138 GB
            print("❌ Download appears incomplete - total size too small")
            return False
            
        # Sort files to ensure correct order
        valid_files.sort()
        
        # Commit the volume to ensure files are saved
        print("\n💾 Committing files to volume...")
        model_cache.commit()
        
        # Trigger merge operation
        print("\n🔄 Starting merge operation...")
        merged_path = merge_model_files.remote(download_dir, valid_files)
        
        if merged_path:
            print("✅ Model files merged successfully")
            return True
        else:
            print("❌ Model merge failed")
            return False
            
    except Exception as e:
        print(f"❌ Download failed with error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

@app.function(
    image=download_image,
    volumes={MODELS_DIR: model_cache},
    timeout=10 * MINUTES,
    container_idle_timeout=300,
)
def init_model(skip_download: bool = False):
    """Initialize the model by downloading and merging files"""
    print("🚀 Starting model initialization...")
    
    if skip_download:
        print("⏩ Skipping download phase as requested")
        return True
    
    org_name = "unsloth"
    model_name = "DeepSeek-R1"
    quant = "UD-IQ1_S"
    repo_id = f"{org_name}/{model_name}-GGUF"
    # More specific patterns for each file
    model_patterns = [
        f"DeepSeek-R1-{quant}-00001-of-00003.gguf",
        f"DeepSeek-R1-{quant}-00002-of-00003.gguf",
        f"DeepSeek-R1-{quant}-00003-of-00003.gguf"
    ]
    revision = "02656f62d2aa9da4d3f0cdb34c341d30dd87c3b6"
    
    # Download model files
    print(f"📥 Downloading model files from {repo_id}")
    print(f"Looking for files: {model_patterns}")
    download_result = download_model.remote(repo_id, model_patterns, revision)
    
    if not download_result:
        print("❌ Model download failed")
        return False
        
    print("✅ Download completed, reloading cache")
    model_cache.reload()
    
    return True

@app.local_entrypoint()
def main(skip_download: bool = False):
    """Run initialization with optional download skip"""
    print(f"🚀 Starting initialization with skip_download={skip_download}")
    return initialize.remote(skip_download=skip_download)

@app.function(
    image=download_image,
    volumes={MODELS_DIR: model_cache},
    timeout=10 * MINUTES,
)
def initialize(skip_download: bool = False):
    """Run initialization before server starts"""
    print("🚀 Starting initialization sequence...")
    
    try:
        # Create model directory if it doesn't exist
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Run init_model and wait for completion
        print("\n⏳ Starting model initialization...")
        init_result = init_model.remote(skip_download=skip_download)  # Pass the flag here
        if not init_result:
            print("❌ Initialization failed")
            return False
        
        # List contents after initialization
        print("\n📁 Contents after initialization:")
        for root, dirs, files in os.walk(MODELS_DIR):
            print(f"Directory: {root}")
            for d in dirs:
                print(f"  Dir: {d}")
            for f in files:
                size = os.path.getsize(os.path.join(root, f)) / (1024**3)
                print(f"  File: {f} ({size:.2f} GB)")
        
        # Always run merge operation regardless of skip_download
        print("\n🔄 Starting merge operation...")
        model_dir = f"{MODELS_DIR}/DeepSeek-R1-UD-IQ1_S"
        valid_files = sorted(glob.glob(f"{model_dir}/**/*.gguf", recursive=True))
        if not valid_files:
            print("❌ No files found to merge")
            return False
            
        merged_path = merge_model_files.remote(model_dir, valid_files)
        
        if not merged_path:
            print("❌ Model merge failed")
            return False
        
        print("✅ Initialization complete")
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed with error: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

@app.function(
    image=validation_image,
    gpu=None,
    timeout=3600,
    container_idle_timeout=300,
    secrets=[secret],
)
@modal.asgi_app()
def validate():
    """Initial validation endpoint that gates access to the LLM server"""
    import json
    import os
    
    async def app(scope, receive, send):
        """Simple ASGI application"""
        if scope["type"] != "http":
            return
            
        # Get the request method and path
        method = scope["method"]
        path = scope["path"]
        
        if method != "POST" or path != "/v1/completions":
            await send({
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"application/json")]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"detail": "Not found"}).encode()
            })
            return
            
        # Check authorization
        headers = dict(scope["headers"])
        auth_header = headers.get(b"authorization", b"").decode()
        
        if not auth_header:
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer")
                ]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"detail": "No authentication token provided"}).encode()
            })
            return
            
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer" or token != os.environ["TOKEN"]:
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"www-authenticate", b"Bearer")
                    ]
                })
                await send({
                    "type": "http.response.body",
                    "body": json.dumps({"detail": "Invalid authentication token"}).encode()
                })
                return
                
        except Exception:
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer")
                ]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"detail": "Invalid authentication header"}).encode()
            })
            return
            
        # Read request body
        message = await receive()
        body = message.get('body', b'')
        
        # Handle more body chunks if any
        while message.get('more_body', False):
            message = await receive()
            body += message.get('body', b'')
            
        try:
            request_data = json.loads(body)
            print(f"Received request data: {request_data}")
            
            if "prompt" not in request_data:
                await send({
                    "type": "http.response.start",
                    "status": 400,
                    "headers": [(b"content-type", b"application/json")]
                })
                await send({
                    "type": "http.response.body",
                    "body": json.dumps({"detail": "prompt is required"}).encode()
                })
                return
                
            # Prepare request for serve
            processed_data = {
                "prompt": request_data["prompt"],
                "max_tokens": request_data.get("max_tokens", 100),
                "temperature": request_data.get("temperature", 0.7),
                "stream": request_data.get("stream", False)  # Use the requested stream value
            }
            
            print(f"Calling serve with data: {processed_data}")
            
            if processed_data["stream"]:
                # Streaming response
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")]
                })
                
                async for chunk in serve.remote_gen(processed_data):
                    await send({
                        "type": "http.response.body",
                        "body": json.dumps(chunk).encode(),
                        "more_body": True
                    })
                
                # Send final empty chunk
                await send({
                    "type": "http.response.body",
                    "body": b"",
                    "more_body": False
                })
            else:
                # Non-streaming response
                response = serve.remote(processed_data)
                
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")]
                })
                await send({
                    "type": "http.response.body",
                    "body": json.dumps(response).encode(),
                    "more_body": False
                })
            
        except json.JSONDecodeError:
            await send({
                "type": "http.response.start",
                "status": 400,
                "headers": [(b"content-type", b"application/json")]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"detail": "Invalid JSON in request body"}).encode()
            })
        except Exception as e:
            print(f"Error: {str(e)}")
            await send({
                "type": "http.response.start",
                "status": 500,
                "headers": [(b"content-type", b"application/json")]
            })
            await send({
                "type": "http.response.body",
                "body": json.dumps({"detail": str(e)}).encode()
            })
    
    return app

@app.function(
    image=vllm_image,
    gpu="H100:3",
    container_idle_timeout=300,
    timeout=3600,
    volumes={
        "/merge_workspace": merge_cache
    },
    secrets=[secret],
    concurrency_limit=1,
)
def serve(request_data: dict):
    """Create and run LLM server - only called after validation"""
    from llama_cpp import Llama
    import os
    
    global _llm
    print(f"Received request data: {request_data}")
    
    # Initialize LLM only if it hasn't been initialized yet
    if _llm is None:
        # Use the correct path from the merge volume
        model_path = "/merge_workspace/DeepSeek-R1-UD-IQ1_S.gguf"
        
        print(f"\nChecking model path: {model_path}")
        if not os.path.exists(model_path):
            print(f"❌ Error: Model not found at {model_path}")
            print("\n📁 Contents of /merge_workspace:")
            for root, dirs, files in os.walk("/merge_workspace"):
                print(f"Directory: {root}")
                for f in files:
                    size = os.path.getsize(os.path.join(root, f)) / (1024**3)
                    print(f"  - {f} ({size:.1f} GiB)")
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        size_gb = os.path.getsize(model_path) / (1024**3)
        print(f"✅ Found model: {model_path} ({size_gb:.1f} GiB)")
        
        print("\n🚀 Initializing LLama model...")
        try:
            # Initialize model with more conservative parameters
            _llm = Llama(
                model_path=model_path,
                n_ctx=4096,           # Reduced from 8096
                n_batch=64,           # Reduced from 512
                n_gpu_layers=-1,      # Still use all layers on GPU
                verbose=True,
                use_mmap=True,        # Enable memory mapping
                use_mlock=False,
                main_gpu=0,
                tensor_split=[0.4, 0.3, 0.3],  # Split across 3 GPUs
                offload_kqv=True,
                n_threads=4           # Reduced from 8
            )
            print("✅ Model initialized successfully")
        except Exception as e:
            print(f"❌ Error during model initialization: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            raise
    else:
        print("✅ Using existing model instance")

    try:
        print("\n💭 Processing completion request...")
        response = _llm.create_completion(
            prompt=request_data["prompt"],
            max_tokens=request_data.get("max_tokens", 100),
            temperature=request_data.get("temperature", 0.7),
            stream=request_data.get("stream", False)
        )
        
        # Return the response directly (no yielding for non-streaming)
        return response
        
    except Exception as e:
        print(f"❌ Error during inference: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    print("🚀 To deploy the LLaMA.cpp server, run:")
    print("modal deploy main.py")
    print("\nOnce deployed, the server will be available at:")
    print("https://[username]--[app-name]-llama-cpp-server-v1-serve.modal.run")
    
    # For local development, you can use:
    # modal deploy --env dev main.py
