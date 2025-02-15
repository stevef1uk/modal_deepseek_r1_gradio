# Gradio GUI for LLaMA.cpp Server
# This script deploys a Gradio GUI on Modal that interacts with a remote LLaMA.cpp server.
# Author: Steven Fisher (stevef@gmail.com)
# 
# Description: This script sets up a Gradio interface on Modal, which communicates
#              with a separate llama.cpp server. It requires the following secrets:
#              - MODAL_SECRET_LLAMA_CPP_API_KEY: API key for accessing the LLaMA.cpp server.
#              - gradio_app_access_key: Access key for securing the Gradio app.
#              - llama_server_url: URL of the remote LLaMA.cpp server.

import modal
import pathlib
import shlex
import subprocess
import os

GRADIO_PORT = 8000

app = modal.App("gradio-app")

image = modal.Image.debian_slim(python_version="3.11").pip_install("gradio", "fastapi", "uvicorn")

# Define the secrets
llm_secret = modal.Secret.from_name("MODAL_SECRET_LLAMA_CPP_API_KEY")
gradio_access_secret = modal.Secret.from_name("gradio_app_access_key")
server_url_secret = modal.Secret.from_name("llama_server_url")  # Add new secret for server URL

fname = "gui_for_llm.py"
gradio_script_local_path = pathlib.Path(__file__).parent / fname
gradio_script_remote_path = pathlib.Path("/root") / fname

if not gradio_script_local_path.exists():
    raise RuntimeError(f"{fname} not found! Place the script with your gradio app in the same directory.")

gradio_script_mount = modal.Mount.from_local_file(
    gradio_script_local_path,
    gradio_script_remote_path,
)

@app.function(
    image=image,
    secrets=[llm_secret, gradio_access_secret, server_url_secret],  # Add server URL secret
    mounts=[gradio_script_mount],
    allow_concurrent_inputs=100,
    concurrency_limit=1, # Drives number of containers!
)
@modal.web_server(GRADIO_PORT, startup_timeout=60)
def web_app():
    target = shlex.quote(str(gradio_script_remote_path))
    cmd = f"python {target} --host 0.0.0.0 --port {GRADIO_PORT}"
    subprocess.Popen(cmd, shell=True)

    api_key = modal.Secret.from_name("MODAL_SECRET_LLAMA_CPP_API_KEY")
    print("API Key Retrieved:", api_key)  # Debugging line
    if not api_key:
        return "Error: LLM API key not found."

