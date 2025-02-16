# Gradio GUI for LLaMA.cpp Server
# This script deploys a Gradio GUI on Modal that interacts with a remote LLaMA.cpp server.
# Author: Steven Fisher (stevef@gmail.com)
# 
# Description: This script sets up a Gradio interface on Modal, which communicates
#              with a separate llama.cpp server. It requires the following secrets:
#              - MODAL_SECRET_LLAMA_CPP_API_KEY: API key for accessing the LLaMA.cpp server.
#              - gradio_app_access_key: Access key for securing the Gradio app.
#              - llama_server_url: URL of the remote LLaMA.cpp server.

from fastapi import FastAPI
import gradio as gr
from gradio.routes import mount_gradio_app
import uvicorn
import os
import requests
import json
import modal

# Create Modal app
app = modal.App("llama-gradio-interface")

# Create Modal secret reference for server URL
server_url_secret = modal.Secret.from_name("llama_server_url")

def chat_with_llm(access_key: str, message: str):
    """Chat function that uses the LLM service."""
    # Retrieve the expected access key from environment variables
    expected_access_key = os.environ.get("MODAL_SECRET_GRADIO_APP_ACCESS_KEY")
    if access_key != expected_access_key:
        return "Error: Invalid access key."

    # Access the LLM API key and server URL
    api_key = os.environ.get("llamakey")
    server_url = os.environ.get("LLAMA_SERVER_URL")
    
    if not api_key:
        return "Error: LLM API key not found."
    if not server_url:
        return "Error: Server URL not found in environment variables"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "prompt": f"{message}\n\n",
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": False,  # Disable streaming for now
        "stop": ["</s>", "\n\n"],
        "echo": False
    }

    try:
        response = requests.post(
            f"{server_url}/v1/completions",
            headers=headers,
            json=payload,
            stream=False,
            timeout=300
        )
        response.raise_for_status()
        
        # Parse the response
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("text", "").strip()
        return "Error: No response generated"

    except Exception as e:
        print(f"Error details: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        return f"Error: {str(e)}"

def create_app():
    """Create and return the ASGI app"""
    # Define the Gradio interface
    demo = gr.Interface(
        fn=chat_with_llm,
        inputs=[
            gr.Textbox(label="Access Key", type="password"),
            gr.Textbox(label="Enter your message")
        ],
        outputs=gr.Textbox(label="Response", interactive=False),
        title="LLM Chat Interface",
        description="Enter the access key and your message to chat with the Llama model.",
        live=False
    )

    # Create FastAPI app and mount Gradio
    app = FastAPI()
    return mount_gradio_app(
        app=app,
        blocks=demo,
        path="/"
    )

@app.function(
    secrets=[server_url_secret],
    is_generator=True  # Add this flag
)
@modal.asgi_app()
def web_app():
    """Return the ASGI app"""
    return create_app()

if __name__ == "__main__":
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)