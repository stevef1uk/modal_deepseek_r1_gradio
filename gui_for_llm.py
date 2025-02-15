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

# Function to interact with the LLM service
def chat_with_llm(access_key: str, message: str):
    """Chat function that uses the LLM service."""
    # Retrieve the expected access key from environment variables
    expected_access_key = os.environ.get("MODAL_SECRET_GRADIO_APP_ACCESS_KEY")
    if access_key != expected_access_key:
        yield "Error: Invalid access key."
        return

    # Access the LLM API key and server URL
    api_key = os.environ.get("llamakey")
    server_url = os.environ.get("LLAMA_SERVER_URL", "https://stevef1uk--myid-llama-cpp-server-v1-serve-dev.modal.run")
    
    if not api_key:
        yield "Error: LLM API key not found."
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "llama2",
        "prompt": f"Question: {message}\n\nAnswer:",
        "max_tokens": 10000,
        "temperature": 0.7,
        "stream": True  # Enable streaming
    }

    try:
        response = requests.post(
            f"{server_url}/v1/completions",  # Use the configured server URL
            headers=headers,
            json=payload,
            stream=True,  # Enable streaming
            timeout=300
        )
        response.raise_for_status()

        # Accumulate the streamed response
        accumulated_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8').strip()
                print("Raw Line:", decoded_line)  # Debugging line
                if decoded_line.startswith("data: "):
                    decoded_line = decoded_line[len("data: "):]
                try:
                    data = json.loads(decoded_line)
                    text = data.get("choices", [{}])[0].get("text", "")
                    accumulated_response += text
                    yield accumulated_response  # Stream the accumulated response
                except json.JSONDecodeError:
                    print("Failed to decode JSON:", decoded_line)  # Debugging line
                    continue

    except Exception as e:
        yield f"Error: {str(e)}"

# Define the Gradio interface with streaming
demo = gr.Interface(
    fn=chat_with_llm,
    inputs=[
        gr.Textbox(label="Access Key", type="password"),
        gr.Textbox(label="Enter your message")
    ],
    outputs=gr.Textbox(label="Response", interactive=False),
    title="LLM Chat Interface",
    description="Enter the access key and your message to chat with the Llama model.",
    live=False  # Disable live updates to prevent pre-submission output
)

# Create a FastAPI app and mount the Gradio app
web_app = FastAPI()
app = mount_gradio_app(
    app=web_app,
    blocks=demo,
    path="/"
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)