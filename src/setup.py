# src/setup.py
from huggingface_hub import login
import os
import subprocess

def init(hf_token):
    """
    Initialize environment: Hugging Face login and install dependencies
    """
    # Hugging Face login
    login(hf_token)

    # Ensure outputs folder exists
    os.makedirs("outputs", exist_ok=True)

    # Install dependencies (if not already installed)
    subprocess.run(["pip", "install", "-q", "diffusers", "transformers", "accelerate", "safetensors", "Pillow", "PyYAML"])
