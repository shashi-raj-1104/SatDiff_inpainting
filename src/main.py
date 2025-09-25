# src/main.py
from src.setup import init
from src.run_inference import run_inference

# Hugging Face token
# HF_TOKEN = "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"
HF_TOKEN = "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"

# Paths
IMG_PATH = "data/image.png"
MASK_PATH = "data/mask.png"
OUTPUT_PATH = "outputs/result.png"

# Model config
CONFIG = {

    'model': {
        'pretrained': 'runwayml/stable-diffusion-inpainting'
    }
}

def main():
    # Initialize environment
    init(HF_TOKEN)

    # Run inference
    run_inference(IMG_PATH, MASK_PATH, OUTPUT_PATH, CONFIG)

if __name__ == "__main__":
    main()
