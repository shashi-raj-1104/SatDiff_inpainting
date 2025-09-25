# src/run_inference.py
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch
import os

def run_inference(image_path, mask_path, output_path, config):
    """
    Run inpainting on a single image with a given mask.
    """
    # Load images
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("RGB")

    # Load pipeline
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        config['model']['pretrained'],
        torch_dtype=torch.float16
    )
    pipe.to("cuda")

    # Run inpainting
    output = pipe(
        prompt=[""] * 1,  # Empty prompt for inpainting
        image=image,
        mask_image=mask,
        num_inference_steps=50,  # You can adjust
        guidance_scale=7.5
    )

    # Save result
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    output.images[0].save(output_path)
    print(f"Result saved at {output_path}")
