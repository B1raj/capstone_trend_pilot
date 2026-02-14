# Stable Diffusion Local Image Generator

"""
This script takes a text prompt and generates an image using a local Stable Diffusion model.
Requires diffusers, torch, and a downloaded Stable Diffusion model.
"""

import sys
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

def generate_image(prompt: str, output_path: str, model_path: str = "CompVis/stable-diffusion-v1-4"):
    # Use float16 if CUDA is available, else float32 for CPU
    
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    pipe = pipe.to(device)
    # Use fewer steps and smaller image for speed
    image = pipe(prompt, num_inference_steps=15, height=240, width=240).images[0]
    image.save(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_image.py <prompt> <output.png>")
        sys.exit(1)
    prompt = sys.argv[1]
    output_path = sys.argv[2]
    generate_image(prompt, output_path)
