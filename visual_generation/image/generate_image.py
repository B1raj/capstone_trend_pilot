# Stable Diffusion Local Image Generator

"""
This script takes a text prompt and generates an image using a local Stable Diffusion model.
Heavy ML libraries are imported lazily inside `generate_image` to keep module import lightweight.
"""

import sys

def generate_image(prompt: str, output_path: str, model_path: str = "CompVis/stable-diffusion-v1-4", disable_safety: bool = True):
    # Import heavy ML libraries lazily so importing this module remains lightweight
    try:
        from diffusers import StableDiffusionPipeline
        import torch
    except Exception as e:
        # Re-raise with clearer message for callers
        raise RuntimeError(f"Missing image generation dependencies: {e}")

    # Use float16 if CUDA is available, else float32 for CPU
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"  # force CPU for compatibility; change to "cuda" if GPU is available and configured
    pipe = pipe.to(device)
    # Optionally disable the built-in NSFW safety checker which replaces images
    # with a black image when content is flagged. Use only when you understand
    # the safety/legal implications of bypassing model filters.
    if disable_safety:
        try:
            # Newer versions of diffusers attach a `safety_checker`; setting to
            # None disables the check. Wrap in try/except for compatibility.
            pipe.safety_checker = None
        except Exception:
            pass
    # Use fewer steps and smaller image for speed
    # image = pipe(prompt, num_inference_steps=15, height=240, width=240).images[0]
    image = pipe(prompt).images[0]
    # save the PIL image
    image.save(output_path)
    print(f"Image saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_image.py <prompt> <output.png> [disable_safety]")
        print("Set third arg to '1' to disable the safety checker (use with caution).")
        sys.exit(1)
    prompt = sys.argv[1]
    output_path = sys.argv[2]
    disable = False
    if len(sys.argv) >= 4 and sys.argv[3] in ("1", "true", "True"):
        disable = True
    generate_image(prompt, output_path, disable_safety=disable)
