import math
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU is available")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU memory: {gpu_mem:.2f} GB")
else:
    print("GPU not available, using CPU")

class GenAIModel:
    def __init__(self, model_dir: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            use_safetensors=True
        ).to(self.device)

    def generate_images(self, prompt, num_images=4):
        images = self.pipeline(prompt, num_images_per_prompt=num_images).images

        cols = math.isqrt(num_images)
        rows = (num_images + cols - 1) // cols  # ceiling division
        w, h = images[0].size
        grid = Image.new("RGB", (cols * w, rows * h))

        for i, img in enumerate(images):
            grid.paste(img, box=((i % cols) * w, (i // cols) * h))

        grid.save(f"./GeneratedImages/{prompt}.png")
        print(f"Saved {num_images} images as a grid: {prompt}.png")
        return images