import argparse
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from PIL import Image, ImageFile, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.amp.autocast()
print(f"is available: {torch.cuda.is_available()}\n")   # should return True
print(f"version: {torch.version.cuda}\n")          # should say 12.4 (or matching version)
print(f"device name: {torch.cuda.get_device_name(0)}\n")

gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
print(f"GPU memory: {gpu_mem:.2f} GB")
parser = argparse.ArgumentParser(description="Training with configurable batch size")
parser.add_argument("--batch-size", type=int, required=True,
                    help="Batch size to use for training (e.g. 1 or 2)")
args = parser.parse_args()

train_batch_size = args.batch_size
if gpu_mem < 20: 
  train_batch_size = 1

print(f"Using batch size: {train_batch_size}")

# your label parser
def parse_label(filename: str) -> str:
    stem = Path(filename).stem
    if "_" in stem:
        artist_part, art_part = stem.split("_", 1)
        artist = artist_part.replace("-", " ").title()
        artwork = art_part.replace("-", " ").title()
        label = f"{artwork}, {artist}"
    else:
        label = stem.replace("-", " ").title()
    return label

# dataset
class ImageCaptionDataset(Dataset):
    def __init__(self, root, tokenizer, size=512):
        self.files = list(Path(root).glob("*.jpg"))
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        try:
            image = Image.open(f).convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            print(f"Warning: Skipping corrupted image {f} ({e})\nFalling back to next image {self.files[i+1]}.\n")
            return self.__getitem__((i + 1) % len(self.files))
        image = self.transform(image)
        caption = parse_label(f.name)
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        return {
            "pixel_values": image,
            "input_ids": tokens.input_ids.squeeze(0),
            "attention_mask": tokens.attention_mask.squeeze(0)
        }

# load base model
model_dir = Path("./fine_tuned_sd2_1")
# model_dir = Path("/home/droodt1/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-base/snapshots/1f758383196d38df1dfe523ddb1030f2bfab7741")
model_id = "stabilityai/stable-diffusion-2-1-base"
tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")

pipeline = StableDiffusionPipeline.from_pretrained(
    model_dir,
    torch_dtype=None,
    safety_checker=None,
    use_safetensors=True
)
pipeline.to("cuda")
text_encoder = pipeline.text_encoder
unet = pipeline.unet
vae = pipeline.vae

# freeze unnecessary parts
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# training setup
dataset = ImageCaptionDataset("./dataset", tokenizer)
dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-6, fused=True)
lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0)
noise_scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

accelerator = Accelerator(mixed_precision="fp16")
unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, dataloader, lr_scheduler
)

unet.enable_gradient_checkpointing()
try:
    unet.enable_xformers_memory_efficient_attention()
    print("xformers enabled ?")
except Exception:
    print("xformers not available ?")

# training loop
epochs = 5
for epoch in range(epochs):
    progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in progress:
        with accelerator.accumulate(unet):
            pixel_values = batch["pixel_values"].to(accelerator.device)
            
            latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps,
                                      (latents.shape[0],), device=latents.device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["input_ids"].to(accelerator.device))[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            loss = nn.functional.mse_loss(model_pred, noise)
            accelerator.backward(loss)

            torch.cuda.empty_cache()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        progress.set_postfix({"loss": loss.item()})

# save model
accelerator.wait_for_everyone()
unet = accelerator.unwrap_model(unet)
pipeline.save_pretrained("./fine_tuned_sd2_1")

prompt = "Vincent Van Gogh, Vase With Carnations And Other Flowers 1886"
num_images = 4
images = pipeline(prompt, num_images_per_prompt=num_images).images

cols = math.isqrt(num_images)
rows = (num_images + cols - 1) // cols  # ceiling division
w, h = images[0].size
grid = Image.new("RGB", (cols * w, rows * h))

for i, img in enumerate(images):
    grid.paste(img, box=((i % cols) * w, (i // cols) * h))

grid.save(f"{prompt}_grid.png")
print(f"Saved {num_images} images as a grid: {prompt}_grid.png")
