import math
import os
import random
from pathlib import Path

from dict import get_random_artist, get_random_artwork, load_artist_dict
from GenAI import GenAIModel
from PIL import Image, ImageDraw, ImageFont

# === CONFIG ===
DATASET_PATH = "./dataset"
OUTPUT_DIR = "./output"
POSTER_PATH = "./poster_a0.jpg"
NUM_PAIRS = 12

# === A0 in pixels at 300 DPI ===
A0_WIDTH, A0_HEIGHT = 9933, 14043


def format_filename(artist: str, artwork: str, dataset_path: str) -> Path | None:
    """Rebuild original filename from artist/artwork according to dataset convention."""
    artist_part = artist.lower().replace(" ", "-")
    artwork_part = artwork.lower().replace(" ", "-")

    for ext in ("jpg", "jpeg", "png", "webp"):
        path = Path(dataset_path) / f"{artist_part}_{artwork_part}.{ext}"
        if path.exists():
            return path
    return None


def generate_and_save_pairs(gen_ai: GenAIModel, artist_dict: dict, output_dir: str, dataset_path: str, num_pairs: int):
    os.makedirs(output_dir, exist_ok=True)
    pairs = []  # (original_path, generated_path, artwork)

    for _ in range(num_pairs):
        artist, artwork = get_random_artwork(artist_dict)
        if not artist or not artwork:
            continue

        original_path = format_filename(artist, artwork, dataset_path)
        if not original_path or not original_path.exists():
            print(f"⚠️ Could not find original file for {artist} - {artwork}")
            continue

        prompt = artwork + " by " + artist;
        print(f"🎨 Generating image for '{prompt}'...")
        result = gen_ai.generate_images(prompt, num_images=1)

        # Handle model output (single PIL.Image or list)
        generated_img = result[0] if isinstance(result, list) else result
        if not isinstance(generated_img, Image.Image):
            raise TypeError("GenAIModel.generate_images() must return a PIL Image or list of PIL Images")

        # Save original and generated
        original_img = Image.open(original_path).convert("RGB")

        gen_file = Path(output_dir) / f"{artist}_{artwork}_generated.jpg"
        orig_file = Path(output_dir) / f"{artist}_{artwork}_original.jpg"

        generated_img.save(gen_file)
        original_img.save(orig_file)
        pairs.append((orig_file, gen_file, artwork))

    return pairs


def build_a0_poster(pairs, output_path: str):
    cols = math.ceil(math.sqrt(len(pairs) * 2))  # two cells per artwork
    rows = math.ceil((len(pairs) * 2) / cols)
    cell_w = A0_WIDTH // cols
    cell_h = A0_HEIGHT // rows

    poster = Image.new("RGB", (A0_WIDTH, A0_HEIGHT), "white")
    draw = ImageDraw.Draw(poster)
    try:
        font = ImageFont.truetype("arial.ttf", 60)
    except:
        font = ImageFont.load_default()

    x, y = 0, 0
    for orig, gen, artwork in pairs:
        for img_path, label in [(orig, "Original"), (gen, "Generated")]:
            img = Image.open(img_path).convert("RGB")
            
            aspect_img = img.width / img.height
            aspect_cell = cell_w / (cell_h - 100)

            if aspect_img > aspect_cell:
                # Image is wider — crop horizontally
                new_height = cell_h - 100
                new_width = int(new_height * aspect_img)
            else:
                # Image is taller — crop vertically
                new_width = cell_w
                new_height = int(new_width / aspect_img)

            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Center crop to fill the cell
            left = (img_resized.width - cell_w) // 2
            top = (img_resized.height - (cell_h - 100)) // 2
            right = left + cell_w
            bottom = top + (cell_h - 100)
            img_cropped = img_resized.crop((left, top, right, bottom))

            cell = Image.new("RGB", (cell_w, cell_h), "white")
            cell.paste(img_cropped, (0, 0))

            draw = ImageDraw.Draw(cell)
            text = f"{artwork} ({label})"
            draw.text((20, cell_h - 80), text, fill="black", font=font)

            # Paste into poster
            poster.paste(cell, (x, y))
            x += cell_w
            if x + cell_w > A0_WIDTH:
                x = 0
                y += cell_h

    poster.save(output_path, "JPEG", quality=95)
    print(f"✅ Poster saved to {output_path}")


if __name__ == "__main__":
    model_dir = Path("./fine_tuned_sd2_1")
    artist_dict = load_artist_dict("artist_dict.json")
    gen_ai = GenAIModel(model_dir)
    pairs = generate_and_save_pairs(gen_ai, artist_dict, OUTPUT_DIR, DATASET_PATH, NUM_PAIRS)
    build_a0_poster(pairs, POSTER_PATH)
