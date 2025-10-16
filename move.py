import os
import shutil

# Paths
dataset_file = "dataset.txt"
wikiart_root = "wikiart"
output_folder = "dataset"

# Make sure dataset folder exists
os.makedirs(output_folder, exist_ok=True)

# Read dataset.txt
with open(dataset_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # skip empty lines

        try:
            relative_path, _ = line.split(",", 1)
            relative_path = relative_path.strip()
        except ValueError:
            print(f"Skipping invalid line: {line}")
            continue

        # Build source and destination paths
        src_path = os.path.join(wikiart_root, relative_path)
        filename = os.path.basename(relative_path)
        dst_path = os.path.join(output_folder, filename)

        # Copy if exists
        if os.path.exists(src_path):
            try:
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} -> {dst_path}")
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
        else:
            print(f"File not found: {src_path}")
