import datasets
import pandas as pd
from datasets import Dataset, Features, Image, Value

# Load dataset.txt
df = pd.read_csv("dataset.txt", names=["path", "label"])

# Prepend wikiart folder path
df["path"] = "wikiart/" + df["path"]

# Use filename or class as the caption text
df["text"] = df["path"].apply(lambda x: x.split("/")[-1].replace("_", " "))

# OR if you want class number instead of filename, uncomment:
# df["text"] = df["label"].astype(str)

# Create Hugging Face Dataset
features = Features({
    "image": datasets.Image(),
    "text": Value("string"),
})

df = df.rename(columns={"path": "image"})
dataset = Dataset.from_pandas(df[["image", "text"]], features=features)

# Save locally so you can reload easily
dataset.save_to_disk("wikiart_dataset")
print("Dataset saved to wikiart_dataset/")