import json
import random
from collections import defaultdict
from pathlib import Path


def build_artist_dict(dataset_path: str = "./dataset") -> dict:
    artist_dict = defaultdict(list)

    for file in Path(dataset_path).glob("*.*"):  # match all files with extensions
        stem = file.stem
        if "_" in stem:
            artist_part, art_part = stem.split("_", 1)
            artist = artist_part.replace("-", " ").title()
            artwork = art_part.replace("-", " ").title()
            if artwork not in artist_dict[artist]:  # prevent duplicates
                artist_dict[artist].append(artwork)
        else:
            # If no underscore, treat the whole name as artwork with "Unknown Artist"
            artwork = stem.replace("-", " ").title()
            if artwork not in artist_dict["Unknown Artist"]:
                artist_dict["Unknown Artist"].append(artwork)

    return dict(artist_dict)

def save_artist_dict(artist_dict: dict, filename: str = "artist_dict.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(artist_dict, f, ensure_ascii=False, indent=2)

def load_artist_dict(filename: str = "artist_dict.json") -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def get_random_artist(artist_dict: dict) -> str:
    """Pick a random artist name from the dictionary."""
    return random.choice(list(artist_dict.keys()))

def get_random_artwork(artist_dict: dict) -> tuple:
    """Pick a random artwork and return (artist, artwork)."""
    if not artist_dict:
        return None, None

    artist = get_random_artist(artist_dict)
    artworks = artist_dict.get(artist, [])
    if not artworks:
        return artist, None

    artwork = random.choice(artworks)
    return artist, artwork
