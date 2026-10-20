from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"

def colour_histogram(path):
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    counts = [np.histogram(pixels[:, :, channel], bins=8, range=(0, 256))[0]
              for channel in range(3)]
    return np.concatenate(counts) / (pixels.shape[0] * pixels.shape[1])

def build_features():
    books = pd.read_csv(DATA / "books_clean.csv", keep_default_na=False, dtype={"book_id": str})
    colours = []
    greyscale = []
    for path in books.image_file:
        colours.append(colour_histogram(DATA / path))
        with Image.open(DATA / path) as image:
            grey = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
            greyscale.append(np.asarray(grey).ravel() / 255.0)

    # Each cover is processed separately. Nothing is fitted across books here.
    colour_columns = [f"colour_{i}" for i in range(24)]
    grey_columns = [f"grey_{i}" for i in range(1024)]
    features = books[["book_id", "category", "price", "rating"]].copy()
    features["text"] = books.title + " " + books.description
    features = pd.concat([features, pd.DataFrame(colours, columns=colour_columns),
                          pd.DataFrame(greyscale, columns=grey_columns)], axis=1)
    if not np.allclose(np.array(colours).reshape(-1, 3, 8).sum(axis=2), 1):
        raise ValueError("Colour histogram proportions do not sum to one")
    if not np.isfinite(features[colour_columns + grey_columns].to_numpy()).all():
        raise ValueError("Invalid image features")
    features.to_csv(DATA / "book_features.csv", index=False)
    print(f"Saved {len(features)} books: text, 24 colour bins, 1024 greyscale pixels, price/rating")

if __name__ == "__main__":
    build_features()