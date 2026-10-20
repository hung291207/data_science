import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parents[1] / "data"

def clean_split():
    books = pd.read_csv(DATA / "books.csv", keep_default_na=False, dtype={"book_id": str})
    if not books.book_id.is_unique or not books.url.is_unique:
        raise ValueError("Duplicate books need checking")
    if set(books.category) != {"Fiction", "Nonfiction"}:
        raise ValueError("Unexpected book categories")
    for column in ["title", "description"]:
        books[column] = books[column].str.replace(r"\s+", " ", regex=True).str.strip()
    books["price"] = pd.to_numeric(books.price.str.replace("£", "", regex=False))
    books["rating"] = books.rating.map({"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5})
    if (books.title.eq("").any() or not np.isfinite(books.price).all() or not np.isfinite(books.rating).all() or not books.price.gt(0).all() or not books.rating.between(1, 5).all()):
        raise ValueError("Missing title or invalid price/rating needs checking")
    for path in books.image_file:
        with Image.open(DATA / path) as image:
            image.verify()

    train_ids, test_ids = train_test_split(
        books.book_id, test_size=0.2, stratify=books.category, random_state=42
    )
    split = books[["book_id"]].copy()
    split["split"] = np.where(books.book_id.isin(train_ids), "train", "test")
    split_file = DATA / "split_ids.csv"
    if split_file.exists() and not pd.read_csv(split_file, dtype=str).equals(split):
        raise ValueError("Saved split changed. Check before replacing it")
    split.to_csv(split_file, index=False)
    books.to_csv(DATA / "books_clean.csv", index=False)
    counts = books.assign(split=split["split"]).groupby(["split", "category"]).size().unstack()
    checks = {
        "random_state": 42,
        "split_counts": counts.to_dict(orient="index"),
        "missing_descriptions_kept_as_empty": int(books.description.eq("").sum()),
        "removed_rows": 0,
        "price_currency": "GBP",
    }
    (DATA / "clean_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(counts)

if __name__ == "__main__":
    clean_split()