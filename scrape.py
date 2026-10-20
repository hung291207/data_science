import json
import time
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

DATA = Path(__file__).resolve().parents[1] / "data"
BASE = "https://books.toscrape.com/"

def get_page(session, url):
    response = session.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    time.sleep(0.5)
    return BeautifulSoup(response.text, "html.parser")

def scrape():
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    home = get_page(session, BASE)
    categories = {}
    for link in home.select(".side_categories a"):
        name = link.get_text(strip=True)
        if name in ["Fiction", "Nonfiction"]:
            categories[name] = urljoin(BASE, link["href"])
    if len(categories) != 2:
        raise ValueError("Could not find both category links")

    rows = []
    expected = {}
    for category, listing_url in categories.items():
        while listing_url:
            listing = get_page(session, listing_url)
            expected[category] = int(listing.select_one("form strong").text)
            for article in listing.select("article.product_pod"):
                url = urljoin(listing_url, article.select_one("h3 a")["href"])
                book = get_page(session, url)
                main = book.select_one(".product_main")
                actual_category = book.select(".breadcrumb li")[-2].get_text(strip=True)
                if actual_category != category:
                    raise ValueError(f"Category mismatch at {url}")
                table = {row.th.text.strip(): row.td.text.strip()
                         for row in book.select("table.table tr")}
                book_id = table["UPC"]
                image_file = f"images/{book_id}.jpg"
                image_path = DATA / image_file
                if not image_path.exists():
                    image_url = urljoin(url, book.select_one(".item.active img")["src"])
                    response = session.get(image_url, timeout=30)
                    response.raise_for_status()
                    image_path.write_bytes(response.content)
                    time.sleep(0.5)
                with Image.open(image_path) as image:
                    image.verify()
                heading = book.select_one("#product_description")
                description = heading.find_next_sibling("p") if heading else None
                rows.append({
                    "book_id": book_id,
                    "title": main.h1.get_text(strip=True),
                    "category": category,
                    "price": main.select_one(".price_color").text.strip(),
                    "rating": main.select_one(".star-rating")["class"][1],
                    "description": description.get_text(" ", strip=True) if description else "",
                    "image_file": image_file,
                    "url": url,
                })
            print(category, sum(row["category"] == category for row in rows), flush=True)
            next_link = listing.select_one("li.next a")
            listing_url = urljoin(listing_url, next_link["href"]) if next_link else None

    books = pd.DataFrame(rows).sort_values("book_id").reset_index(drop=True)
    counts = books.groupby("category").size().to_dict()
    if not books.book_id.is_unique or not books.url.is_unique or counts != expected:
        raise ValueError("Duplicate books or incomplete pagination")
    books.to_csv(DATA / "books.csv", index=False)
    checks = {
        "counts": counts,
        "advertised_counts": expected,
        "duplicate_titles": int(books.title.duplicated().sum()),
        "missing_descriptions": int(books.description.eq("").sum()),
        "verified_images": len(books),
        "source": BASE,
        "scraped_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    (DATA / "scrape_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(checks)

if __name__ == "__main__":
    scrape()
