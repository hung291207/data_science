import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

REQUIRED = ["Title", "Price", "Availability", "Rating", "Link",
            "Thumbnail URL", "Thumbnail File Name"]


def read_file(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="error")
    except (FileNotFoundError, pd.errors.ParserError,
            UnicodeDecodeError, OSError) as error:
        print(f"Could not read CSV: {error}")
        return None


def check_columns(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    for col in missing:
        print(f"Column missing: {col}")
    return not missing


def check_price(df):
    price = pd.to_numeric(df["Price"], errors="coerce")
    bad = df.index[price.isna() | (price <= 0)]
    for i in bad:
        print(f"Row {i}: invalid price {df.at[i, 'Price']!r}")
    return len(bad) == 0


def check_duplicate(df):
    dupes = df[df.duplicated(subset="Link", keep=False)]
    if dupes.empty:
        return True
    print(f"{dupes['Link'].nunique()} duplicated links:")
    print(dupes.sort_values("Link")[["Title", "Link"]].to_string())
    return False


def check_empty(df):
    cols = df[REQUIRED]
    blank = cols.isna() | cols.astype(str).apply(lambda c: c.str.strip() == "")
    hits = blank.stack()
    hits = hits[hits].index
    for row, col in hits:
        print(f"Row {row}: empty {col}")
    return len(hits) == 0


def check_thumbnail_files(df):
    """Local check: no network needed."""
    ok = True
    for row, path in df["Thumbnail File Name"].items():
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            print(f"Row {row}: image missing or empty: {path}")
            ok = False
    return ok

def check_urls(df, column, workers=20):
    def check(item):
        index, link = item
        try:
            r = requests.get(str(link).strip(), timeout=10)
            if r.status_code != 200:
                print(f"Corrupted link at {index}")
                return False
        except requests.RequestException:
            print(f"Corrupted link at {index}")
            return False
        return True

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(check, df[column].items()))

    return all(results)


if __name__ == "__main__":
    df = read_file("data_sheet/books_data.csv")
    if df is None or not check_columns(df):
        raise SystemExit(1)

    print("Errors detected:")
    check_empty(df)
    check_price(df)
    check_duplicate(df)
    check_thumbnail_files(df)
    check_urls(df, "Link")