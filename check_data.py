
import pandas as pd
from PIL import Image


if __name__ == "__main__":
    books = pd.read_csv("data_sheet/books_data.csv")
    train = pd.read_csv("data_sheet/train.csv")
    test = pd.read_csv("data_sheet/test.csv")
    features = pd.read_csv("data_sheet/features.csv")

    book_links = set(books["Link"])
    train_links = set(train["Link"])
    test_links = set(test["Link"])

    for link in train_links:
        assert link not in test_links
    assert train_links.union(test_links) == book_links
    assert len(train) + len(test) == len(books)
    assert abs(len(test) - len(books) * 0.2) < 1
    assert books["Link"].is_unique

    for description in books["Description"]:
        assert pd.notna(description)
        assert str(description).strip() != ""

    for price in books["Price"]:
        assert price > 0

    for category in books["Category"].unique():
        total = (books["Category"] == category).sum()
        test_count = (test["Category"] == category).sum()
        assert abs(test_count - total * 0.2) < 1

    assert len(features) == len(books)
    assert features["Link"].is_unique
    assert features.isna().sum().sum() == 0

    feature_train = features[features["split"] == "train"]
    feature_test = features[features["split"] == "test"]
    assert set(feature_train["Link"]) == train_links
    assert set(feature_test["Link"]) == test_links

    book_lookup = books.set_index("Link")
    for i in range(len(features)):
        link = features.loc[i, "Link"]
        category = book_lookup.loc[link, "Category"]
        description = book_lookup.loc[link, "Description"]

        if category == "Fiction":
            expected_label = 0
        elif category == "Nonfiction":
            expected_label = 1
        else:
            expected_label = None

        assert features.loc[i, "label"] == expected_label
        assert features.loc[i, "Description"] == description

    for path in books["Thumbnail File Name"]:
        with Image.open(path) as image:
            image.verify()

    print("Data checks passed:", len(books), "books,", len(train), "train,", len(test), "test")
