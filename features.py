import re

import numpy as np
import pandas as pd
from PIL import Image

# The website writes ratings as words -> convert to numbers
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

# Category names become 0 and 1.
LABEL_MAP = {"Fiction": 0, "Nonfiction": 1}

# The three colours of an image: red, green, blue.
COLOURS = ["R", "G", "B"]

# Each channel is split into this many groups for the histogram.
N_BINS = 4


def load_data(path):
    #Read CSV file  and prepare for analysis

    # Read the CSV file into a table
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Remove duplicated books
    df = df.drop_duplicates(subset="Link")

    # renumber after removing duplicates since it will have gaps
    df = df.reset_index(drop=True)

    return df

#Creat simple features
def basic_features(df):

    features = pd.DataFrame()

    # Copy the price
    features["price"] = df["Price"]

    # Rating: change the words to numbers
    features["rating"] = df["Rating"].map(RATING_MAP)

    # Title length
    features["title_length"] = df["Title"].str.len()

    # Two features that need a loop: number of words, and "has a digit?"
    title_words = []
    title_has_digit = []
    for title in df["Title"]:
        # Split the title at spaces and count pieces
        words = title.split()
        title_words.append(len(words))

        # Digit in the title? 1 = yes, 0 = no
        if re.search(r"\d", title):
            title_has_digit.append(1)
        else:
            title_has_digit.append(0)

    features["title_words"] = title_words
    features["title_has_digit"] = title_has_digit

    return features


def image_features(path):
    #Turn image into numbers

    # Open the image and make sure it has exactly 3 colour channels (RGB).
    image = Image.open(path).convert("RGB")

    # Turn the image into a grid of numbers: height x width x 3
    pixels = np.array(image)

    features = {}

    # Go through the colours one by one: 0 = red, 1 = green, 2 = blue
    for i in range(3):
        color = COLOURS[i]
        channel = pixels[:, :, i] #All rows, all columns, colour i           # only this colour of all pixels

        # How much of this colour the cover has on average
        features["mean_" + color] = channel.mean()

        # How many pixels fall in each group
        counts, edges = np.histogram(channel, bins=N_BINS, range=(0, 256))

        # Make big and small images comparable
        shares = counts / channel.size

        for j in range(N_BINS):
            features["hist_" + color + "_" + str(j)] = shares[j]

    # Brightness = average of all pixel values of all colours (R, G, B)
    features["brightness"] = pixels.mean()

    return features


def all_image_features(df):
    # Return a table with one row per book after running image_features()

    rows = []
    for path in df["Thumbnail File Name"]:
        rows.append(image_features(path))   # One dictionary per book

    # A list of dictionaries becomes a table
    return pd.DataFrame(rows)


if __name__ == "__main__":
    train = load_data("data_sheet/train.csv")
    test = load_data("data_sheet/test.csv")
    df = pd.concat([train, test], ignore_index=True)

    
    basic = basic_features(df)
    images = all_image_features(df)

    # Keep raw text for later text features. No vocabulary is fitted here.
    result = pd.concat([df[["Title", "Description", "Link"]], basic, images], axis=1)

    # Add target: 0 = Fiction, 1 = Nonfiction
    result["label"] = df["Category"].map(LABEL_MAP)
    result["split"] = ["train"] * len(train) + ["test"] * len(test)

    # Check if there are empty cells
    print("Empty cells:", result.isna().sum().sum())

    result.to_csv("data_sheet/features.csv", index=False)
    print("Shape (rows, columns):", result.shape)
    print(result.head())
    print("Saved to data_sheet/features.csv")
