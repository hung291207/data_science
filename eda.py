import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Two categories
CATEGORIES = ["Fiction", "Nonfiction"]

# convert the website ratings to numbers.
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

# common words 
STOP_WORDS = ["the", "a", "an", "of", "and", "to", "in", "for", "on", "with",
              "is", "from", "by", "at", "my", "your", "you", "i", "it", "or", "as"]

# Create the folder for the figures if it does not exist yet.
os.makedirs("figures", exist_ok=True)


def load_data(path):
    #Read CSV file  and prepare for analysis

    # Read the CSV file into a table
    df = pd.read_csv(path, encoding="utf-8-sig")

    # Remove books that appear twice
    rows_before = len(df)
    df = df.drop_duplicates(subset="Link")
    rows_after = len(df)
    print("Removed", rows_before - rows_after, "duplicate books")

    # renumber after removing duplicates since it will have gaps
    df = df.reset_index(drop=True)

    # Change the rating from words to number
    df["Rating"] = df["Rating"].map(RATING_MAP)

    #  how many characters each title has
    df["Title Length"] = df["Title"].str.len()

    return df

#Print basic facts
def basic_info(df):
    

    # Number of rows and columns
    print("\n Shape (rows, columns)")
    print(df.shape)

    # Column names, data types and number of non-empty values
    print("\n Column types")
    df.info()

    # Count the empty cells in every column
    print("\n Missing value")
    print(df.isna().sum())

    # How many books are in each category, and what percentage that is
    print("\nBooks per category")
    total = len(df)
    for category in CATEGORIES:
        count = len(df[df["Category"] == category])
        percent = count / total * 100
        print(category, ":", count, "books (", round(percent, 1), "% )")

    # All different values in Availability 
    print("\n Values in Availability")
    print(df["Availability"].value_counts())

#Compare Price, Rating, Title Length
def summary_by_category(df):
    

    for category in CATEGORIES:
        # Keep only the rows of this category
        subset = df[df["Category"] == category]

        print("\n" + category)
        # Average, middle value and spread for each of the three columns
        for column in ["Price", "Rating", "Title Length"]:
            mean = subset[column].mean()
            median = subset[column].median()
            std = subset[column].std()
            print("  ", column, " - mean:", round(mean, 2),
                  "median:", round(median, 2), "std:", round(std, 2))

#Print the n most common words in the titles of each category.
def common_words(df, n=10):
    for category in CATEGORIES:
        subset = df[df["Category"] == category]

        # Collect every useful word of every title in this list
        all_words = []
        for title in subset["Title"]:
            # Make lowercase, then find all words (letters only)
            words = re.findall(r"[a-z']+", title.lower())
            for word in words:
                # Skip filler words like "the" and "of"
                if word not in STOP_WORDS:
                    all_words.append(word)

        # Counter counts how many times each word appears
        word_counts = Counter(all_words)
        print(category, ":", word_counts.most_common(n))


def plot_overview(df):
    #Figure 1: three small plots next to each other.
    #(1) books per category, (2) price per category, (3) rating per category."""

    # Split the data once, so we can reuse it in the plots
    fiction = df[df["Category"] == "Fiction"]
    nonfiction = df[df["Category"] == "Nonfiction"]

    # One figure with 3 empty plots in one row
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Plot 1: bar chart with the number of books per category
    counts = [len(fiction), len(nonfiction)]
    axes[0].bar(CATEGORIES, counts)
    axes[0].set_title("Books per category")
    axes[0].set_ylabel("Number of books")

    # Plot 2: boxplot of the prices 
    axes[1].boxplot([fiction["Price"], nonfiction["Price"]])
    axes[1].set_xticks([1, 2])
    axes[1].set_xticklabels(CATEGORIES)
    axes[1].set_title("Price per category")
    axes[1].set_ylabel("Price")

    # Plot 3: histogram of the ratings, one bar colour per category.
    # The bins are chosen so each rating (1 to 5) gets its own group of bars.
    rating_bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    axes[2].hist([fiction["Rating"], nonfiction["Rating"]],
                 bins=rating_bins, label=CATEGORIES)
    axes[2].set_title("Rating per category")
    axes[2].set_xlabel("Rating")
    axes[2].set_ylabel("Number of books")
    axes[2].legend()

    # Fix the spacing and save the figure as a picture file
    plt.tight_layout()
    plt.savefig("figures/overview.png", dpi=150)
    plt.close()


def plot_samples(df, n=6):
    #Figure 2: n random book covers from each category in a grid,
    #so we can compare them with our own eyes."""

    # One row per category, one column per book
    fig, axes = plt.subplots(len(CATEGORIES), n, figsize=(2 * n, 3 * len(CATEGORIES)))

    for row in range(len(CATEGORIES)):
        category = CATEGORIES[row]
        subset = df[df["Category"] == category]

        # Pick n random books (random_state makes the choice the same every run)
        sample = subset.sample(n, random_state=0)

        for col in range(n):
            # Get the image path of this book, open it and draw it
            path = sample.iloc[col]["Thumbnail File Name"]
            image = Image.open(path)
            axes[row, col].imshow(image)
            axes[row, col].axis("off")   # hide the axis numbers

        # Write the category name above the first image of the row
        axes[row, 0].set_title(category, loc="left")

    plt.tight_layout()
    plt.savefig("figures/samples.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    # Only explore training books. Keep the test set for final evaluation.
    df = load_data("data_sheet/train.csv")
    basic_info(df)
    summary_by_category(df)
    common_words(df)
    plot_overview(df)
    plot_samples(df)
    print("\nFigures saved in figures/")
