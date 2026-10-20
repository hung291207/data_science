import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler

from features import colour_histogram

DATA = Path(__file__).resolve().parents[1] / "data"
ORDER = ["Fiction", "Nonfiction"]


def analyse():
    output = DATA / "eda"
    output.mkdir(exist_ok=True)
    books = pd.read_csv(DATA / "books_clean.csv", keep_default_na=False, dtype={"book_id": str})
    split = pd.read_csv(DATA / "split_ids.csv", dtype=str)
    train_ids = split.loc[split.split.eq("train"), "book_id"]
    train = books.loc[books.book_id.isin(train_ids)].copy()
    if len(train) != len(train_ids):
        raise ValueError("Training IDs do not match cleaned data")
    train["description_words"] = train.description.str.split().str.len()
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    sns.countplot(data=train, x="category", order=ORDER, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_ylabel("Books")
    for column, label, ax in zip(
        ["price", "rating", "description_words"],
        ["Price (GBP)", "Rating (1–5)", "Description words"], axes.flat[1:]
    ):
        sns.boxplot(data=train, x="category", y=column, order=ORDER, ax=ax, color="lightblue")
        ax.set_ylabel(label)
    for ax in axes.flat:
        ax.set_xlabel("")
    fig.suptitle(f"Training set only (n={len(train)})")
    fig.tight_layout()
    fig.savefig(output / "training_distributions.png", dpi=160)
    plt.close(fig)

    vectorizer = CountVectorizer(stop_words="english")
    words = vectorizer.fit_transform(train.title + " " + train.description)
    terms = vectorizer.get_feature_names_out()
    top_terms, class_counts = {}, {}
    for category in ORDER:
        counts = np.asarray(words[train.category.eq(category).to_numpy()].sum(axis=0)).ravel()
        class_counts[category] = counts
        share = counts / counts.sum()
        top = np.argsort(counts)[-15:][::-1]
        top_terms[category] = [{"term": terms[i], "count": int(counts[i]),
                                "token_share": float(share[i])} for i in top]
    (output / "top_terms.json").write_text(json.dumps(top_terms, indent=2), encoding="utf-8")
    distinctive = {}
    for category, other in (("Fiction", "Nonfiction"), ("Nonfiction", "Fiction")):
        counts, other_counts = class_counts[category], class_counts[other]
        difference = counts / counts.sum() - other_counts / other_counts.sum()
        candidates = np.flatnonzero((counts + other_counts >= 3) & (difference > 0))
        top = candidates[np.argsort(difference[candidates])[-10:][::-1]]
        distinctive[category] = [{"term": terms[i], "token_share_difference": float(difference[i])}
                                 for i in top]
    (output / "distinctive_terms.json").write_text(json.dumps(distinctive, indent=2), encoding="utf-8")

    means = []
    for path in train.image_file:
        with Image.open(DATA / path) as image:
            means.append(np.asarray(image.convert("RGB")).mean(axis=(0, 1)))
    colours = pd.DataFrame(means, columns=["red", "green", "blue"], index=train.index)
    mean_colour = colours.assign(category=train.category).groupby("category").mean()
    mean_colour.to_csv(output / "mean_cover_colours.csv")
    histograms = np.stack([colour_histogram(DATA / path) for path in train.image_file])
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(scaler.fit_transform(histograms))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mean_colour.reindex(ORDER).plot.bar(ax=axes[0], color=["#b34b4b", "#559455", "#4a6eaf"])
    axes[0].set_ylabel("Mean channel value (0–255)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=0)
    for category in ORDER:
        mask = train.category.eq(category).to_numpy()
        axes[1].scatter(coordinates[mask, 0], coordinates[mask, 1], label=category, alpha=0.7, s=25)
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    axes[1].legend()
    fig.suptitle("Cover features: training set only")
    fig.tight_layout()
    fig.savefig(output / "training_covers_pca.png", dpi=160)
    plt.close(fig)
    summary = {
        "training_rows": len(train), "counts": train.category.value_counts().to_dict(),
        "medians": train.groupby("category")[["price", "rating", "description_words"]].median().to_dict(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    analyse()
