from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data"

def make_model(parts, min_df=3):
    transforms = []
    if "text" in parts:
        transforms.append(("text", CountVectorizer(stop_words="english", min_df=min_df), "text"))
    if "colour" in parts:
        transforms.append(("colour", StandardScaler(), [f"colour_{i}" for i in range(24)]))
    if "grey" in parts:
        grey = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=20, svd_solver="randomized", random_state=42)),
        ])
        transforms.append(("grey", grey, [f"grey_{i}" for i in range(1024)]))
    if "numeric" in parts:
        transforms.append(("numeric", StandardScaler(), ["price", "rating"]))
    return Pipeline([
        ("features", ColumnTransformer(transforms)),
        ("model", LogisticRegression(C=1, class_weight="balanced", solver="liblinear",
                                     max_iter=2000, random_state=42)),
    ])

def compare_models():
    output = DATA / "results"
    output.mkdir(exist_ok=True)
    books = pd.read_csv(DATA / "book_features.csv", keep_default_na=False, dtype={"book_id": str})
    split = pd.read_csv(DATA / "split_ids.csv", dtype=str)
    books = books.merge(split, on="book_id", validate="one_to_one")
    train = books.loc[books.split.eq("train")].reset_index(drop=True)
    y = train.category
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(train, y))
    fold_ids = train[["book_id"]].copy()
    fold_ids["validation_fold"] = -1
    for number, (_, validation) in enumerate(folds, start=1):
        fold_ids.loc[validation, "validation_fold"] = number
    fold_ids.to_csv(output / "cv_folds.csv", index=False)

    models = [("majority", DummyClassifier(strategy="most_frequent"))]
    vocabulary_sizes = []
    # These sizes describe the training set. CV fits separate vectorizers below.
    for min_df in [1, 2, 3]:
        vectorizer = CountVectorizer(stop_words="english", min_df=min_df)
        vectorizer.fit(train.text)
        vocabulary_sizes.append({"min_df": min_df, "training_terms": len(vectorizer.vocabulary_)})
        models.append((f"text_df{min_df}", make_model(["text"], min_df)))
    pd.DataFrame(vocabulary_sizes).to_csv(output / "vocabulary_sizes.csv", index=False)

    for name, parts in [
        ("colour", ["colour"]),
        ("greyscale", ["grey"]),
        ("numeric", ["numeric"]),
        ("text_colour", ["text", "colour"]),
        ("text_greyscale", ["text", "grey"]),
        ("combined_colour", ["text", "colour", "numeric"]),
        ("combined_greyscale", ["text", "grey", "numeric"]),
    ]:
        models.append((name, make_model(parts)))

    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "f1_macro"}
    summary = []
    fold_results = []
    for name, model in models:
        # CV fits a fresh pipeline on each training fold, not saved fitted matrices.
        scores = cross_validate(model, train, y, cv=folds, scoring=scoring, error_score="raise")
        row = {"experiment": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[metric] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
        summary.append(row)
        for fold in range(5):
            fold_results.append({"experiment": name, "fold": fold + 1,
                                 **{metric: scores[f"test_{metric}"][fold] for metric in scoring}})
        print(f"{name}: balanced accuracy {row['balanced_accuracy']:.3f}", flush=True)
    results = pd.DataFrame(summary)
    if not np.isfinite(results.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Invalid CV scores")
    results.to_csv(output / "cv_results.csv", index=False)
    pd.DataFrame(fold_results).to_csv(output / "cv_fold_results.csv", index=False)
    print("\n", results[["experiment", "accuracy", "balanced_accuracy", "macro_f1"]].to_string(index=False))
    print("Test set not scored. These are training-only CV results.")

if __name__ == "__main__":
    compare_models()