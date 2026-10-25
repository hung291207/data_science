#try the different feature sets and see which works best
import pandas as pd
import os
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":
    df = pd.read_csv("data_sheet/features.csv")
    df = df[df["split"] == "train"].reset_index(drop=True)
    y = df["label"]

    print("Training shape:", df.shape)
    print(y.value_counts())
    df["text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")

    #different combinations of features
    feature_sets = ["baseline", "text_df1", "text_df2", "text_df3",
                    "image", "numeric", "text+image", "all"]
    simplicity = ["baseline", "numeric", "image", "text_df3", "text_df2",
                  "text_df1", "text+image", "all"]

    image_columns = ["mean_R", "mean_G", "mean_B"]
    for col in df.columns:
        if col.startswith("hist_"):
            image_columns.append(col)
    image_columns.append("brightness")
    numeric_columns = ["price", "rating"]

    #same split for each test
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(cv.split(df, y))

    rows = []
    for name in feature_sets:
        print("\nFeature set:", name)
        balanced_scores = []
        f1_scores = []
        accuracy_scores = []
        vocab_sizes = []

        for fold, (train_index, validation_index) in enumerate(folds, 1):
            train_rows = df.iloc[train_index]
            validation_rows = df.iloc[validation_index]
            y_train = y.iloc[train_index]
            y_validation = y.iloc[validation_index]
            vocab_size = 0

            #baseline just guesses the most common label
            if name == "baseline":
                x_train = np.zeros((len(train_rows), 1))
                x_validation = np.zeros((len(validation_rows), 1))
                model = DummyClassifier(strategy="most_frequent")
            else:
                train_parts = []
                validation_parts = []

                #text features
                if name in ["text_df1", "text_df2", "text_df3", "text+image", "all"]:
                    if name == "text_df1":
                        min_df = 1
                    elif name == "text_df2":
                        min_df = 2
                    else:
                        min_df = 3

                    vectorizer = CountVectorizer(stop_words="english", min_df=min_df)
                    text_train = vectorizer.fit_transform(train_rows["text"])
                    text_validation = vectorizer.transform(validation_rows["text"])
                    train_parts.append(text_train)
                    validation_parts.append(text_validation)
                    vocab_size = len(vectorizer.vocabulary_)

                #image data
                if name in ["image", "text+image", "all"]:
                    image_scaler = StandardScaler()
                    image_train = image_scaler.fit_transform(train_rows[image_columns])
                    image_validation = image_scaler.transform(validation_rows[image_columns])
                    train_parts.append(csr_matrix(image_train))
                    validation_parts.append(csr_matrix(image_validation))

                #price and rating
                if name in ["numeric", "all"]:
                    numeric_scaler = StandardScaler()
                    numeric_train = numeric_scaler.fit_transform(train_rows[numeric_columns])
                    numeric_validation = numeric_scaler.transform(validation_rows[numeric_columns])
                    train_parts.append(csr_matrix(numeric_train))
                    validation_parts.append(csr_matrix(numeric_validation))

                x_train = hstack(train_parts, format="csr")
                x_validation = hstack(validation_parts, format="csr")
                model = LogisticRegression(class_weight="balanced", max_iter=1000,
                                           random_state=42)

            vocab_sizes.append(vocab_size)
            print("Fold", fold, ":", len(train_index), "train /",
                  len(validation_index), "validation")

            model.fit(x_train, y_train)
            prediction = model.predict(x_validation)
            balanced_scores.append(balanced_accuracy_score(y_validation, prediction))
            f1_scores.append(f1_score(y_validation, prediction, average="macro"))
            accuracy_scores.append(accuracy_score(y_validation, prediction))

        #get average scores
        row = {
            "feature_set": name,
            "vocab_size_mean": np.mean(vocab_sizes),
            "balanced_accuracy_mean": np.mean(balanced_scores),
            "balanced_accuracy_std": np.std(balanced_scores, ddof=1),
            "macro_f1_mean": np.mean(f1_scores),
            "macro_f1_std": np.std(f1_scores, ddof=1),
            "accuracy_mean": np.mean(accuracy_scores),
            "accuracy_std": np.std(accuracy_scores, ddof=1)
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values("balanced_accuracy_mean", ascending=False)

    #if scores are close i would rather use less features
    best = result.iloc[0]
    threshold = best["balanced_accuracy_mean"] - best["balanced_accuracy_std"]
    eligible = result[result["balanced_accuracy_mean"] >= threshold]
    for name in simplicity:
        if name in eligible["feature_set"].values:
            chosen = name
            break
    result["selected"] = result["feature_set"] == chosen

    print("\nCV results:")
    for _, row in result.iterrows():
        print(row["feature_set"])
        print(" balanced acc:", round(row["balanced_accuracy_mean"], 3),
              "+/-", round(row["balanced_accuracy_std"], 3))
        print(" f1:", round(row["macro_f1_mean"], 3),
              "accuracy:", round(row["accuracy_mean"], 3))
        print(" avg vocab:", round(row["vocab_size_mean"], 1))

    print("Best score:", best["feature_set"])
    print("Selected:", chosen)

    os.makedirs("results", exist_ok=True)
    result.to_csv("results/cv_results.csv", index=False)
    print("saved results/cv_results.csv")
