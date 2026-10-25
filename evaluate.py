"""Evaluate the chosen text model"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score)


if __name__ == "__main__":
    # Only run final test evaluation once
    if os.path.exists("results/test_metrics.csv"):
        raise SystemExit("Test results already exist. Review saved results instead of rerunning.")

    cv_results = pd.read_csv("results/cv_results.csv")
    selected = cv_results[cv_results["selected"] == True]
    assert selected["feature_set"].tolist() == ["text_df3"]
    cv_row = cv_results[cv_results["feature_set"] == "text_df3"].iloc[0]

    df = pd.read_csv("data_sheet/features.csv")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    assert len(train) == 140
    assert len(test) == 35
    assert train["label"].value_counts().to_dict() == {0: 52, 1: 88}
    assert test["label"].value_counts().to_dict() == {0: 13, 1: 22}
    assert train["Link"].is_unique
    assert test["Link"].is_unique
    assert set(train["Link"]).isdisjoint(set(test["Link"]))

    train["text"] = train["Title"].fillna("") + " " + train["Description"].fillna("")
    test["text"] = test["Title"].fillna("") + " " + test["Description"].fillna("")
    y_train = train["label"]
    y_test = test["label"]

    print("Training / test books:", len(train), "/", len(test))
    print("Test labels:")
    print(y_test.value_counts())

    # The vocabulary is learned from the training books, never the test books
    vectorizer = CountVectorizer(stop_words="english", min_df=3)
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    print("Vocabulary size:", len(vectorizer.vocabulary_))

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]

    baseline = np.ones(len(test), dtype=int)
    cv_mean = cv_row["balanced_accuracy_mean"]
    cv_std = cv_row["balanced_accuracy_std"]
    balanced_accuracy = balanced_accuracy_score(y_test, prediction)

    metrics = {
        "feature_set": "text_df3",
        "train_books": len(train),
        "test_books": len(test),
        "vocab_size": len(vectorizer.vocabulary_),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": f1_score(y_test, prediction, average="macro"),
        "accuracy": accuracy_score(y_test, prediction),
        "roc_auc": roc_auc_score(y_test, probability),
        "baseline_balanced_accuracy": balanced_accuracy_score(y_test, baseline),
        "baseline_accuracy": accuracy_score(y_test, baseline),
        "cv_balanced_accuracy_mean": cv_mean,
        "cv_balanced_accuracy_std": cv_std,
        "within_cv_mean_plus_minus_sd": cv_mean - cv_std <= balanced_accuracy <= cv_mean + cv_std
    }

    print("\nMetrics:")
    for name in metrics:
        print(name, ":", metrics[name])
    print("CV mean +/- SD describes fold spread, not a confidence interval.")

    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    labelled_matrix = pd.DataFrame(matrix, index=["True Fiction", "True Nonfiction"], columns=["Predicted Fiction", "Predicted Nonfiction"])
    print("\nConfusion matrix:")
    print(labelled_matrix)

    report = classification_report(y_test, prediction, labels=[0, 1], target_names=["Fiction", "Nonfiction"], zero_division=0)
    print("\n" + report)

    os.makedirs("results", exist_ok=True)
    labelled_matrix.to_csv("results/confusion_matrix.csv")
    with open("results/classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    # save all predictions and the incorrectly classified books.
    predictions = test[["Title", "Description", "Link"]].copy()
    predictions["true_label"] = y_test
    predictions["predicted_label"] = prediction
    predictions["probability"] = probability
    predictions["confidence"] = np.maximum(probability, 1 - probability)
    predictions.to_csv("results/test_predictions.csv", index=False)

    errors = predictions[predictions["true_label"] != predictions["predicted_label"]]
    errors = errors.sort_values("confidence", ascending=False)
    errors.to_csv("results/misclassified.csv", index=False)
    print("Misclassified books:", len(errors))

    # Find the 10 words that contribute most to each class
    words = vectorizer.get_feature_names_out()
    weights = model.coef_[0]
    order = np.argsort(weights)
    rows = []
    for index in order[-10:][::-1]:
        rows.append({"towards": "Nonfiction", "word": words[index], "weight": weights[index]})
    for index in order[:10]:
        rows.append({"towards": "Fiction", "word": words[index], "weight": weights[index]})
    top_words = pd.DataFrame(rows)
    top_words.to_csv("results/top_words.csv", index=False)

    # Save the confusion matrix picture.
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Fiction", "Nonfiction"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Held-out test: title + description, min_df=3")
    fig.tight_layout()
    fig.savefig("results/figure2.png", dpi=150)
    plt.close(fig)

    pd.DataFrame([metrics]).to_csv("results/test_metrics.csv", index=False)
    print("Results saved to", "results")
