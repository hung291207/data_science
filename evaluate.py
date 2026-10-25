#evaluate on the test books
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, ConfusionMatrixDisplay


if __name__ == "__main__":
    df = pd.read_csv("data_sheet/features.csv")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    train["text"] = train["Title"].fillna("") + " " + train["Description"].fillna("")
    test["text"] = test["Title"].fillna("") + " " + test["Description"].fillna("")
    y_train = train["label"]
    y_test = test["label"]

    print("Training / test books:", len(train), "/", len(test))
    print("Test labels:")
    print(y_test.value_counts())

    #use the words from training for the test too
    vectorizer = CountVectorizer(stop_words="english", min_df=3)
    x_train = vectorizer.fit_transform(train["text"])
    x_test = vectorizer.transform(test["text"])
    print("Vocabulary size:", len(vectorizer.vocabulary_))

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)[:, 1]

    baseline = np.ones(len(test), dtype=int)
    balanced_accuracy = balanced_accuracy_score(y_test, prediction)

    #some scores for the report
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
        "baseline_accuracy": accuracy_score(y_test, baseline)
    }

    print("\nMetrics:")
    for name in metrics:
        print(name, ":", metrics[name])

    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    labelled_matrix = pd.DataFrame(matrix,
                                   index=["True Fiction", "True Nonfiction"],
                                   columns=["Predicted Fiction", "Predicted Nonfiction"])
    print("\nConfusion matrix:")
    print(labelled_matrix)

    report = classification_report(y_test, prediction, labels=[0, 1],
                                   target_names=["Fiction", "Nonfiction"], zero_division=0)
    print("\n" + report)

    os.makedirs("results", exist_ok=True)
    labelled_matrix.to_csv("results/confusion_matrix.csv")
    with open("results/classification_report.txt", "w", encoding="utf-8") as file:
        file.write(report)

    #keep predictions so i can look at the wrong ones later
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

    #words with biggest coefficients
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

    #confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(matrix, display_labels=["Fiction", "Nonfiction"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Held-out test: title + description, min_df=3")
    fig.tight_layout()
    fig.savefig("results/figure2.png", dpi=150)
    plt.close(fig)

    pd.DataFrame([metrics]).to_csv("results/test_metrics.csv", index=False)
    print("results saved")
