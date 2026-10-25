import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df = pd.read_csv("data_sheet/books_data.csv")
    df = df.drop_duplicates(subset="Link").reset_index(drop=True)

    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Category"])

    train.to_csv("data_sheet/train.csv", index=False)
    test.to_csv("data_sheet/test.csv", index=False)
    print("Training books:", len(train))
    print("Test books:", len(test))
