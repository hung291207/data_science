# Fiction / Nonfiction project

Run from `D:\b_b_p\DS\project` in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe code\scrape.py
.\.venv\Scripts\python.exe code\clean_split.py
.\.venv\Scripts\python.exe code\eda.py
.\.venv\Scripts\python.exe code\features.py
.\.venv\Scripts\python.exe code\train.py
```

## Files

- `code/scrape.py`: follows Fiction and Nonfiction pagination, reads product pages, saves `data/books.csv` and covers. UPC pairs each row with its image. Existing images are checked and reused. HTML is requested directly, without page caching or automatic retries. Network errors stop execution.
- `code/clean_split.py`: cleans text, price and rating, checks images, saves `data/books_clean.csv` and stratified 80/20 `data/split_ids.csv` (seed 42). Refuses to silently replace a changed split.
- `code/eda.py`: reads training rows only. Saves two figures and word statistics under `data/eda/`.
- `code/features.py`: creates `data/book_features.csv` with raw text, 24 colour-histogram values, 1024 greyscale pixels, price and rating. Cover processing is independent per book. No vocabulary, scaler or PCA is fitted here.
- `code/train.py`: runs training-only cross-validation and saves scores under `data/results/`.
- `notes.md`: short decisions and observed results.

## Experiments

All models use identical five stratified CV folds, shuffled with seed 42. Every validation fold has 28 books. Learned preprocessing lives inside a scikit-learn Pipeline, so each fold gets its own vocabulary, scaler and PCA. ColumnTransformer selects predictors explicitly and excludes category, UPC, paths and URLs.

Text uses title + description with lowercase unigram counts and English stop words removed. Compared `min_df=1/2/3`. Chose 3 for combinations: fewer rare words and similar CV performance to 1. Full-training vocabulary sizes are reported separately, not reused for CV.

Colour uses 8 bins per RGB channel, normalised by image size, then StandardScaler. Greyscale resizes covers to 32×32, divides pixel values by 255, then uses StandardScaler and 20-component PCA inside each fold. Twenty components is a fixed small starting choice, not a tuned optimum.

Numeric features are standardised price and rating. The site says these values were randomly assigned, so this is a control. Combinations join text with either image representation, with and without numeric features.

Classifier: L2-regularised logistic regression, C=1, balanced class weights. This handles sparse text and reduces overfitting risk without a deep model. DummyClassifier always predicts Nonfiction as a majority baseline.

Results include mean accuracy, balanced accuracy, macro F1 and sample standard deviation across folds. Fold scores and validation IDs are saved too. The 35-book held-out test set has not been scored. CV results guide feature/model choices and are not final independent performance estimates.

## References

- Books to Scrape: https://books.toscrape.com/
- Scikit-learn citation: https://scikit-learn.org/stable/about.html#citing-scikit-learn
- CountVectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
- Pipeline: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- PCA: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
- LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- DummyClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
- StratifiedKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
