# Lab notes

- Setup: `code/`, `.venv`, local `data/`. Tools: requests, BeautifulSoup, pandas, Pillow, scikit-learn. Matplotlib + seaborn for plots.
- Code simplified after first pass. Removed retries, hashed page caching, temporary-file writes, transformer saving and sentinel unit tests. Kept HTTP error checks, image checks and saved-split protection.

## Step 1 - scraping

- Source: https://books.toscrape.com/. Exact Fiction/Nonfiction shelves, all pages. Product pages: title, category, description, price, rating, cover. UPC pairs row/image. Other shelves excluded.
- Static HTML -> requests + BeautifulSoup. UTF-8, resolved relative URLs, 0.5 s pause after requests. Current scraper requests HTML directly, reuses valid images.
- Found 65 Fiction + 110 Nonfiction = 175. Matches site totals. No duplicate IDs/URLs/titles or missing descriptions. All 175 images valid. Site assigns price/rating randomly -> control only.

## Step 2 - cleaning and split

- Normalised whitespace. GBP prices -> floats. Rating words -> integers 1–5. Removed 0 rows. No duplicate descriptions. Missing-description policy: empty text, retain title/cover.
- Stratified 80/20, seed 42. Train 52 Fiction + 88 Nonfiction = 140. Test 13 + 22 = 35. IDs saved before plots, preserved on reruns.
- Feature decisions use training only. Category = target. UPC, path, URL = identifiers. Excluded from predictors.

## Step 3 - training-only EDA

- `training_distributions.png`: imbalance 52/88. Fiction/Nonfiction medians: price £36.65/£32.15, rating 3/3, description words 245.5/268.5. Overlap -> no clear numeric/length split.
- `training_covers_pca.png`: similar mean RGB, overlapping classes. PC1 + PC2 = 32.8% variance. No clear split here, remaining components may contain signal.
- Word token shares normalised by class totals. Fiction: `novel`, `family`, `love`. Nonfiction: `book`, `world`, `people`. Text worth testing.
- Short-description check: *Kindle Paperwhite User's Guide* repeats title. Matches source -> kept.

## Step 4 - feature experiments

- Initial text: 6,340 terms / 140 training books -> overfitting risk. Compared `min_df=1/2/3`: 6,340/2,162/1,193 full-training terms. Chose 3 for combinations -> 81.2% fewer terms. All classifiers use L2-regularised logistic regression, C=1.
- Text balanced accuracy: 78.5% / 75.3% / 78.8% for thresholds 1/2/3. Thresholds 1 and 3 perform similarly, fold variation exceeds difference. Prefer smaller vocabulary. Threshold 3 accuracy 80.7%, macro F1 0.792.
- Colour: 8 bins/RGB channel -> 24 normalised values, then scaling. Greyscale: 32×32 -> 1024 pixels, then scaling + PCA to 20 components. Colour loses layout. Greyscale loses colour and detail. Fixed 20 components keeps model small, not claimed optimal.
- Raw image extraction treats each cover separately. Vocabulary/scalers/PCA fit inside Pipeline on each CV training fold. No CV on old fitted matrices. Full-training vocabulary counts are descriptive only.

## Step 5 - training-only CV

- Same five stratified folds, seed 42. Logistic regression uses balanced class weights. Metrics: accuracy, balanced accuracy, macro F1. Scores + fold IDs in `data/results/`.
- Majority baseline always Nonfiction: accuracy 62.9%, balanced accuracy 50.0%, macro F1 0.386. Accuracy alone hides imbalance.
- Balanced accuracy: colour 56.1%, greyscale 56.9%, numeric 49.5%. Greyscale gives no clear gain over colour. Both image-only accuracies fall below majority baseline.
- With `min_df=3`: text + colour 70.9%, text + greyscale 73.2%. Adding numeric features gives 71.5% / 70.7%. All below text alone at 78.8% -> prefer text.
- Text threshold 3 balanced-accuracy fold SD = 7.1 percentage points. Small dataset and repeated CV comparisons -> selection uncertainty. These scores guide choices, not independent final evidence.
- Checked actual fold: 112 fit / 28 validation books. Scaler/PCA sample counts all 112. All 55 fold results present, no held-out IDs in CV. Simplified scraper rerun matched feature rows and preserved split.

## Remaining

- Test set still unscored. Evaluate chosen model once on saved 35 books. One error changes accuracy by 2.9 percentage points.
- Handcrafted images miss cover text/semantics. Deep model from scratch poorly supported by 175 books. More data or pretrained embeddings possible improvements, not tested.
- Final report: 4–6 pages, max 2 figures, methods/results, limitations/fixes, effect of earlier choices, references. Submission ZIP: PDF + code, no data/cache/environment.
