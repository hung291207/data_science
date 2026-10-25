# Lab notes

train.py: decisions

- Only the 140 training rows are used
- 5-fold StratifiedKFold, shuffle, random_state=42, created once and reused for every feature set (fair comparison).
- Each fold: 112 train / 28 validation.
- Fit-inside-folds: CountVectorizer and StandardScaler are fitted on the fold's training rows only, then applied to its validation rows (no leakage).
- Model: LogisticRegression(class_weight=balanced, max_iter=1000).
  - Why: interpretable (coefficients), gives probabilities (needed for roc-auc), standard baseline.
  - Why balanced class weights: the 52/88 imbalance.
- Baseline: DummyClassifier(most_frequent) = always nonfic.
- Metrics: balanced accuracy (main), macro-F1, accuracy. Report mean +- fold sd (ddof=1).
  - Why not accuracy alone: the baseline gets 0.629 accuracy while finding 0 fic books.
- Selection rule (fixed before looking at results): highest mean balanced accuracy. If other sets are within 1 sd of the best, pick the simplest.
  - Simplicity order: baseline < numeric < image < text_df3 < text_df2 < text_df1 < text+image < all.train.py: bugs found and fixed

1. Text column used Title twice, so the description was ignored (text jumped 0.65 to 0.77 after the fix)
2. Vectorizer/scalers were fitted on all 140 rows before cv (leakage). Moved inside the folds
3. Each feature set got different folds (random_state=42+i). Now one shared fold list
4. f1_score without average=macro (it was binary F1 for nonfic, so the baseline showed a fake 0.772)
5. Std was divided by sqrt(n) (standard error, not sd). Removed
6. The winner was chosen by plain accuracy, so the baseline won. Now chosen by balanced accuracy + the 1-sd rule
7. min_df experiment: min_df was hard-coded to 3, vocab measured rows instead of columns, simplicity order reversed

   train.py: results

| Feature set      | Bal. acc.      | Macro-F1       | Accuracy       | Text vocab |
| ---------------- | -------------- | -------------- | -------------- | ---------- |
| text+image (df3) | 0.771 +- 0.107 | 0.770 +- 0.107 | 0.800 +- 0.074 | 879        |
| text_df3         | 0.769 +- 0.094 | 0.769 +- 0.090 | 0.793 +- 0.069 | 879        |
| all (df3)        | 0.761 +- 0.095 | 0.762 +- 0.096 | 0.793 +- 0.064 | 879        |
| text_df2         | 0.753 +- 0.064 | 0.761 +- 0.065 | 0.793 +- 0.047 | 1,714      |
| text_df1         | 0.732 +- 0.073 | 0.739 +- 0.077 | 0.786 +- 0.051 | 5,326      |
| image            | 0.626 +- 0.113 | 0.616 +- 0.107 | 0.636 +- 0.102 | 0          |
| numeric          | 0.527 +- 0.047 | 0.517 +- 0.041 | 0.536 +- 0.044 | 0          |
| baseline         | 0.500 +- 0.000 | 0.386 +- 0.007 | 0.629 +- 0.020 | 0          |

- Best mean: text+image 0.771. 1-sd threshold: 0.664. Eligible: text+image, text_df3, all, text_df2, text_df1. Rule picks: text_df3

  Interpretation (raw points to turn into report prose)
- Text carries the signal (about 0.77 vs 0.50 baseline)
- Images add nothing measurable on top of text (+0.002, sd about 0.1). Image alone is only weakly above chance. Posible reason: thumbnails (small, low detail), and colour histograms ignore layout and cover text
- Price/rating are at chance (0.53). This is consistent with the site assigning them randomly, so they work as a sanity control.
- min_df: 5,326 to 879 terms (-83.5%) with no loss (0.732 to 0.769, within 1 sd). Many more features than the 112 books per fold = curse of dimensionality (, . Don't claim df3 is better, only at least as good with 6x fewer features
- Fold sd is large (about 0.1) because each validation fold has 28 books (1 book = 3.6 pp). The rankings between close sets are uncertain
- df1/df2 have a smaller sd than df3. The sd itself is noisy with 5 folds

  evaluate.py: decisions

  * Final model = the cv winner unchanged: CountVectorizer(stop_words=english, min_df=3) + LogisticRegression(class_weight=balaced, max_iter=1000), fitted on all 140 training books.
  * Vocabulary fitted on the 140 training books only. 1,112 terms (more than the cv mean of 879 because each cv fold had only 112 books, so fewer words passed min_df=3).
  * Positive class = nonfic (label 1). Probability column = predict_proba[:, 1] = P(nonfic).
  * Metrics: balanced accuracy (main), macro-F1, accuracy, roc-auc (from probabilities), per-class precision/recall, confusion matrix. Baseline = always nonfic, same as in cv.
  * FPR = FP / (FP + TN)  With nonfic as positive: FP = fic books called nonfic.
  * Saved: test_metrics.csv, test_predictions.csv, misclassified.csv (most confident errors first), confusion_matrix.csv, classification_report.txt, top_words.csv, figure2.png (= report Figure 2).

    evaluate.py: bugs found and fixed

  1. Vectorizer was fitted on train + test text together (leakage: test words shaped the vocabulary, and min_df counted test books). Now fitted on train only.
  2. probability = predict_proba[:, 0] = P(fic), while every label/metric treats nonfic (1) as positive. Now [:, 1].
  3. roc_auc_score was given the hard 0/1 predictions, not probabilities. That gives a single-threshold auc (same as balanced accuracy), not the area under the roc curve. Now uses probabilities.
  4. Baseline was np.zeros = always fic, the minority class. Inconsistent with the cv baseline (always nonfic). Now np.ones.
  5. misclassified.csv was sorted by confidence ascending, so the most confident (most interesting) errors came last
  6. Top-words labels were swapped. Positive coef pushes towards label 1 = nonfic, so the largest coefs are nonfic words, not fic. Fixed

     evaluate.py: results (35 held-out test books, run once)

  | Metric            | Test  | cv (train, mean +- sd) | Majority baseline (test) |
  | ----------------- | ----- | ---------------------- | ------------------------ |
  | Balanced accuracy | 0.663 | 0.769 +- 0.094         | 0.500                    |
  | Macro-F1          | 0.669 | 0.769 +- 0.090         | not saved                |
  | Accuracy          | 0.714 | 0.793 +- 0.069         | 0.629                    |
  | roc-auc           | 0.857 | not computed           | 0.5                      |

  Confusion matrix (rows = true, columns = predicted):

  |             | Pre fic | Pre nonfic |
  | ----------- | -------- | ----------- |
  | True fic    | 6        | 7           |
  | True nonfic | 3        | 19          |

  * Per class: fic precision 0.67, recall 0.46 (6/13). nonfic precision 0.73, recall 0.86 (19/22)
  * With nonfic positive: TPR = 19/22 = 0.86, FPR = 7/(7+6) = 0.54.
  * Test balanced accuracy is outside cv mean +- 1 sd (0.675 to 0.863), just below
  * 10 misclassified books (7 fic, 3 nonfic)
  * Top nonfic words: america, like, work, memoir, day, true, stories, guide, don, tells.
  * Top fic words: novel, son, family, finds, marriage, france, secrets, man, la, begins
  * don (from don't) and la (french) are tokenisation artifacts, not real signal.=

  Computed afterwards from the saved test_predictions.csv (no refit, test not re-scored). not produced by any repo script yet, so write a small script before these numbers go in the report

  * Bootstrap 95% CI (5,000 resamples, seed 42): balanced accuracy [0.51, 0.82], roc-auc [0.72, 0.96]
  * Average precision (PR curve): fic as positive 0.76 vs 0.37 chance level. nonfic as positive 0.93 vs 0.63

    evaluate.py: interpretation
  * Model beats the baseline on every metric, but the test score is lower than cv suggested. The bootstrap CI (0.51 to 0.82) is wide and includes the cv mean, so 35 books cannot tell worse than cv apart from unlucky split. 1 test error = 2.9 pp accuracy, 1 fic error = 7.7 pp fic recall
  * The errors are lopsided: fic recall 0.46, nonfic recall 0.86. Balanced class weights did not stop the model leaning towards the majority class on the test set
  * auc 0.86 vs balanced accuracy 0.66: the ranking is decent, the hard predictions at 0.5 are worse. 4 of the 7 missed fic books sit between P(NF) 0.51 and 0.66, so they only just fell on the wrong side

    * But don't overclaim the threshold: choosing the cutoff on out-of-fold training probabilities only moved cv balanced accuracy from 0.768 to 0.781. Never pick a threshold on the test set
  * Error analysis:

    * fic called nonfic: realistic or historical fic with real-world settings (I Am Pilgrim: Saudi Arabia, Syria, Cometh the Hour: politics, Iron Curtain, Mothering Sunday: 1924, Memoirs of a Geisha: true confessions, Japan, The Firm: law firm, mortgage, loans). Real places, history and money words push towards nonfic.
    * nonfic called fic: narrative nonfic that reads like a story (Into the Wild: a young man, a journey, A Murder Over a Girl: a crime story) and The Five Love Languages (love, spouse, relationship words, love and marriage lean fic)
    * Memoirs of a Geisha has novel in its description (the strongest fic word) and was still called nonfic (0.65). So the model adds up many topic words and no single cue decides
  * Conclusion for the report: bag of words learns topic vocabulary (history, places, self-help vs. family, relationships, novel), not narrative form. fic and nonfic overlap where fic is set in the real world and nonfic is told as a story. This is exactly where word order and context would hel
