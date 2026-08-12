# Credit Risk Analysis — UCI German Credit & Taiwan Default

RMIT Case Studies in Data Science, Individual Task 1.
Comparing Random Forest and LightGBM on two consumer credit datasets.

## Datasets

- **Statlog (German Credit Data)** — 1,000 applicants, 20 attributes, 30% bad.
  Application-time data only. Ships a documented 5:1 cost matrix.
  https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

- **Default of Credit Card Clients** — 30,000 Taiwanese cardholders, 23 attributes,
  22.1% default. Six months of repayment behaviour.
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Both from the UCI Machine Learning Repository, CC BY 4.0.

## Running

Install dependencies:

    pip install pandas scikit-learn lightgbm xlrd matplotlib

Open `analysis.ipynb` and run all cells. Data files are in `data/`.

## Key findings

- Accuracy misleads under class imbalance: 0.812 on Taiwan against a
  do-nothing baseline of 0.779, while missing two-thirds of actual defaults.
- Tuning the decision threshold from 0.50 to 0.25 cut expected cost under the
  5:1 matrix from 201 to 97 — a larger effect than the choice of algorithm.
- Cross-validation shows the two models are indistinguishable on German
  (overlapping intervals) and equivalent on Taiwan.
- Removing age, sex and foreign-worker status cost 0.007–0.009 AUC, well
  within fold-to-fold variation.