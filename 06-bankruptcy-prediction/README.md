# Assignment 6: Corporate Bankruptcy Prediction

Predict whether a firm files for bankruptcy in the following year from accounting
ratios and market data. The target is rare, which is what makes the problem
interesting: accuracy is useless here and the whole exercise turns on ranking.

## Data

| Source | File | Coverage |
| --- | --- | --- |
| Compustat annual fundamentals | `COMPUSTAT_funda_annual.csv` | 1964 to 2020 |
| CRSP monthly stock file | `msf_raw_1964To2023.csv` | 1964 to 2023 |
| Bankruptcy filings | `bankruptcy_1964_2020.csv` | PERMNO and filing date |

None of these are committed. Point the notebook at your own copies before running.

Compustat and CRSP are joined on the first six CUSIP digits plus fiscal year.
Bankruptcy flags are joined on PERMNO and year. The merged panel holds 521,159
firm-years, of which 0.47% are bankruptcy years.

## Features

Accounting ratios, all scaled by total assets unless noted: net income, total
liabilities, net working capital (current assets minus current liabilities),
retained earnings, EBIT, and sales. Missing statement items are carried forward
within a firm before the ratios are computed, and anything still missing is filled
with the column median.

Market variables aggregated to the year: mean return, return volatility, mean
market return, summed excess return over the market, and year-end market cap.

Every feature is lagged one year within the firm, so the model predicting year `t+1`
sees only year `t-1` financials and market data. Pairs with absolute correlation
above 0.6 are pruned, one of each pair dropped, which leaves 11 usable features.

## Evaluation

The main split is by time: train on 1964 to 1990 (85,773 rows, 493 bankruptcies),
test on 1991 onward (314,824 rows, 1,013 bankruptcies). A random 75/25 split is also
run as a reference point, and a rolling expanding-window variant is compared against
the static fit.

## Results

Out-of-sample discrimination, 1964 to 1990 train, post-1990 test:

| Model | AUC | KS |
| --- | --- | --- |
| Random forest (depth 5, 100 trees) | 0.875 | 0.632 |
| Ridge logistic (alpha 10,000) | 0.868 | 0.644 |
| Post-LASSO logistic (5 features) | 0.858 | 0.595 |
| KNN (k = 5) | 0.545 | 0.090 |

Logistic regression with balanced class weights on the random split reaches AUC
0.861. Fitting the same model as a rolling expanding window rather than once
statically raises KS from 0.608 to 0.639, and a fixed rolling window reaches 0.640.

LASSO keeps 5 of the 11 features at alpha 0.0013: mean return, return volatility,
net income over assets, liabilities over assets, and net working capital over assets.
That is a compact and readable model, and it gives up only 0.01 AUC against ridge.

Ranking quality, ridge logistic, test set decile 10 is highest predicted risk:

| Decile | Observed bankruptcy rate | Count |
| --- | --- | --- |
| 1 | 0.07% | 31,483 |
| 5 | 0.03% | 31,492 |
| 8 | 0.20% | 31,482 |
| 9 | 0.62% | 31,482 |
| 10 | 2.07% | 31,483 |

The top decile carries a bankruptcy rate about 30 times the bottom decile, and
almost all of the separation happens in the last two deciles.

Gradient boosting on raw accuracy, for contrast:

| Model | Accuracy | Recall on bankruptcies |
| --- | --- | --- |
| XGBoost (25 rounds) | 0.9967 | 0.003 |
| LightGBM (100 rounds) | 0.9953 | 0.020 |

## What the numbers say

XGBoost tuned for accuracy reaches 99.67% and catches essentially no bankruptcies.
Predicting "no bankruptcy" for every row scores 99.68%. That is the whole lesson of
the assignment: with a 0.47% base rate, accuracy measures nothing, and the models
worth keeping are the ones scored on AUC, KS, and decile lift.

Class weighting is what makes logistic regression usable. Without it the model
collapses to the majority class; with it, recall on bankruptcies reaches 0.77 on the
random split, at the cost of precision falling to 0.017. That trade is acceptable
when the output is a risk ranking rather than a hard yes or no.

KNN fails outright at AUC 0.545. With 11 dimensions and a rare target, the five
nearest neighbours of a distressed firm are almost always healthy firms, so the
predicted probability barely moves.

The rolling window beats the static fit on KS, which says the relationship between
these ratios and default is not stable from 1964 to 2020 and benefits from refitting.
