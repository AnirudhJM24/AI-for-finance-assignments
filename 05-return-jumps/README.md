# Assignment 5: Predicting Monthly Return Jumps

Classify whether a US stock will move more than 10% in absolute terms next month,
using its own return history, a rolling market beta, and macro conditions.

## Data

| Source | File | Coverage |
| --- | --- | --- |
| CRSP monthly stock file | `MSF_1996_2023.csv` | 1996 to 2023, all listed firms |
| FRED macro series | `macros/*.csv` | AAA, BAA, FEDFUNDS, M2REAL, T10Y2YM, UNRATE, USREC, VIXCLS |

Neither file is committed. Point the notebook at your own copies before running.

## Target

A firm-month is labelled a jump when the *next* month's return excluding dividends
satisfies `|RETX(t+1)| > 0.10`. Across the full CRSP panel 35.7% of firm-months
qualify. After restricting to the 100 firms with the longest histories and dropping
rows without a full beta window, the rate falls to 29.7%, so the classes are close
enough to balanced that no resampling is used.

## Features

Everything is lagged by one month, so a prediction made at the end of month `t`
only sees information available at that point.

- Rolling market beta from a 12-month regression of `RETX` on `vwretd`.
- 6-month realised volatility of the stock and of the market.
- Prior price, volume, shares outstanding, and return.
- 3, 6, and 12-month rolling z-scores of prior return, price, and volume.
- Excess return over the market, dollar trading volume, and a beta times VIX term.
- Macro levels: unemployment, fed funds, real M2, term spread, recession flag,
  the BAA minus AAA credit spread, and month-end VIX.
- Industry from the SIC code, dense-rank encoded.

Raw macro levels correlate with the jump rate only weakly on their own. The rolling
z-scores and the beta-VIX interaction are what give the models something to work with.

## Evaluation

Three splits, all respecting time order:

- **Fixed**: train on 1996 to 2017, test on 2018 to 2023.
- **Expanding**: for each year from 2018, train on everything before it.
- **Moving window**: same, but the training window slides rather than grows.

## Results

Logistic regression, fixed split:

| | Train | Test |
| --- | --- | --- |
| AUC | 0.705 | 0.680 |
| KS | 0.297 | 0.282 |
| Accuracy | 0.723 | 0.696 |

Logistic regression, expanding window by test year:

| Test year | Test AUC | Test KS | Test accuracy |
| --- | --- | --- | --- |
| 2018 | 0.688 | 0.275 | 0.740 |
| 2019 | 0.670 | 0.255 | 0.753 |
| 2020 | 0.680 | 0.294 | 0.640 |
| 2021 | 0.666 | 0.282 | 0.743 |
| 2022 | 0.646 | 0.255 | 0.653 |
| 2023 | 0.669 | 0.268 | 0.657 |

The moving window results track the expanding ones within about 0.01 AUC, so the
older data is neither helping nor hurting much.

Model comparison on the fixed split:

| Model | Selected hyperparameters | Train AUC | Test AUC | Test accuracy |
| --- | --- | --- | --- | --- |
| Logistic regression | none | 0.705 | 0.680 | 0.696 |
| Post-LASSO logistic | alpha 0.0057, 10 of 27 features kept | 0.703 | 0.684 | 0.697 |
| Ridge classifier | alpha 483, all features kept | | | 0.698 |
| KNN | k = 49 | | | 0.692 |
| XGBoost | depth 3, gamma 5, subsample 0.9 | 0.764 | 0.680 | 0.696 |

## What the numbers say

Test AUC lands between 0.65 and 0.68 for every model tried. XGBoost fits the
training set noticeably better (0.764 versus 0.705) without any gain out of sample,
which is the clearest sign that the extra flexibility is fitting noise rather than
structure.

LASSO keeps 10 of 27 features and matches the full model, so most of the feature
set is redundant. Ridge picks a very large penalty (483) and shrinks every
coefficient rather than dropping any, ending up in the same place.

Accuracy moves more across test years than AUC does, and it drops in 2020 and 2022.
Those are the years when the base rate of jumps rose, so a fixed 0.5 threshold
misclassifies more. AUC, which ignores the threshold, is far more stable, and it is
the number to trust when comparing years.

The single most useful predictor is a large shift in VIX, which raises the jump rate
in the following month. Everything else adds only a little on top of that.
