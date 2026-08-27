# Assignment 9: Syndicated Loan Pricing

Predict the all-in-drawn spread on a syndicated loan, in basis points over LIBOR,
from loan terms, borrower financials, borrower risk measures, and macro conditions.

## Data

| Source | File | Contribution |
| --- | --- | --- |
| DealScan | `data/loan_pricing_dealscan.csv` | Loan terms and pricing, 1989 to 2024 |
| Compustat | `data/compustat_log.csv` | Borrower financials, log transformed |
| Precomputed betas | `beta.csv` | Beta and volatility at 12, 24, 36 months |
| FRED macro series | `macros/*.csv` | AAA, BAA, FEDFUNDS, M2REAL, T10Y2YM, UNRATE |

None of these are committed. Point the notebook at your own copies before running.

## Cleaning

DealScan arrives at 160,041 rows with heavy duplication. The same facility appears
many times because `datadate`, `fyear`, and `PERMNO` vary while the pricing does not.
Checking directly confirms that `allindrawn` is constant within `facilityid`, so
deduplicating on facility, date, year, and PERMNO is safe and loses no pricing
variation.

Columns with more than 14,000 nulls are dropped. What remains is filled with the
column median for numerics and the mode for categoricals. Only completed deals are
kept.

`minbps` and `maxbps` are dropped before modelling. They correlate with the target at
0.77 and 0.69 and are components of the same pricing grid, so keeping them would
leak the answer.

Merging Compustat on gvkey and year, then betas on PERMNO and year, then macro
averages on year, leaves 8,164 loan-years covering 820 firms. Splits are by time:
train before 2018 (3,590 rows), test 2018 to 2022.

## Features

Loan terms: facility amount, deal amount, maturity in months, and duration from
first facility start to last facility end. One-hot encodings of loan type, primary
purpose, seniority, and deal purpose.

Borrower financials: sales, operating income, interest expense, net income, SG&A,
assets, liabilities, common equity, current assets and liabilities, operating and
investing and financing cash flow, capex, cash change, dividends and investment over
net assets, and book leverage.

Risk: beta, total volatility, market volatility, systematic volatility, idiosyncratic
volatility, and returns, each at 12, 24, and 36 month horizons. Industry dummies.

Macro: annual averages of the six FRED series.

## Results

Target is `allindrawn`, in basis points.

| Model | Train RMSE | Test RMSE | Train R2 | Test R2 |
| --- | --- | --- | --- | --- |
| LightGBM | 24.4 | 38.7 | 0.902 | 0.591 |
| XGBoost | 18.7 | 40.0 | 0.942 | 0.563 |
| KNN, k = 10, unscaled | 20.6 | 41.0 | 0.930 | 0.542 |
| KNN, k = 10, scaled | 2.5 | 44.3 | 0.999 | 0.465 |
| MLP, early stopped | 36.3 | 52.0 | 0.782 | 0.263 |
| LASSO | 55.2 | 58.8 | 0.496 | 0.057 |

Multitask MLP predicting `allindrawn` and `allinundrawn` from a shared trunk, on
standardised targets: test R2 0.309 on the main task and 0.202 on the auxiliary task.
That beats the single-task MLP at 0.263.

MLP hyperparameter grid over hidden size, learning rate, and activation. Best by test
RMSE was 64 hidden units, learning rate 5e-4, ReLU, at test R2 0.297. Best by
validation loss was 256 hidden units, learning rate 1e-3, ReLU, and that model scored
test R2 -0.016.

Feature importance:

| LASSO, by coefficient | LightGBM, by split count | XGBoost, by gain |
| --- | --- | --- |
| tvol36 | facilityamt | dealpurpose_LBO |
| dealpurpose_Corp. purposes | dealamount | industry_Mining |
| xint | xint | dealpurpose_Debtor-in-poss. |
| maturity | ivncf | AAA |
| loantype_Revolver >= 1yr | duration | primarypurpose_Debtor-in-poss. |

## What the numbers say

LASSO reaches train R2 0.496 and test R2 0.057. The relationship it fits is real
in-sample and almost entirely gone out of sample, which says loan pricing is not
close to linear in these inputs.

The gradient boosting models roughly triple that, to test R2 0.56 to 0.59. LightGBM
edges out XGBoost while overfitting less (train R2 0.90 versus 0.94), so the shallower
fit is also the better one here.

Scaling makes KNN worse, from test R2 0.542 to 0.465, which is unusual. Standardising
gives every one of the 70-plus features equal weight in the distance metric, including
dozens of sparse one-hot columns. Unscaled, the large-magnitude features such as
facility amount dominate the distance, and those happen to be the ones that matter.
The scaled model reaches train R2 0.999, which is memorisation.

The neural networks come last. With 3,590 training rows and this many features they
have less to work with than the trees do, and the best model by validation loss
scored negative R2 on the test set, so validation loss is not a reliable guide at
this sample size.

The two model families disagree about what drives pricing. LASSO picks up long-horizon
volatility and macro spreads, the boosting models pick up loan size and purpose, and
XGBoost puts most of its weight on the LBO and debtor-in-possession flags. Those flags
identify the genuinely distressed and leveraged deals, which are the ones priced far
from the average, and a linear model cannot express that kind of conditional jump.
