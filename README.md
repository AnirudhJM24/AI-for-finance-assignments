# AI for Finance Assignments

Coursework applying machine learning to equity returns, credit risk, factor
estimation, loan pricing, and prediction markets.

Each assignment lives in its own folder with three things: a notebook holding the
full analysis, a written report as PDF, and a README summarising the method and the
results.

## Contents

| Assignment | Topic | Question |
| --- | --- | --- |
| [05](05-return-jumps/) | Return jumps | Will a stock move more than 10% next month? |
| [06](06-bankruptcy-prediction/) | Bankruptcy prediction | Will a firm file for bankruptcy next year? |
| [07](07-neural-beta/) | Neural beta | Can a neural network estimate market beta better than a rolling regression? |
| [08](08-crypto-beta/) | Crypto beta | What are cryptocurrencies actually exposed to? |
| [09](09-loan-pricing/) | Loan pricing | What determines the spread on a syndicated loan? |
| [10](10-prediction-markets/) | Prediction markets | Do Kalshi and Polymarket prices lead the underlying assets? |

## Headline results

| Assignment | Best model | Metric | Score |
| --- | --- | --- | --- |
| 05 | Post-LASSO logistic | Test AUC | 0.684 |
| 06 | Random forest | Test AUC | 0.875 |
| 07 | MLP, returns and industry | Test RMSE | 0.128 |
| 08 | MLP, window 3, sigmoid | Val RMSE | 0.227 |
| 09 | LightGBM | Test R2 | 0.591 |
| 10 | Kalshi against CME FedWatch | Correlation | 0.79 |

## Recurring themes

**Time-ordered splits everywhere.** Every assignment that predicts something splits
train and test by date rather than at random. Where both are available, the random
split flatters the model.

**Accuracy is the wrong metric for rare events.** Assignment 6 has a 0.47%
bankruptcy rate, and XGBoost hits 99.67% accuracy while catching almost nothing.
AUC, KS, and decile lift are what actually separate the models.

**More features are not better.** LASSO keeps 10 of 27 features in assignment 5 and
5 of 11 in assignment 6, matching the full models both times. Adding Fama-French
factors to the neural beta in assignment 7 made test RMSE worse.

**Trees beat neural networks on tabular data at these sample sizes.** In assignment 9,
LightGBM reaches test R2 0.591 against 0.263 for an MLP on 3,590 training rows.
Neural networks are the right tool in assignments 7 and 8 because the loss there
is structural, not because they generalise better.

## Data

None of the source data is committed. The notebooks read from local paths that need
to be pointed at your own copies. Each assignment README lists what it needs.

| Source | Used by |
| --- | --- |
| CRSP monthly stock file | 05, 06, 07, 08 |
| Compustat annual fundamentals | 06, 09 |
| DealScan syndicated loans | 09 |
| Fama-French factors | 07, 08 |
| FRED macro series | 05, 09 |
| Coinbase, Yahoo Finance | 08, 10 |
| Kalshi and Polymarket APIs | 10 |

Assignment 10 pulls from live APIs, so re-running it will produce different numbers
than the ones recorded in its README.

## Setup

```bash
pip install polars pandas numpy scikit-learn xgboost lightgbm torch \
            matplotlib seaborn scipy yfinance requests scikit-survival
jupyter lab
```
