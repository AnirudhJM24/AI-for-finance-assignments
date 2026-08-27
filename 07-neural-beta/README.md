# Assignment 7: Neural Beta with an MLP

Estimate a time-varying market beta with a neural network instead of a rolling
regression. The network never sees a beta label. It outputs a beta, that beta is
used to predict next month's return, and the prediction error is what gets minimised.

## Idea

A small MLP maps a feature vector at time `t` to a single number, `beta_hat`.
Next month's return is then reconstructed as

```
r_hat(t+1) = alpha + beta_hat * mkt(t+1)
```

where `alpha` is one learned scalar shared across the panel. The loss is the RMSE of
`r_hat` against the realised return. Beta is identified as whatever value makes that
reconstruction work, so it comes out of the training loop rather than a regression
window.

The output layer is left linear. Betas can be negative, and any squashing activation
on the output would rule that out.

## Data

CRSP monthly returns (`MSF_1996_2023.csv`) for 2005 to 2023, restricted to 10 firms
in each of 8 industries, plus Fama-French factors (`FAMA.csv`). Neither file is
committed.

Each sample stacks a 12-month lookback of the firm's own returns and the market
return, plus 8 industry dummies, for 32 input features. The Fama-French variant adds
12 months each of Mkt-RF, SMB, HML, and RF, for 80 features.

Splits are by year: train 2005 to 2012, validate 2013 to 2017, test 2018 to 2023.
That gives 16,165 samples overall and 4,418 in the test set.

## Tuning

Grid over hidden units [32, 64, 128], learning rates [1e-2, 1e-3, 3e-4],
activations [linear, sigmoid, tanh, relu], and lookback [12, 24, 36]. The winner was
128 hidden units, learning rate 3e-4, ReLU, 12-month lookback, trained for 30 epochs
with batch size 256.

The grid is gated behind a `done = True` flag in the notebook so it does not re-run
on every execution. The chosen values are hardcoded in the cell that follows it.

## Results

| Model | Val RMSE | Test RMSE |
| --- | --- | --- |
| Returns and industry (32 features) | 0.0877 | 0.1280 |
| Plus Fama-French factors (80 features) | 0.0886 | 0.1323 |

Predicted beta by industry, test set, base model:

| Industry | N | Mean beta | Std | Median |
| --- | --- | --- | --- | --- |
| Transportation | 360 | 1.300 | 0.252 | 1.302 |
| Finance | 648 | 1.117 | 0.307 | 1.154 |
| Services | 576 | 1.110 | 0.366 | 1.185 |
| Wholesale | 539 | 1.095 | 0.421 | 1.183 |
| Construction | 576 | 1.094 | 0.342 | 1.199 |
| Retail | 470 | 0.980 | 0.303 | 1.050 |
| Manufacturing | 601 | 0.912 | 0.376 | 0.834 |
| Mining | 648 | 0.844 | 0.407 | 0.764 |

Same table with Fama-French factors added:

| Industry | Mean beta | Std |
| --- | --- | --- |
| Transportation | 1.380 | 0.601 |
| Finance | 1.281 | 0.620 |
| Construction | 1.261 | 0.639 |
| Services | 1.258 | 0.637 |
| Wholesale | 1.229 | 0.661 |
| Retail | 1.198 | 0.630 |
| Manufacturing | 1.140 | 0.654 |
| Mining | 1.095 | 0.643 |

Beta-sorted quintile portfolios over the test period, Fama-French model:

| Quintile | Mean beta | Equal-weighted return | Value-weighted return |
| --- | --- | --- | --- |
| 1 | 0.315 | 0.0138 | 0.0307 |
| 2 | 0.857 | 0.0142 | 0.0177 |
| 3 | 1.243 | 0.0205 | 0.0104 |
| 4 | 1.578 | 0.0093 | 0.0166 |
| 5 | 2.115 | 0.0063 | 0.0156 |
| Q5 minus Q1 | | -0.0074 | -0.0151 |

## What the numbers say

Adding Fama-French factors made the model worse on both validation and test RMSE,
and it flattened the cross-section. Industry means compress into a 1.10 to 1.38 band
and every standard deviation roughly doubles. The extra 48 inputs give the network
more ways to fit the training window without improving what it learns about beta.

The base model's industry ordering is the sensible one. Transportation is highest at
1.30, and the test window covers 2020, when transport names moved hardest with the
market. Mining and manufacturing sit below 1.

Both quintile spreads are negative. High-beta stocks underperformed low-beta stocks
over 2018 to 2023, by 74 basis points per month equal-weighted and 151 basis points
value-weighted. This is the betting-against-beta pattern and it runs directly against
what CAPM predicts. The value-weighted spread is the wider of the two, so it is not
being driven by small illiquid names.
