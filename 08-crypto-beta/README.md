# Assignment 8: Crypto Neural Beta Across Three Risk Factors

Apply the neural beta method from assignment 7 to cryptocurrencies, and measure
their exposure to three different things at once: energy equities, the dollar, and
the broad equity market.

## Data

Monthly panel from 2015 to 2023, built from daily series resampled to month end.

| Symbol | Series | Role |
| --- | --- | --- |
| BTC, ETH, LC, BPC | Coinbase USD prices | Assets |
| VDE | Vanguard Energy ETF | Energy factor |
| FIAT | Trade-weighted dollar index (DTWEXBGS) | Currency factor |
| vwretd | CRSP value-weighted market return | Equity factor |
| GOLD | NASDAQ gold index | Control |
| Mkt-RF, SMB, HML, RF | Fama-French factors | Optional inputs |

All price series are converted to monthly log returns. The four cryptos are melted
into long format, so each row is one coin in one month with the shared factor
returns attached. Files live under `data/` and are not committed.

## Features

Per coin, over a rolling window: mean return, return standard deviation, and return
z-score. The same three statistics for each factor, plus a rolling correlation
between the coin and the market column being modelled. The Fama-French variant adds
rolling statistics for Mkt-RF, SMB, HML, and RF.

Validation and test periods are extended backwards by the window length so the first
rows of each period have a complete lookback instead of a truncated one.

Splits: train 2015 to 2019, validate 2020 to 2021, test 2021 to 2023.

## Tuning

Grid over window [3, 6, 12], hidden units [4, 8, 16], learning rate [0.001, 0.01, 0.1],
and activation [linear, sigmoid, tanh, relu], 50 epochs each. Best configuration was
window 3, 8 hidden units, learning rate 0.001, sigmoid, at validation RMSE 0.2271.

The hidden layers here are far smaller than in assignment 7 because the panel is far
smaller: four assets over about a hundred months.

## Results

Predicted betas on the test set, pooled across coins:

| Beta | Mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| Energy | -0.158 | 0.012 | -0.202 | -0.129 |
| Fiat | -0.814 | 0.011 | -0.844 | -0.784 |
| Equity | 0.053 | 0.010 | 0.028 | 0.090 |
| Energy with Fama-French | -0.149 | 0.075 | -0.281 | 0.005 |
| Fiat with Fama-French | 0.094 | 0.332 | -0.442 | 0.694 |
| Equity with Fama-French | 0.236 | 0.091 | 0.044 | 0.401 |

Quartile sort by beta with mean realised excess return, test period:

| Factor | Lowest beta coin | Highest beta coin | Q4 minus Q1 return |
| --- | --- | --- | --- |
| Energy | LC | BTC | +0.0254 |
| Fiat | ETH | LC | -0.0462 |
| Equity | LC | ETH | +0.0462 |

## What the numbers say

The strongest and most stable result is the fiat beta at -0.81 with a standard
deviation of 0.011. When the dollar strengthens, crypto falls, and the relationship
holds tightly across all four coins and the whole test window. Energy beta is
negative but small at -0.16. Equity beta is positive but nearly zero at 0.05.

Adding Fama-French factors changes the picture in ways worth reading carefully:

- Fiat beta flips from -0.81 to +0.09 and its standard deviation grows from 0.011 to
  0.33. Most of what the fiat beta was capturing was overlapping with the standard
  factors rather than being a separate dollar exposure.
- Equity beta rises from 0.05 to 0.24. Once the size and value factors absorb their
  share, the residual link to the equity market is stronger than the raw estimate
  suggested.
- Energy beta barely moves in level (-0.158 to -0.149) but its dispersion grows six
  times over.

Across the board the Fama-French version produces wider dispersion. That is expected:
controlling for common factors leaves the beta free to move with what is actually
specific to each asset, rather than tracking a single shared market signal.

The quartile spreads are unreliable and should be read as illustrative only. With
four coins there is one asset per quartile, so a spread is the difference between two
individual coins, not a portfolio result. The beta differences within each factor are
also tiny (0.053 versus 0.054 for equity), which means the sort is close to arbitrary.
