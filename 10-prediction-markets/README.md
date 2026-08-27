# Assignment 10: Prediction Markets

Pull live prices from Kalshi and Polymarket, line them up against the underlying
assets, and check whether the markets lead the assets or follow them.

Three markets are studied, one per section of the notebook.

## Market 1: Will the S&P 500 finish 2025 positive

Kalshi series `KXINXPOS`, contract `KXINXPOS-25DEC31H1600-T5881.63`. Pays out if the
S&P 500 closes above 5881.63 on 31 December 2025.

Daily candlesticks are pulled from the Kalshi API from 8 May 2025 onward, then merged
with S&P 500 closes from Yahoo Finance. That leaves 118 trading days.

Contract price statistics over the period: mean 0.81, median 0.87, min 0.30, max 0.98.
The contract opened at 0.40 in May and reached 0.97 by late November as the index rose
from 5844 to 6849.

Volume is heavily skewed: median 100 contracts, mean 1,321, max 42,618 on 27 November.
Correlation between contract price and volume is 0.297, so activity picks up when the
price moves but the relationship is loose.

A linear regression of the contract price on lagged features (previous close, index
level, index returns at 1 and 7 days, volume, moving averages, and a price-volume
interaction) gives in-sample R2 of 0.514 and RMSE of 0.066. This is a fit on the same
rows it was trained on, not a forecast, and should be read as a description of the
comovement rather than as predictive performance.

## Market 2: What price will Bitcoin hit, 24 to 30 November

Polymarket event `what-price-will-bitcoin-hit-november-24-30`, covering 11 strike
levels from "dip to $72,000" up to "reach $100,000".

Volume by market ranges from $424,436 on the $94,000 strike down to $38,347 on the
$76,000 dip. Hourly median prices are pulled per strike from the CLOB API and merged
into one frame, then joined against hourly BTC prices from Yahoo Finance.

The strike curve behaves the way an options chain does. Over the six days, the
$94,000 strike falls from 0.255 to 0.07 and the $100,000 strike falls from 0.10 to
0.0065 as BTC drops from about $87,400 to the mid $80,000s. The dip strikes move the
other way early on, then also collapse as the window closes and the remaining time
runs out.

A lead-lag correlation between strike prices and BTC over a window of plus or minus
8 hours is computed for the $94,000 strike, the $84,000 dip, and BTC itself.

Two classifiers predict whether BTC ticks up in the next hour, trained on hours before
28 November and tested on the 9 trading hours of 28 November:

| Model | Accuracy | Notes |
| --- | --- | --- |
| Random forest | 0.667 | Predicts down for 8 of 9 hours |
| Gradient boosting | 0.222 | Worse than always guessing down |

Nine test observations is far too few to conclude anything. The random forest achieves
its 0.667 by predicting "down" almost always, and the actual split was 5 down and 4 up.
Neither result is evidence of a usable signal.

## Market 3: December 2025 Fed decision

Kalshi event `KXFEDDECISION-25DEC`, five contracts covering cut by more than 25bps,
cut by 25bps, no change, hike by 25bps, and hike by more than 25bps.

| Contract | Volume | Price on 29 Nov |
| --- | --- | --- |
| Cut 25bps | 5,689,991 | 0.86 |
| No change | 6,548,879 | 0.13 |
| Cut more than 25bps | 4,447,259 | 0.01 |
| Hike 25bps | 183,867 | near 0 |
| Hike more than 25bps | 44,374 | near 0 |

These are compared against CME FedWatch implied probabilities for the same meeting,
read from `FedMeeting_20251210.csv` and restricted to the three relevant target bands
given the current 4.00 to 4.25% band.

Correlations from October onward, Kalshi contract price against CME band probability:

| CME band | Kalshi contract | Correlation |
| --- | --- | --- |
| 350-375 | cut more than 25 | 0.62 |
| 350-375 | cut 25 | 0.77 |
| 350-375 | no change | -0.79 |
| 375-400 | cut more than 25 | -0.64 |
| 375-400 | cut 25 | -0.78 |
| 375-400 | no change | 0.79 |

A lead-lag heatmap over plus or minus 5 days is produced for each contract against
each band.

## What the numbers say

The Kalshi Fed contracts and CME FedWatch line up closely and with the right signs.
The 375-400 band corresponds to a 25bps cut from the current level, and it correlates
at 0.79 with the no-change contract and -0.78 with the 25bps cut contract. Two venues
with different participants and different mechanics are pricing the same event
consistently, which is a reasonable check that neither is badly mispriced.

The lead-lag results do not show a clean winner. Correlations peak at or very near
zero lag in most pairs, meaning the prediction markets and the underlying assets are
absorbing the same information at roughly the same time rather than one leading the
other. At daily and hourly resolution, that is what you would expect from two liquid
venues watching the same news.

The Bitcoin classifiers are the weakest part of the study. Nine test hours cannot
distinguish a model from a coin flip, and the gap between the random forest at 0.667
and gradient boosting at 0.222 is well inside what noise alone would produce at that
sample size.
