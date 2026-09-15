"""Ask what the learned beta is actually tracking.

A beta estimator can be judged without any portfolio at all, by checking whether
it behaves like a beta. This script compares the neural beta against a rolling
regression beta on four things:

  - what it correlates with in the cross-section, month by month;
  - how much cross-sectional dispersion it carries;
  - whether it is ever negative, which a beta can be and a volatility cannot;
  - how much it moves month to month, which sets the turnover it will cost.

    python diagnose_beta.py --msf MSF_dl.csv --betas neural_betas.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import bab
import run_bab

MIN_NAMES = 30   # months thinner than this give unstable correlations


def cross_sectional_corr(panel: pd.DataFrame, a: str, b: str) -> float:
    """Average of the within-month correlations between two columns.

    Pooling instead would mix cross-sectional and time-series variation, and it
    is the cross-section that a portfolio sort acts on.
    """
    per_month = panel.groupby("t").apply(
        lambda g: g[a].corr(g[b]) if len(g) > MIN_NAMES else np.nan,
        include_groups=False,
    )
    return float(per_month.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msf", default="MSF_dl.csv")
    ap.add_argument("--betas", default="neural_betas.csv")
    ap.add_argument("--start", type=int, default=2018)
    ap.add_argument("--end", type=int, default=2023)
    ap.add_argument("--ols-window", type=int, default=60)
    ap.add_argument("--ols-min-periods", type=int, default=36)
    args = ap.parse_args()

    raw = run_bab.load_crsp(args.msf, args.start, args.end,
                            lookback_years=args.ols_window // 12 + 1)
    panel = run_bab.to_panel(raw, "RET")
    panel["beta_ols"] = bab.rolling_ols_beta(
        panel, window=args.ols_window, min_periods=args.ols_min_periods)

    # Characteristics as of formation: every window ends at t-1, so none of
    # these can see the month whose return the beta is used to trade.
    g = panel.sort_values(["PERMNO", "t"]).groupby("PERMNO")["ret"]
    panel["mom12"] = g.transform(lambda s: s.shift(1).rolling(12).sum())
    panel["vol12"] = g.transform(lambda s: s.shift(1).rolling(12).std())
    panel["rev1"] = g.shift(1)

    neural = pd.read_csv(args.betas).rename(columns={"beta": "beta_nn"})
    panel = panel.merge(neural[["PERMNO", "year", "month", "beta_nn"]],
                        on=["PERMNO", "year", "month"], how="left")

    traded = panel[(panel["year"] >= args.start) & (panel["year"] <= args.end)]
    sample = traded.dropna(subset=["beta_nn", "beta_ols", "mom12", "vol12", "rev1"])
    print(f"# What is the learned beta tracking?\n\n{len(sample):,} firm-months, "
          f"{args.start}-{args.end}\n")

    characteristics = [("vol12", "Trailing 12-month volatility"),
                       ("beta_ols", "A 60-month regression beta"),
                       ("mom12", "Trailing 12-month return"),
                       ("rev1", "Previous month's return")]
    print("| Predicted beta correlates with | Neural | Rolling OLS |")
    print("| --- | --- | --- |")
    for col, label in characteristics:
        cells = []
        for beta in ("beta_nn", "beta_ols"):
            cells.append("—" if col == beta
                         else f"{cross_sectional_corr(sample, beta, col):.3f}")
        print(f"| {label} | {cells[0]} | {cells[1]} |")

    print("\n| | Mean | Std | p1 | p99 | Share negative | Median monthly change |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for beta, label in (("beta_nn", "Neural"), ("beta_ols", "Rolling OLS")):
        s = sample[beta]
        change = (sample.sort_values(["PERMNO", "t"])
                        .groupby("PERMNO")[beta].diff().abs().median())
        print(f"| {label} | {s.mean():.3f} | {s.std():.3f} | {s.quantile(.01):.3f} "
              f"| {s.quantile(.99):.3f} | {(s < 0).mean():.2%} | {change:.3f} |")

    print("\nA beta can be negative; a volatility cannot. A learned beta that is "
          "never\nnegative, tracks trailing volatility more closely than it tracks "
          "a regression\nbeta, and moves faster than one, is a volatility estimate "
          "wearing a beta label.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
