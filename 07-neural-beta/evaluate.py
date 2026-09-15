"""Compare beta estimators statistically and as trading strategies.

Four estimators are put through the same machinery:

    MLP          the walk-forward network from walkforward.py
    Rolling OLS  a 60-month regression ending at the formation month
    Shrunk OLS   0.67 * rolling + 0.33, the Blume adjustment toward one
    Beta = 1     no estimation at all, the honest null

The statistical test is hedging: how much of next month's return variance is
removed by holding minus beta times the market. The economic test is a
monthly-rebalanced quintile sort and a beta-neutral betting-against-beta
factor, judged against the Fama-French factors with Newey-West inference.

    python evaluate.py --msf MSF_dl.csv --fama FAMA.csv --betas betas_pit.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import polars as pl

import bab
import pit

BLUME_SLOPE, BLUME_INTERCEPT = 0.67, 0.33


def md(df: pd.DataFrame, fmt="{:.4f}") -> str:
    out = df.copy()
    body = out.apply(lambda c: c.map(
        lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
        else ("" if pd.isna(v) else str(v))))
    head = "| " + " | ".join(str(c) for c in body.columns) + " |"
    rule = "| " + " | ".join("---" for _ in body.columns) + " |"
    return "\n".join([head, rule] + ["| " + " | ".join(r) + " |" for r in body.to_numpy()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msf", default="MSF_dl.csv")
    ap.add_argument("--fama", default="FAMA.csv")
    ap.add_argument("--betas", default="betas_pit.csv")
    ap.add_argument("--portfolios", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--min-price", type=float, default=0.0)
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    panel, _ = pit.build(args.msf)
    neural = pl.read_csv(args.betas)
    panel = panel.join(neural.select(["PERMNO", "t", "beta_nn"]),
                       on=["PERMNO", "t"], how="inner")
    if args.min_price > 0:
        panel = panel.filter(pl.col("PRC").abs() >= args.min_price)

    df = panel.to_pandas()
    df["beta_ols"] = df["beta_60"]
    df["beta_shrunk"] = BLUME_SLOPE * df["beta_60"] + BLUME_INTERCEPT
    df["beta_one"] = 1.0

    estimators = {"MLP": "beta_nn", "Rolling OLS": "beta_ols",
                  "Shrunk OLS": "beta_shrunk", "Beta = 1": "beta_one"}

    print(f"# Beta estimators, {df.year.min()}-{df.year.max()}\n")
    print(f"{len(df):,} firm-months, {df.PERMNO.nunique():,} firms, "
          f"{df.t.nunique()} months, median {df.groupby('t').size().median():.0f} "
          f"names per month.")
    print("\nPositions are formed at the close of month t on information through "
          "t,\nand earn month t+1. Costs are "
          f"{args.cost_bps:.0f}bps per unit of weight traded.")

    # ---- statistical: does the beta hedge? ----
    print("\n\n## Hedging\n")
    print("Variance of `r(t+1) - beta * mkt(t+1)`. Lower is better; the "
          "unhedged\nreturn is the baseline.\n")
    base = df["ret_next"].var()
    rows = []
    for label, col in estimators.items():
        resid = df["ret_next"] - df[col] * df["mkt_next"]
        rows.append({"estimator": label, "residual_variance": resid.var(),
                     "variance_removed": 1 - resid.var() / base,
                     "mean_beta": df[col].mean(), "sd_beta": df[col].std(),
                     "share_negative": (df[col] < 0).mean()})
    print(md(pd.DataFrame(rows)))

    # ---- cross-sectional agreement ----
    print("\n\n## What each estimate tracks\n")
    print("Mean within-month cross-sectional correlation.\n")
    targets = [("vol_12", "Trailing 12-month volatility"),
               ("beta_60", "60-month regression beta"),
               ("mean_ret_12", "Trailing 12-month mean return")]
    rows = []
    for label, col in estimators.items():
        if df[col].std() == 0:
            continue
        row = {"estimator": label}
        for tcol, tname in targets:
            r = df.groupby("t").apply(
                lambda g: g[col].corr(g[tcol]) if len(g) > 30 else np.nan,
                include_groups=False)
            row[tname] = r.mean()
        rows.append(row)
    print(md(pd.DataFrame(rows)))

    # ---- economic: portfolios ----
    ff = bab.load_fama_french(args.fama)
    ff["t"] = ff["year"] * 12 + ff["month"] - 1
    # Factors are dated by the month the position earns, which is t+1.
    ff["t"] = ff["t"] - 1
    factor_cols = [c for c in ("Mkt-RF", "SMB", "HML") if c in ff.columns]
    factors = ff.set_index("t")[factor_cols]
    rf = ff.set_index("t")["RF"] if "RF" in ff.columns else None

    sortable = {k: v for k, v in estimators.items() if df[v].std() > 0}
    print("\n\n## Strategy comparison\n")
    print("`ff_alpha` is the monthly alpha against Mkt-RF, SMB and HML, with "
          "Newey-West\nt-statistics throughout. Beta = 1 is constant, so it "
          "cannot be sorted, and\nBlume shrinkage is monotone, so it gives "
          "the same quintiles as the raw\nregression and differs only in the "
          "levered BAB factor.\n")
    for weight in ("ew", "vw"):
        table = bab.compare_estimators(
            df, sortable, ret_col="ret_next", rf=rf, factors=factors,
            n_portfolios=args.portfolios, weight=weight,
            cost_bps=args.cost_bps, include_bab=(weight == "ew"),
            # In the point-in-time convention a row's market cap is already the
            # value at formation, because the return it earns is next month's.
            mktcap_lag="mktcap")
        keep = [c for c in ["n_months", "mean_monthly", "t_stat", "ann_return",
                            "ann_vol", "sharpe", "max_drawdown", "ff_alpha",
                            "ff_alpha_t", "turnover_monthly", "net_mean_monthly"]
                if c in table.columns]
        print(f"\n### {'Equal' if weight == 'ew' else 'Value'}-weighted\n")
        print(md(table[keep].reset_index()))

    # ---- the Fama-French factors themselves, for scale ----
    print("\n\n## The Fama-French factors over the same months\n")
    window = factors.loc[factors.index.isin(df["t"].unique())]
    rows = [{"factor": c, **{k: v for k, v in
                             bab.newey_west_mean(window[c]).items()
                             if k in ("mean", "tstat")}}
            for c in factor_cols]
    print(md(pd.DataFrame(rows).rename(columns={"mean": "mean_monthly",
                                                "tstat": "t_stat"})))

    print("\n\nA Q5-minus-Q1 spread is long roughly one unit of market, so "
          "`ff_alpha`,\nnot `mean_monthly`, is the CAPM test on those rows. "
          "The BAB rows are levered\nbeta neutral at formation and need no "
          "such adjustment.")

    if args.plot:
        _plot(df, sortable, rf, args.plot)
        print(f"\nWrote {args.plot}")
    return 0


def _dates(index) -> pd.DatetimeIndex:
    y, m = bab.decode_month(np.asarray(index))
    return pd.to_datetime([f"{a}-{b:02d}-01" for a, b in zip(y, m)])


def _plot(df, estimators, rf, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, col in estimators.items():
        factor = bab.bab_factor(df, beta_col=col, ret_col="ret_next", rf=rf)
        if not factor.empty:
            ax.plot(_dates(factor.index), (1 + factor["bab"]).cumprod(), label=f"{label} BAB")

    mkt = df.groupby("t")["mkt_next"].first()
    ax.plot(_dates(mkt.index), (1 + mkt).cumprod(), label="Market",
            linestyle="--", color="gray")
    ax.set_title("Betting against beta, by beta estimator")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
