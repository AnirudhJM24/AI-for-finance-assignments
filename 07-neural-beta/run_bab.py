"""Run the betting-against-beta analysis on the neural beta estimates.

    python run_bab.py --msf MSF_1996_2023.csv --fama FAMA.csv \
                      --betas neural_betas.csv --start 2018 --end 2023

`--betas` is the CSV exported by the last section of neural-beta.ipynb, with
columns PERMNO, year, month, beta. Without it the script still runs, using only
the rolling-regression benchmark, which is a useful way to check the plumbing
before wiring the network in.

Every table is printed as markdown so it can be pasted into the README.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import polars as pl

import bab

INDUSTRY_BOUNDS = [
    (1, 999, "Agriculture"), (1000, 1499, "Mining"), (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"), (4000, 4999, "Transportation"),
    (5000, 5199, "Wholesale"), (5200, 5999, "Retail"), (6000, 6799, "Finance"),
    (7000, 8999, "Services"), (9000, 9999, "PublicAdmin"),
]


def load_crsp(path: str, start: int, end: int, lookback_years: int = 5) -> pl.DataFrame:
    """Read the CRSP monthly file, keeping enough history to form betas.

    History before `start` is retained so the first month of the sample has a
    full estimation window rather than a truncated one.
    """
    schema = {
        "PERMNO": pl.Int64, "date": pl.Utf8, "SICCD": pl.Int32,
        "RET": pl.Float64, "RETX": pl.Float64, "vwretd": pl.Float64,
        "SHROUT": pl.Int64, "PRC": pl.Float64,
    }
    df = pl.read_csv(path, schema_overrides=schema, try_parse_dates=False,
                     null_values=["", "NA", "NaN", "B", "C"], ignore_errors=True)
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, format="%m/%d/%Y"))
    df = df.with_columns([
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
    ])

    expr = pl.when(pl.col("SICCD").is_null()).then(pl.lit("Other"))
    for lo, hi, name in INDUSTRY_BOUNDS:
        expr = expr.when((pl.col("SICCD") >= lo) & (pl.col("SICCD") <= hi)).then(pl.lit(name))
    df = df.with_columns(expr.otherwise(pl.lit("Other")).alias("industry"))

    df = df.filter((pl.col("year") >= start - lookback_years) & (pl.col("year") <= end))

    # PRC is negative when CRSP records a bid-ask midpoint instead of a trade,
    # so the sign carries no information about size.
    return df.with_columns(
        (pl.col("PRC").abs() * pl.col("SHROUT")).alias("mktcap")
    )


def build_universe(df: pl.DataFrame, mode: str, start: int, end: int,
                   firms_per_industry: int = 10, seed: int = 1337) -> pl.DataFrame:
    """Select the firms to trade.

    `assignment` reproduces the notebook: ten firms per industry, each with a
    complete return history over the whole sample. That filter is applied with
    hindsight, so it keeps only firms that survived to the end of the sample
    and drops every delisting. For a strategy that is the wrong universe, since
    the returns you would actually have earned include the firms that failed.

    `wide` keeps every firm and lets the estimation window decide when a firm
    becomes tradable, which is the decision an investor could have made at the
    time.
    """
    if mode == "wide":
        return df

    if mode != "assignment":
        raise ValueError(f"Unknown universe mode {mode!r}")

    expected = (end - start + 1) * 12
    counts = (df.filter((pl.col("year") >= start) & (pl.col("year") <= end))
                .group_by("PERMNO")
                .agg([pl.len().alias("n_obs"), pl.first("industry").alias("industry")])
                .filter(pl.col("n_obs") >= expected))

    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for industry in sorted(counts["industry"].unique().to_list()):
        permnos = counts.filter(pl.col("industry") == industry)["PERMNO"].to_list()
        if len(permnos) < firms_per_industry:
            continue
        keep.extend(rng.choice(permnos, firms_per_industry, replace=False).tolist())

    return df.filter(pl.col("PERMNO").is_in(keep))


def to_panel(df: pl.DataFrame, ret_col: str) -> pd.DataFrame:
    cols = ["PERMNO", "year", "month", "industry", "mktcap", "vwretd", ret_col]
    panel = df.select([c for c in cols if c in df.columns]).to_pandas()
    panel = panel.rename(columns={ret_col: "ret", "vwretd": "mkt"})
    panel = panel.dropna(subset=["ret", "mkt"])
    return bab.prepare_panel(panel)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a frame as a markdown table without pulling in a dependency."""
    show = df.reset_index() if df.index.name else df.copy()
    formatted = show.apply(
        lambda c: c.map(lambda v: floatfmt.format(v)
                        if isinstance(v, (float, np.floating)) and np.isfinite(v)
                        else ("" if pd.isna(v) else str(v)))
    )
    head = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    rule = "| " + " | ".join("---" for _ in formatted.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in formatted.to_numpy()]
    return "\n".join([head, rule, *body])


def section(title: str) -> None:
    print(f"\n\n## {title}\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--msf", default="MSF_1996_2023.csv", help="CRSP monthly stock file")
    ap.add_argument("--fama", default="FAMA.csv", help="Fama-French factor file")
    ap.add_argument("--betas", default=None,
                    help="CSV of neural betas: PERMNO, year, month, beta")
    ap.add_argument("--start", type=int, default=2018, help="first year traded")
    ap.add_argument("--end", type=int, default=2023, help="last year traded")
    ap.add_argument("--universe", choices=["assignment", "wide"], default="assignment")
    ap.add_argument("--return-col", choices=["RET", "RETX"], default="RET",
                    help="RET includes dividends and is what a holder earns; "
                         "RETX excludes them and is what the betas were fit on")
    ap.add_argument("--portfolios", type=int, default=5)
    ap.add_argument("--cost-bps", type=float, default=10.0,
                    help="one-way cost per unit of weight traded")
    ap.add_argument("--ols-window", type=int, default=60)
    ap.add_argument("--ols-min-periods", type=int, default=36)
    ap.add_argument("--plot", default=None, help="write a cumulative return chart here")
    args = ap.parse_args(argv)

    print(f"# Betting against beta, {args.start}-{args.end}")
    print(f"\nUniverse: {args.universe}. Returns: {args.return_col}. "
          f"Costs: {args.cost_bps:.0f}bps per unit traded.")

    raw = load_crsp(args.msf, args.start, args.end,
                    lookback_years=max(args.ols_window // 12 + 1, 2))
    raw = build_universe(raw, args.universe, args.start, args.end)
    panel = to_panel(raw, args.return_col)

    ff = bab.load_fama_french(args.fama)
    panel = panel.merge(ff, on=["year", "month"], how="left")

    panel["beta_ols"] = bab.rolling_ols_beta(
        panel, window=args.ols_window, min_periods=args.ols_min_periods)

    beta_cols = {"Rolling OLS": "beta_ols"}
    if args.betas:
        neural = pd.read_csv(args.betas)
        missing = {"PERMNO", "year", "month", "beta"} - set(neural.columns)
        if missing:
            print(f"error: --betas is missing {sorted(missing)}", file=sys.stderr)
            return 1
        neural = neural.rename(columns={"beta": "beta_nn"})
        before = len(panel)
        panel = panel.merge(neural[["PERMNO", "year", "month", "beta_nn"]],
                            on=["PERMNO", "year", "month"], how="left")
        assert len(panel) == before, "the beta merge duplicated rows"
        matched = panel["beta_nn"].notna().sum()
        print(f"\nMatched {matched:,} of {before:,} firm-months to a neural beta.")
        if matched == 0:
            print("error: no rows matched. Check that the beta export covers "
                  "these years and uses the same PERMNOs.", file=sys.stderr)
            return 1
        beta_cols["Neural"] = "beta_nn"

    traded = panel[(panel["year"] >= args.start) & (panel["year"] <= args.end)].copy()
    per_month = traded.groupby("t")["PERMNO"].nunique()
    print(f"Trading {len(per_month)} months, median {per_month.median():.0f} "
          f"names per month.")

    # Compare the estimators on the same strategy, which is the question the
    # project is asking: is the learned beta worth trading on?
    factors = traded.groupby("t")[[c for c in ("Mkt-RF", "SMB", "HML")
                                   if c in traded.columns]].first()
    rf = traded.groupby("t")["RF"].first() if "RF" in traded.columns else None

    section("Strategy comparison")
    for weight in ("ew", "vw"):
        table = bab.compare_estimators(
            traded, beta_cols, rf=rf, factors=factors,
            n_portfolios=args.portfolios, weight=weight, cost_bps=args.cost_bps,
            include_bab=(weight == "ew"))  # BAB uses rank weights, so it is the
                                           # same series in both tables
        keep = [c for c in ["n_months", "mean_monthly", "t_stat", "ann_return",
                            "ann_vol", "sharpe", "max_drawdown", "ff_alpha",
                            "ff_alpha_t", "turnover_monthly", "net_mean_monthly",
                            "breakeven_cost_bps"] if c in table.columns]
        print(f"\n### {'Equal' if weight == 'ew' else 'Value'}-weighted\n")
        print(md_table(table[keep].reset_index()))

    section("Portfolio detail")
    for label, col in beta_cols.items():
        for weight in ("ew", "vw"):
            res = bab.quantile_portfolios(traded, beta_col=col,
                                          n_portfolios=args.portfolios,
                                          weight=weight)
            summary = pd.DataFrame({
                "mean_beta": res.mean_beta.mean(),
                "mean_return": res.returns.mean(),
                "t_stat": [bab.newey_west_mean(res.returns[c])["tstat"]
                           for c in res.returns.columns],
                "ann_vol": res.returns.std(ddof=1) * np.sqrt(12),
                "avg_names": res.counts.mean().reindex(res.returns.columns),
            })
            summary.index.name = "portfolio"
            print(f"\n### {label}, {'equal' if weight == 'ew' else 'value'}-weighted\n")
            print(md_table(summary.reset_index()))

    section("BAB factor by year")
    for label, col in beta_cols.items():
        factor = bab.bab_factor(traded, beta_col=col, rf=rf)
        if factor.empty:
            continue
        factor["year"], _ = bab.decode_month(factor.index)
        yearly = factor.groupby("year").agg(
            bab_mean=("bab", "mean"), bab_total=("bab", "sum"),
            beta_low=("beta_low", "mean"), beta_high=("beta_high", "mean"),
            months=("bab", "size"))
        print(f"\n### {label}\n")
        print(md_table(yearly.reset_index()))

    if args.plot:
        _write_plot(traded, beta_cols, rf, args.plot)
        print(f"\nWrote {args.plot}")

    print("\n\nRead the Q5-minus-Q1 rows as market exposure plus an anomaly: the "
          "spread is long roughly one unit of market, so the ff_alpha column, "
          "not mean_monthly, is the CAPM test. The BAB rows are levered to be "
          "beta neutral at formation and need no such adjustment.")
    return 0


def _dates(index) -> pd.DatetimeIndex:
    years, months = bab.decode_month(index)
    return pd.to_datetime([f"{y}-{m:02d}-01" for y, m in zip(years, months)])


def _write_plot(traded: pd.DataFrame, beta_cols: dict[str, str],
                rf: pd.Series | None, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, col in beta_cols.items():
        factor = bab.bab_factor(traded, beta_col=col, rf=rf)
        if factor.empty:
            continue
        ax.plot(_dates(factor.index), (1 + factor["bab"]).cumprod(),
                label=f"{label} BAB")

    mkt = traded.groupby("t")["mkt"].first()
    ax.plot(_dates(mkt.index), (1 + mkt).cumprod(), label="Market",
            linestyle="--", color="gray")

    ax.set_title("Cumulative return, betting against beta versus the market")
    ax.set_ylabel("Growth of 1")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    raise SystemExit(main())
