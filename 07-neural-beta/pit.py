"""Point-in-time panel and features for estimating beta with an MLP.

The whole file exists to enforce one rule: everything known at the moment a
position is opened, and nothing known after it.

Timing convention, used everywhere downstream:

    features for firm i use returns through month t   ->   beta_hat[i, t]
    portfolio formed at the close of month t
    position earns r[i, t+1]

So a row keyed (i, t) carries features from data up to and including t, and a
label `ret_next` realised in t+1. Nothing in a row's features may come from
t+1 or later. `test_pit.py` checks this by construction rather than by reading.

Returns are decimals. Market cap is the value at t, the formation date.
"""

from __future__ import annotations

import numpy as np
import polars as pl

WINDOWS = (12, 36, 60)
MIN_FRACTION = 0.6      # a window needs this share of its months to be usable

INDUSTRY_BOUNDS = [
    (1, 999, "Agriculture"), (1000, 1499, "Mining"), (1500, 1799, "Construction"),
    (2000, 3999, "Manufacturing"), (4000, 4999, "Transportation"),
    (5000, 5199, "Wholesale"), (5200, 5999, "Retail"), (6000, 6799, "Finance"),
    (7000, 8999, "Services"),
]


def load_crsp(path: str) -> pl.DataFrame:
    """Read the CRSP monthly file into a tidy panel.

    Keeps ordinary common shares (10, 11) and drops nothing on the basis of
    what a firm did later, so delisted firms stay in the panel for every month
    they actually traded.
    """
    schema = {
        "PERMNO": pl.Int64, "date": pl.Utf8, "SHRCD": pl.Int32, "SICCD": pl.Int32,
        "RET": pl.Float64, "RETX": pl.Float64, "vwretd": pl.Float64,
        "SHROUT": pl.Int64, "PRC": pl.Float64,
    }
    df = pl.read_csv(
        path, schema_overrides=schema, try_parse_dates=False,
        # CRSP marks an unavailable return with a letter code, not a blank.
        null_values=["", "NA", "NaN", "A", "B", "C", "D", "E", "S", "T", "P"],
        ignore_errors=True,
    )

    sample = df["date"].drop_nulls().head(1).item()
    fmt = "%Y-%m-%d" if len(sample) == 10 and sample[4] == "-" else "%m/%d/%Y"
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, format=fmt))

    if "SHRCD" in df.columns:
        df = df.filter(pl.col("SHRCD").is_in([10, 11]))

    # SICCD 9999 is CRSP's unclassified code, not Public Administration.
    expr = pl.when(pl.col("SICCD").is_null() | pl.col("SICCD").is_in([0, 9999]))
    expr = expr.then(pl.lit("Unclassified"))
    for lo, hi, name in INDUSTRY_BOUNDS:
        expr = expr.when((pl.col("SICCD") >= lo) & (pl.col("SICCD") <= hi)).then(pl.lit(name))

    df = df.with_columns([
        expr.otherwise(pl.lit("Other")).alias("industry"),
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        # PRC is negative when CRSP stores a bid-ask midpoint rather than a
        # trade, so the sign carries no information about size.
        (pl.col("PRC").abs() * pl.col("SHROUT")).alias("mktcap"),
    ])
    df = df.with_columns(
        (pl.col("year") * 12 + pl.col("month") - 1).alias("t")
    )

    return (df.select(["PERMNO", "t", "year", "month", "industry", "RET",
                       "vwretd", "mktcap", "PRC"])
              .rename({"RET": "ret", "vwretd": "mkt"})
              .drop_nulls(["ret", "mkt"])
              .unique(subset=["PERMNO", "t"], keep="first")
              .sort(["PERMNO", "t"]))


def _contiguous_blocks(df: pl.DataFrame) -> pl.DataFrame:
    """Label runs of consecutive months within each firm.

    Rolling windows operate on row position, so a firm that stops trading for
    a year and returns would otherwise have a window spanning the gap as if
    the months were adjacent. Numbering the runs lets every window be computed
    inside one unbroken stretch of months.
    """
    return df.with_columns(
        ((pl.col("t").diff().over("PERMNO") != 1)
         .fill_null(True)
         .cum_sum()
         .alias("block"))
    )


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Attach point-in-time features and the next month's label.

    Every rolling window ends at t inclusive, so a row's features describe the
    past and present only. The label is shifted back from t+1, and is kept only
    where t+1 is genuinely the next calendar month for that firm.
    """
    df = _contiguous_blocks(df.sort(["PERMNO", "t"]))
    key = ["PERMNO", "block"]

    df = df.with_columns([
        (pl.col("ret") * pl.col("mkt")).alias("_rm"),
        (pl.col("mkt") ** 2).alias("_mm"),
    ])

    features: list[str] = []
    for w in WINDOWS:
        p = max(int(w * MIN_FRACTION), 6)
        mean_r = pl.col("ret").rolling_mean(w, min_samples=p).over(key)
        mean_m = pl.col("mkt").rolling_mean(w, min_samples=p).over(key)
        mean_rm = pl.col("_rm").rolling_mean(w, min_samples=p).over(key)
        mean_mm = pl.col("_mm").rolling_mean(w, min_samples=p).over(key)

        cov = mean_rm - mean_r * mean_m
        var = mean_mm - mean_m * mean_m
        sd_r = pl.col("ret").rolling_std(w, min_samples=p).over(key)
        sd_m = pl.col("mkt").rolling_std(w, min_samples=p).over(key)

        df = df.with_columns([
            # The rolling regression beta over this window. Handing the network
            # the benchmark's own estimate makes the comparison a fair one: it
            # can only win by improving on what the regression already sees.
            (cov / pl.when(var > 0).then(var).otherwise(None)).alias(f"beta_{w}"),
            (cov / pl.when((sd_r > 0) & (sd_m > 0))
                     .then(sd_r * sd_m).otherwise(None)).alias(f"corr_{w}"),
            sd_r.alias(f"vol_{w}"),
            mean_r.alias(f"mean_ret_{w}"),
        ])
        features += [f"beta_{w}", f"corr_{w}", f"vol_{w}", f"mean_ret_{w}"]

    # Twelve months of the firm's own and the market's returns, most recent
    # first. Lag 0 is month t, which is known at formation.
    for lag in range(12):
        df = df.with_columns([
            pl.col("ret").shift(lag).over(key).alias(f"ret_lag{lag}"),
            pl.col("mkt").shift(lag).over(key).alias(f"mkt_lag{lag}"),
        ])
        features += [f"ret_lag{lag}", f"mkt_lag{lag}"]

    df = df.with_columns(
        pl.when(pl.col("mktcap") > 0).then(pl.col("mktcap").log())
          .otherwise(None).alias("log_mktcap")
    )
    features.append("log_mktcap")

    industries = sorted(df["industry"].unique().to_list())
    for ind in industries:
        col = f"ind_{ind}"
        df = df.with_columns((pl.col("industry") == ind).cast(pl.Float64).alias(col))
        features.append(col)

    # The label: next month's return and market return, pulled back to row t,
    # and only where the next row really is the next month.
    df = df.with_columns([
        pl.col("ret").shift(-1).over(key).alias("ret_next"),
        pl.col("mkt").shift(-1).over(key).alias("mkt_next"),
        pl.col("t").shift(-1).over(key).alias("_t_next"),
    ])
    df = df.with_columns([
        pl.when(pl.col("_t_next") == pl.col("t") + 1)
          .then(pl.col(c)).otherwise(None).alias(c)
        for c in ("ret_next", "mkt_next")
    ])

    return df.drop(["_rm", "_mm", "_t_next"]), features


def build(path: str, min_year: int | None = None):
    """Load, build features, and drop rows that cannot be traded.

    A row survives only with a full feature vector and a realised next month,
    since a beta with no following return cannot be evaluated or held.
    """
    panel = load_crsp(path)
    panel, features = add_features(panel)

    panel = panel.drop_nulls(features + ["ret_next", "mkt_next"])
    if min_year is not None:
        panel = panel.filter(pl.col("year") >= min_year)

    return panel.sort(["t", "PERMNO"]), features


def to_arrays(panel: pl.DataFrame, features: list[str]):
    """Feature matrix, labels and keys, in panel order."""
    X = panel.select(features).to_numpy().astype(np.float32)
    y = panel["ret_next"].to_numpy().astype(np.float32)
    m = panel["mkt_next"].to_numpy().astype(np.float32)
    t = panel["t"].to_numpy()
    return X, y, m, t
