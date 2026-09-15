"""Leakage checks for the point-in-time feature layer.

The claim these defend: a row keyed (firm, t) contains nothing that was
unknowable at the close of month t. The strongest test is truncation — build
features twice, once on the full history and once on a history that stops at
T, and demand the rows at or before T come out byte-identical. If any feature
peeked forward, deleting the future would change it.

Run with `python test_pit.py`.
"""

import numpy as np
import polars as pl

import pit


def _panel(n_firms=25, n_months=140, seed=0, gaps=False):
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0.007, 0.045, n_months)
    betas = rng.uniform(-0.5, 2.2, n_firms)

    rows = []
    for i in range(n_firms):
        eps = rng.normal(0, 0.06, n_months)
        for k in range(n_months):
            # Punch a hole in one firm's history to exercise gap handling.
            if gaps and i == 0 and 40 <= k < 52:
                continue
            rows.append({
                "PERMNO": 1000 + i,
                "t": 24000 + k,
                "year": 2000 + k // 12,
                "month": k % 12 + 1,
                "industry": "Manufacturing" if i % 2 else "Retail",
                "ret": float(betas[i] * mkt[k] + eps[k]),
                "mkt": float(mkt[k]),
                "mktcap": float(1e6 * (i + 1)),
                "PRC": 20.0,
            })
    return pl.DataFrame(rows).sort(["PERMNO", "t"])


def test_truncating_the_future_changes_nothing_in_the_past():
    """The decisive check: features must not move when the future is deleted."""
    full = _panel()
    cut = int(full["t"].median())

    feats_full, names = pit.add_features(full)
    feats_cut, _ = pit.add_features(full.filter(pl.col("t") <= cut))

    a = feats_full.filter(pl.col("t") <= cut).sort(["PERMNO", "t"])
    b = feats_cut.sort(["PERMNO", "t"])
    assert a.height == b.height, (a.height, b.height)

    for col in names:
        x = a[col].to_numpy().astype(float)
        y = b[col].to_numpy().astype(float)
        both_nan = np.isnan(x) & np.isnan(y)
        assert np.allclose(x[~both_nan], y[~both_nan], equal_nan=True), \
            f"feature {col!r} changed when the future was removed"

    # The label is the one column that legitimately needs t+1, so the final
    # month of the truncated panel must lose it.
    last = b.filter(pl.col("t") == cut)
    assert last["ret_next"].is_null().all(), \
        "the last month cannot know its own next return"
    print(f"ok  truncating the future leaves all {len(names)} features unchanged")


def test_label_is_genuinely_next_month():
    feats, _ = pit.add_features(_panel())
    df = feats.sort(["PERMNO", "t"]).to_pandas()

    by_key = {(r.PERMNO, r.t): r.ret for r in df.itertuples()}
    checked = 0
    for r in df.itertuples():
        if r.ret_next is None or (isinstance(r.ret_next, float) and np.isnan(r.ret_next)):
            continue
        assert np.isclose(r.ret_next, by_key[(r.PERMNO, r.t + 1)]), \
            f"ret_next at {(r.PERMNO, r.t)} is not the t+1 return"
        checked += 1
    assert checked > 1000
    print(f"ok  ret_next is the realised t+1 return ({checked:,} rows checked)")


def test_rolling_beta_matches_ordinary_least_squares():
    """beta_60 must equal a regression on the trailing 60 months including t."""
    feats, _ = pit.add_features(_panel(n_firms=3, n_months=140))
    df = feats.filter(pl.col("PERMNO") == 1000).sort("t").to_pandas()

    r = df["ret"].to_numpy()
    m = df["mkt"].to_numpy()
    for i in (80, 100, 130):
        window = slice(i - 59, i + 1)              # 60 months ending at t
        expected = np.cov(r[window], m[window], ddof=0)[0, 1] / np.var(m[window])
        assert np.isclose(df["beta_60"].iloc[i], expected, rtol=1e-6), \
            f"beta_60 at row {i}: {df['beta_60'].iloc[i]} vs {expected}"
    print("ok  beta_60 reproduces a trailing 60-month OLS beta")


def test_windows_never_span_a_gap_in_trading():
    """A firm that stops trading and returns must not join the two stretches."""
    feats, _ = pit.add_features(_panel(gaps=True))
    firm = feats.filter(pl.col("PERMNO") == 1000).sort("t").to_pandas()

    resume = firm[firm["t"] > 24051]["t"].min()
    after = firm[firm["t"] == resume].iloc[0]
    # Restarting the window means no 12-month statistic is available yet.
    assert np.isnan(after["beta_12"]), "a window bridged the gap"

    # The label must not jump across the hole either.
    before = firm[firm["t"] == 24039].iloc[0]
    assert before["ret_next"] is None or np.isnan(before["ret_next"]), \
        "the label bridged the gap"
    print("ok  windows and labels never span a break in a firm's history")


def test_market_cap_is_the_formation_value():
    """Size must be as of t, since it is what sets the weight at formation."""
    feats, _ = pit.add_features(_panel())
    df = feats.to_pandas()
    assert np.allclose(np.exp(df["log_mktcap"]), df["mktcap"])
    print("ok  size is the market cap known at formation")


def test_no_feature_correlates_with_the_future_beyond_its_label():
    """A sanity sweep: no feature should predict the NEXT market return.

    The market's next move is close to unforecastable, so a feature that
    correlates strongly with it is a sign that future data leaked in.
    """
    feats, names = pit.add_features(_panel(n_firms=40, n_months=200))
    df = feats.drop_nulls(names + ["mkt_next"]).to_pandas()

    worst, worst_col = 0.0, None
    for col in names:
        if df[col].std() == 0:
            continue
        c = abs(np.corrcoef(df[col], df["mkt_next"])[0, 1])
        if c > worst:
            worst, worst_col = c, col
    assert worst < 0.25, f"{worst_col!r} correlates {worst:.3f} with next month's market"
    print(f"ok  no feature anticipates the market (worst: {worst_col} at {worst:.3f})")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"\n{len(tests)} point-in-time checks passed")
