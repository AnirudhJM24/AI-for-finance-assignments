"""Betting-against-beta strategy machinery for the neural beta estimates.

The neural beta notebook produces a beta for each firm-month. This module turns
those betas into portfolios and measures whether they are worth trading:
monthly-rebalanced quintile sorts, a Frazzini-Pedersen beta-neutral BAB factor,
Newey-West t-statistics, factor alphas, turnover, and costs.

Everything here works on a tidy long panel with one row per firm-month:

    PERMNO  year  month  ret  mktcap  beta

`beta` must be formed from information available before `ret` is realised.
`build_keyed_arrays` below enforces that alignment when extracting betas from
the notebook's model.

Returns are decimals (0.03 = 3%), not percent. `load_fama_french` rescales the
Ken French files, which ship in percent, so the two never get mixed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "build_keyed_arrays",
    "decode_month",
    "load_fama_french",
    "prepare_panel",
    "rolling_ols_beta",
    "assign_quantiles",
    "quantile_portfolios",
    "bab_factor",
    "newey_west_mean",
    "factor_alpha",
    "max_drawdown",
    "performance_summary",
    "compare_estimators",
    "PortfolioResult",
]

MONTHS_PER_YEAR = 12
DEFAULT_NW_LAGS = 6


# --------------------------------------------------------------------------
# Getting betas out of the notebook with their keys attached
# --------------------------------------------------------------------------

def build_keyed_arrays(df, lookback: int = 12, fama: bool = False):
    """Build neural-beta training arrays that carry their firm-month keys.

    Same feature construction as `build_neuralbeta_arrays` in the notebook, but
    it also returns the (PERMNO, year, month) of every sample.

    The notebook version reconstructs the mapping afterwards by prepending
    `test_df.height - len(beta_pred)` NaNs to the top of the panel. That count
    is the total number of rows dropped across all firms, but the rows are
    dropped from the head of *each firm's* history, so the offset shifts every
    beta onto the wrong firm-month. Returning the keys removes the need to
    infer the mapping at all.

    Parameters
    ----------
    df : polars.DataFrame
        Must contain PERMNO, year, month, RETX, vwretd and the industry dummy
        columns produced by `to_dummies`.
    lookback : int
        Months of history in each feature window.
    fama : bool
        Append lookback months of Mkt-RF, SMB, HML and RF to each window.

    Returns
    -------
    X : np.ndarray [N, D]
    r_next : np.ndarray [N]      return realised in the sample's month
    mkt_next : np.ndarray [N]    market return in the same month
    keys : pd.DataFrame [N, 3]   PERMNO, year, month of that month

    Features use months t-lookback .. t-1; `r_next` is the month-t return. The
    beta is therefore known before the return it is paired with, which is what
    makes the portfolio sorts implementable.
    """
    needed = {"PERMNO", "year", "month", "RETX", "vwretd"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fama_cols: list[str] = []
    if fama:
        lookup = {c.lower(): c for c in df.columns}
        for want in ("Mkt-RF", "SMB", "HML", "RF"):
            for cand in (want, want.lower(), want.replace("-", "_"),
                         want.replace("-", "_").lower()):
                if cand in df.columns:
                    fama_cols.append(cand)
                    break
                if cand in lookup:
                    fama_cols.append(lookup[cand])
                    break
            else:
                raise ValueError(
                    f"Could not find Fama-French column {want!r}. "
                    f"Columns present: {list(df.columns)}"
                )

    df = df.sort(["PERMNO", "year", "month"])
    industry_cols = [c for c in df.columns if c.startswith("industry_")]

    X_list, r_list, m_list, key_list = [], [], [], []

    for _, g in df.group_by("PERMNO", maintain_order=True):
        g = g.drop_nulls(["RETX", "vwretd"])
        n = g.height
        if n <= lookback:
            continue

        firm_ret = g["RETX"].to_numpy()
        mkt_ret = g["vwretd"].to_numpy()
        dummies = g.select(industry_cols).to_numpy() if industry_cols else None
        ff = g.select(fama_cols).to_numpy() if fama else None

        permno = g["PERMNO"].to_numpy()
        yr = g["year"].to_numpy()
        mo = g["month"].to_numpy()

        for t in range(lookback, n):
            parts = [firm_ret[t - lookback:t], mkt_ret[t - lookback:t]]
            if fama:
                parts.append(ff[t - lookback:t].ravel())
            if dummies is not None:
                parts.append(dummies[t])
            X_list.append(np.concatenate(parts))
            r_list.append(firm_ret[t])
            m_list.append(mkt_ret[t])
            key_list.append((permno[t], yr[t], mo[t]))

    if not X_list:
        raise RuntimeError(
            "No samples constructed. Check the panel, its nulls, and lookback."
        )

    keys = pd.DataFrame(key_list, columns=["PERMNO", "year", "month"])
    return (
        np.stack(X_list).astype(np.float32),
        np.asarray(r_list, dtype=np.float32),
        np.asarray(m_list, dtype=np.float32),
        keys,
    )


# --------------------------------------------------------------------------
# Panel preparation
# --------------------------------------------------------------------------

def load_fama_french(path: str, start: int | None = None,
                     end: int | None = None) -> pd.DataFrame:
    """Read a Ken French monthly factor file and return decimals.

    The distributed files state factors in percent (0.83 means 0.83%). CRSP
    returns are decimals. Mixing the two silently scales the factors by 100,
    which both distorts any regression on them and, if they are fed to a
    network as inputs, swamps every other feature. Divide by 100 once, here.
    """
    ff = pd.read_csv(path)
    ff.columns = [c.strip() for c in ff.columns]

    date_col = next((c for c in ff.columns if c.lower() in ("date", "dateff")),
                    ff.columns[0])
    dates = ff[date_col].astype(int)
    ff["year"] = dates // 100
    ff["month"] = dates % 100
    ff = ff.drop(columns=[date_col])

    value_cols = [c for c in ff.columns if c not in ("year", "month")]
    ff[value_cols] = ff[value_cols].apply(pd.to_numeric, errors="coerce")

    # Monthly equity factors sit within a few percent. A median absolute value
    # above 0.5 can only mean the file is in percent.
    scale = ff[value_cols].abs().stack().median()
    if scale > 0.5:
        ff[value_cols] = ff[value_cols] / 100.0

    if start is not None:
        ff = ff[ff["year"] >= start]
    if end is not None:
        ff = ff[ff["year"] <= end]

    return ff.sort_values(["year", "month"]).reset_index(drop=True)


def prepare_panel(panel: pd.DataFrame, ret_col: str = "ret",
                  beta_col: str | None = None,
                  mktcap_col: str = "mktcap") -> pd.DataFrame:
    """Sort the panel, add a month index, and lag market cap by one month.

    Value weights must use the market cap known at formation. Weighting by the
    contemporaneous cap lets a stock's own return in the holding month set its
    weight in that same month, which mechanically tilts the portfolio toward
    whatever went up.

    `beta_col` is optional because betas are normally merged on afterwards;
    pass it to assert they are already present.
    """
    required = ["PERMNO", "year", "month", ret_col]
    if beta_col is not None:
        # Betas are usually merged on after preparation, so they are only
        # checked when the caller says they should already be here.
        required.append(beta_col)
    for col in required:
        if col not in panel.columns:
            raise ValueError(f"Panel is missing required column {col!r}")

    out = panel.copy()
    out = out.sort_values(["PERMNO", "year", "month"]).reset_index(drop=True)
    # Month index, zero-based within the year, so that December decodes back to
    # December: year = t // 12, month = t % 12 + 1. Encoding month directly
    # would push December to t = (year + 1) * 12 and label it January.
    out["t"] = out["year"].astype(int) * 12 + (out["month"].astype(int) - 1)

    if mktcap_col in out.columns:
        grp = out.groupby("PERMNO", sort=False)
        prev_t = grp["t"].shift(1)
        prev_cap = grp[mktcap_col].shift(1)
        # Only trust the lag when the previous row really is the previous month.
        out["mktcap_lag"] = prev_cap.where(prev_t == out["t"] - 1)

    return out


def decode_month(t) -> tuple:
    """Turn the month index produced by `prepare_panel` back into (year, month)."""
    t = np.asarray(t)
    return (t // 12).astype(int), (t % 12 + 1).astype(int)


def rolling_ols_beta(panel: pd.DataFrame, ret_col: str = "ret",
                     mkt_col: str = "mkt", window: int = 60,
                     min_periods: int = 36) -> pd.Series:
    """Rolling univariate OLS beta, the benchmark the neural beta must beat.

    The window ends at t-1, so the beta is comparable to the neural beta: both
    are known before the month whose return they are used to trade.
    """
    out = panel.sort_values(["PERMNO", "year", "month"])

    r = out[ret_col]
    m = out[mkt_col]
    rm = r * m
    mm = m * m

    def roll(s: pd.Series) -> pd.Series:
        # Shift and roll inside each firm, so a window never spans two firms.
        return s.groupby(out["PERMNO"], sort=False).transform(
            lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean()
        )

    mean_r, mean_m = roll(r), roll(m)
    mean_rm, mean_mm = roll(rm), roll(mm)

    cov = mean_rm - mean_r * mean_m
    var = mean_mm - mean_m * mean_m
    beta = cov / var.where(var > 0)
    return beta.reindex(panel.index)


# --------------------------------------------------------------------------
# Portfolio formation
# --------------------------------------------------------------------------

def assign_quantiles(panel: pd.DataFrame, beta_col: str = "beta",
                     n_portfolios: int = 5,
                     min_names: int | None = None) -> pd.Series:
    """Rank betas *within each month* and cut into n equal-count buckets.

    Ranking across the pooled panel instead would put a 2018 observation and a
    2022 observation in the same bucket, so the resulting "spread" compares
    averages taken over different periods rather than a portfolio held at any
    point in time.
    """
    if min_names is None:
        min_names = n_portfolios * 2

    def cut(s: pd.Series) -> pd.Series:
        valid = s.notna()
        if valid.sum() < min_names:
            return pd.Series(np.nan, index=s.index)
        ranks = s[valid].rank(method="first")
        q = np.ceil(ranks / len(ranks) * n_portfolios)
        return pd.Series(q, index=s[valid].index).reindex(s.index)

    return (panel.groupby("t", sort=False)[beta_col]
                 .transform(cut)
                 .clip(1, n_portfolios))


@dataclass
class PortfolioResult:
    """Monthly returns and diagnostics for one sorted strategy."""
    returns: pd.DataFrame          # index t, columns q1..qN plus 'spread'
    mean_beta: pd.DataFrame        # formation beta per bucket per month
    counts: pd.DataFrame           # names per bucket per month
    weights: dict = field(default_factory=dict)  # bucket -> wide weight frame


def _weight_frame(panel: pd.DataFrame, weight: str, mktcap_lag: str) -> pd.Series:
    if weight == "ew":
        w = pd.Series(1.0, index=panel.index)
    elif weight == "vw":
        if mktcap_lag not in panel.columns:
            raise ValueError(
                "Value weighting needs a lagged market cap; run prepare_panel first."
            )
        w = panel[mktcap_lag].astype(float)
        w = w.where(w > 0)
    else:
        raise ValueError(f"weight must be 'ew' or 'vw', got {weight!r}")
    return w


def quantile_portfolios(panel: pd.DataFrame, beta_col: str = "beta",
                        ret_col: str = "ret", n_portfolios: int = 5,
                        weight: str = "ew",
                        mktcap_lag: str = "mktcap_lag",
                        quantile_col: str | None = None,
                        warn_thin: int = 50) -> PortfolioResult:
    """Form beta-sorted portfolios fresh each month and hold for one month."""
    df = panel.copy()
    if quantile_col is None:
        df["_q"] = assign_quantiles(df, beta_col, n_portfolios)
        quantile_col = "_q"

    df = df.dropna(subset=[quantile_col, ret_col, beta_col])
    df["_w"] = _weight_frame(df, weight, mktcap_lag)
    df = df.dropna(subset=["_w"])

    per_month = df.groupby("t")["PERMNO"].nunique()
    if len(per_month) and per_month.median() < warn_thin:
        warnings.warn(
            f"Median cross-section is {per_month.median():.0f} names per month. "
            f"With {n_portfolios} buckets that is "
            f"~{per_month.median() / n_portfolios:.0f} per portfolio, so the "
            "spread will be dominated by individual stocks rather than a "
            "diversified portfolio.",
            stacklevel=2,
        )

    def agg(g: pd.DataFrame) -> pd.Series:
        w = g["_w"] / g["_w"].sum()
        return pd.Series({
            "ret": float((w * g[ret_col]).sum()),
            "beta": float((w * g[beta_col]).sum()),
            "n": int(len(g)),
        })

    grouped = df.groupby(["t", quantile_col]).apply(agg, include_groups=False)
    wide = grouped.unstack(quantile_col)

    rename = {q: f"q{int(q)}" for q in wide["ret"].columns}
    returns = wide["ret"].rename(columns=rename).sort_index(axis=1)
    betas = wide["beta"].rename(columns=rename).sort_index(axis=1)
    counts = wide["n"].rename(columns=rename).sort_index(axis=1)

    hi, lo = f"q{n_portfolios}", "q1"
    if hi in returns.columns and lo in returns.columns:
        returns["spread"] = returns[hi] - returns[lo]

    weights = {}
    for q, g in df.groupby(quantile_col):
        w = g.copy()
        w["_wn"] = w["_w"] / w.groupby("t")["_w"].transform("sum")
        weights[f"q{int(q)}"] = w.pivot_table(
            index="t", columns="PERMNO", values="_wn", fill_value=0.0
        )

    return PortfolioResult(returns=returns, mean_beta=betas,
                           counts=counts, weights=weights)


def bab_factor(panel: pd.DataFrame, beta_col: str = "beta",
               ret_col: str = "ret", rf: pd.Series | None = None,
               beta_floor: float = 0.25,
               shrink: float = 0.0) -> pd.DataFrame:
    """Frazzini-Pedersen betting-against-beta factor.

    Each month, rank betas and put rank-proportional weights on the two halves:
    long the low-beta half, short the high-beta half, each leg normalised to
    sum to one. Then lever each leg to beta one,

        r_bab = (r_low - rf) / beta_low - (r_high - rf) / beta_high

    so the factor is ex-ante beta neutral. A plain q1-minus-q5 spread is not:
    it is short roughly one unit of market, which is most of why a raw
    high-minus-low spread looks negative in a rising market.

    Parameters
    ----------
    beta_floor : float
        Lower bound on a leg's beta before levering. Without it, a leg beta
        near zero produces unbounded leverage.
    shrink : float
        Optional shrinkage of each beta toward 1 before ranking, as in
        Frazzini-Pedersen's 0.6/0.4 split. 0 leaves betas untouched.

    Returns
    -------
    DataFrame indexed by month with the factor return, both leg returns, the
    leg betas, and the leverage applied to each leg.
    """
    df = panel.dropna(subset=[beta_col, ret_col]).copy()
    if shrink:
        df[beta_col] = (1 - shrink) * df[beta_col] + shrink * 1.0

    rf_map = {} if rf is None else dict(rf)
    rows = []

    for t, g in df.groupby("t", sort=True):
        if len(g) < 4:
            continue
        z = g[beta_col].rank(method="first")
        zbar = z.mean()
        denom = (z - zbar).abs().sum()
        if denom <= 0:
            continue
        k = 2.0 / denom

        w_hi = np.maximum(z - zbar, 0.0) * k
        w_lo = np.maximum(zbar - z, 0.0) * k
        if w_hi.sum() <= 0 or w_lo.sum() <= 0:
            continue
        w_hi = w_hi / w_hi.sum()
        w_lo = w_lo / w_lo.sum()

        b_hi = float((w_hi * g[beta_col]).sum())
        b_lo = float((w_lo * g[beta_col]).sum())
        r_hi = float((w_hi * g[ret_col]).sum())
        r_lo = float((w_lo * g[ret_col]).sum())

        lev_hi = 1.0 / max(abs(b_hi), beta_floor) * np.sign(b_hi or 1.0)
        lev_lo = 1.0 / max(abs(b_lo), beta_floor) * np.sign(b_lo or 1.0)

        r_free = float(rf_map.get(t, 0.0))
        rows.append({
            "t": t,
            "bab": (r_lo - r_free) * lev_lo - (r_hi - r_free) * lev_hi,
            "ret_low": r_lo, "ret_high": r_hi,
            "beta_low": b_lo, "beta_high": b_hi,
            "lev_low": lev_lo, "lev_high": lev_hi,
        })

    return pd.DataFrame(rows).set_index("t")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def _nw_lags(n: int, lags: int | None) -> int:
    if lags is not None:
        return lags
    return max(1, int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0))))


def newey_west_mean(r: pd.Series, lags: int | None = None) -> dict:
    """Mean of a return series with a Newey-West heteroskedasticity- and
    autocorrelation-consistent t-statistic.

    A plain t-test on monthly portfolio returns overstates significance when
    the series is autocorrelated, which sorted portfolios generally are.
    """
    import statsmodels.api as sm

    r = pd.Series(r).dropna()
    n = len(r)
    if n < 3:
        return {"mean": np.nan, "tstat": np.nan, "pvalue": np.nan, "n": n}

    model = sm.OLS(r.values, np.ones(n)).fit(
        cov_type="HAC", cov_kwds={"maxlags": _nw_lags(n, lags)}
    )
    return {
        "mean": float(model.params[0]),
        "tstat": float(model.tvalues[0]),
        "pvalue": float(model.pvalues[0]),
        "n": n,
    }


def factor_alpha(r: pd.Series, factors: pd.DataFrame,
                 lags: int | None = None) -> dict:
    """Regress a return series on factors; report alpha with a Newey-West t.

    `r` should already be in excess of the risk-free rate when the factor set
    contains a market factor stated in excess terms. A long-short or BAB
    series is self-financing and needs no adjustment.
    """
    import statsmodels.api as sm

    joined = pd.concat([pd.Series(r, name="_r"), factors], axis=1).dropna()
    if len(joined) < len(factors.columns) + 3:
        return {"alpha": np.nan, "alpha_t": np.nan, "n": len(joined)}

    y = joined["_r"].values
    X = sm.add_constant(joined[factors.columns].values, has_constant="add")
    model = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": _nw_lags(len(joined), lags)}
    )

    out = {
        "alpha": float(model.params[0]),
        "alpha_t": float(model.tvalues[0]),
        "r2": float(model.rsquared),
        "n": int(len(joined)),
    }
    for i, name in enumerate(factors.columns, start=1):
        out[f"b_{name}"] = float(model.params[i])
        out[f"t_{name}"] = float(model.tvalues[i])
    return out


def max_drawdown(r: pd.Series) -> float:
    """Worst peak-to-trough decline of the compounded series."""
    r = pd.Series(r).dropna()
    if r.empty:
        return np.nan
    wealth = (1.0 + r).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Two-way turnover per rebalance: the total absolute weight traded.

    Weights are compared at the start of consecutive months without drifting
    the old weights by their realised returns. Drift is second-order for
    monthly rebalancing and ignoring it overstates turnover slightly, which
    keeps the cost estimate conservative.
    """
    w = weights.fillna(0.0).sort_index()
    return w.diff().abs().sum(axis=1).iloc[1:]


def performance_summary(r: pd.Series, name: str = "strategy",
                        factors: pd.DataFrame | None = None,
                        turnover_series: pd.Series | None = None,
                        cost_bps: float = 10.0,
                        lags: int | None = None) -> dict:
    """Mean, t-stat, Sharpe, drawdown, alpha, turnover and net-of-cost return."""
    r = pd.Series(r).dropna()
    stats = newey_west_mean(r, lags)

    out = {
        "name": name,
        "n_months": stats["n"],
        "mean_monthly": stats["mean"],
        "t_stat": stats["tstat"],
        "p_value": stats["pvalue"],
        "ann_return": stats["mean"] * MONTHS_PER_YEAR,
        "ann_vol": float(r.std(ddof=1)) * np.sqrt(MONTHS_PER_YEAR),
        "max_drawdown": max_drawdown(r),
    }
    out["sharpe"] = (out["ann_return"] / out["ann_vol"]
                     if out["ann_vol"] > 0 else np.nan)

    if factors is not None and len(factors.columns):
        out.update({f"ff_{k}": v for k, v in factor_alpha(r, factors, lags).items()})

    if turnover_series is not None:
        tno = pd.Series(turnover_series).dropna()
        out["turnover_monthly"] = float(tno.mean())
        cost = tno * (cost_bps / 10_000.0)
        net = (r - cost.reindex(r.index)).dropna()
        if len(net):
            net_stats = newey_west_mean(net, lags)
            out["net_mean_monthly"] = net_stats["mean"]
            out["net_t_stat"] = net_stats["tstat"]
        # The break-even cost is the level that erases the gross return, so it
        # only means anything when there is a gross return to erase.
        if out["turnover_monthly"] > 0 and stats["mean"] > 0:
            out["breakeven_cost_bps"] = (
                stats["mean"] / out["turnover_monthly"] * 10_000.0
            )
        else:
            out["breakeven_cost_bps"] = np.nan

    return out


def compare_estimators(panel: pd.DataFrame, beta_cols: dict[str, str],
                       ret_col: str = "ret", rf: pd.Series | None = None,
                       factors: pd.DataFrame | None = None,
                       n_portfolios: int = 5, weight: str = "ew",
                       cost_bps: float = 10.0,
                       include_bab: bool = True) -> pd.DataFrame:
    """Run the same strategy on several beta estimates and stack the results.

    This is the question the project is really asking: a beta estimator earns
    its keep only if trading on it beats trading on the rolling regression it
    replaces.

    `include_bab` exists because the BAB factor uses rank weights and does not
    depend on the equal-versus-value choice, so repeating it in a second table
    would just duplicate the same rows.
    """
    rows = []
    for label, col in beta_cols.items():
        if col not in panel.columns:
            warnings.warn(f"Skipping {label!r}: no column {col!r} in the panel.")
            continue

        res = quantile_portfolios(panel, beta_col=col, ret_col=ret_col,
                                  n_portfolios=n_portfolios, weight=weight)
        if "spread" in res.returns:
            hi = f"q{n_portfolios}"
            tno = (turnover(res.weights[hi]).add(turnover(res.weights["q1"]),
                                                 fill_value=0.0)
                   if hi in res.weights and "q1" in res.weights else None)
            rows.append(performance_summary(
                res.returns["spread"], f"{label}: Q{n_portfolios}-Q1 ({weight})",
                factors, tno, cost_bps))

        if include_bab:
            factor = bab_factor(panel, beta_col=col, ret_col=ret_col, rf=rf)
            if len(factor):
                rows.append(performance_summary(
                    factor["bab"], f"{label}: BAB", factors, None, cost_bps))

    return pd.DataFrame(rows).set_index("name")
