"""Treasury discount curve and the current mortgage rate.

Two things the pricing model needs from the outside world: a zero curve to
discount cash flows against, and the rate a borrower could refinance into
today. Both come from FRED. Both are cached, and a snapshot is committed so
the model runs with no network at all.

The Treasury series are constant-maturity *par* yields quoted bond-equivalent,
so they are bootstrapped to zeros before anything is discounted. Using them as
if they were zeros would overstate the discount rate on every intermediate
cash flow, and a 30-year pool has 360 of those.

    python curves.py --refresh      # pull from FRED, rewrite the snapshot
    python curves.py                # print the curve currently in use
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.request

import numpy as np

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"

# FRED series id -> maturity in years. DGS1MO is quoted as a coupon-equivalent
# yield, so it slots into the par curve alongside the rest.
TREASURY_SERIES = {
    "DGS1MO": 1 / 12, "DGS3MO": 0.25, "DGS6MO": 0.5, "DGS1": 1.0,
    "DGS2": 2.0, "DGS3": 3.0, "DGS5": 5.0, "DGS7": 7.0,
    "DGS10": 10.0, "DGS20": 20.0, "DGS30": 30.0,
}

SNAPSHOT = pathlib.Path(__file__).with_name("curve_snapshot.json")


def _fred_latest(series: str, start: str = "2026-01-01", timeout: float = 30.0) -> float:
    """Most recent non-missing observation of a FRED series, in percent.

    FRED writes a missing daily value as "." rather than leaving it blank, so
    holidays and the weekly mortgage series both need filtering rather than a
    plain `tail -1`.
    """
    url = FRED_CSV.format(series=series, start=start)
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        rows = fh.read().decode("utf-8").splitlines()

    for line in reversed(rows[1:]):
        _, _, value = line.partition(",")
        if value and value.strip() != ".":
            return float(value)
    raise ValueError(f"no usable observations in FRED series {series}")


def refresh(path: pathlib.Path = SNAPSHOT, start: str = "2026-01-01") -> dict:
    """Download the curve and the mortgage rate, and rewrite the snapshot."""
    par = {}
    for series, maturity in TREASURY_SERIES.items():
        par[f"{maturity:.4g}"] = _fred_latest(series, start)

    snap = {
        "asof": "downloaded",
        "source": "FRED daily constant-maturity Treasury series and MORTGAGE30US",
        "par_yields": par,
        "mortgage30us": _fred_latest("MORTGAGE30US", start),
    }
    path.write_text(json.dumps(snap, indent=2) + "\n")
    return snap


def load(path: pathlib.Path = SNAPSHOT) -> dict:
    """Read the committed snapshot."""
    return json.loads(path.read_text())


def par_curve(snap: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Par yields as (maturities in years, yields as decimals), sorted."""
    snap = snap or load()
    items = sorted((float(k), v / 100.0) for k, v in snap["par_yields"].items())
    return np.array([m for m, _ in items]), np.array([y for _, y in items])


class ZeroCurve:
    """Continuously compounded zero rates, bootstrapped from a par curve.

    The bootstrap runs on a semiannual grid because that is how Treasury
    coupons actually pay. Par yields are interpolated linearly onto that grid
    first, which is the market convention for filling the gaps between quoted
    constant maturities, then discount factors are stripped one period at a
    time:

        1 = c * sum(DF_1..DF_{n-1}) + (1 + c) * DF_n

    Off-grid maturities are read back by linear interpolation of the zero
    rate, flat beyond the last point. A monthly pool cash flow lands off the
    semiannual grid eleven times out of twelve, so this is the hot path.
    """

    def __init__(self, maturities: np.ndarray, zero_rates: np.ndarray, shift: float = 0.0):
        self.maturities = np.asarray(maturities, dtype=float)
        self.zero_rates = np.asarray(zero_rates, dtype=float)
        self.shift = float(shift)

    @classmethod
    def from_par(cls, maturities: np.ndarray, par_yields: np.ndarray,
                 horizon: float = 30.0) -> "ZeroCurve":
        grid = np.arange(0.5, horizon + 1e-9, 0.5)
        par = np.interp(grid, maturities, par_yields)

        discounts = np.empty_like(grid)
        running = 0.0
        for i, y in enumerate(par):
            coupon = y / 2.0
            discounts[i] = (1.0 - coupon * running) / (1.0 + coupon)
            running += discounts[i]

        # -ln(DF)/t is the continuously compounded rate for that maturity.
        zeros = -np.log(discounts) / grid

        # The bootstrap starts at six months; anchor the front with the
        # shortest quoted par yield so cash flows in months 1-5 discount at
        # something closer to the bill rate than to the six-month zero.
        front_rate = 2.0 * np.log1p(par_yields[0] / 2.0)
        return cls(np.concatenate([[1e-6], grid]), np.concatenate([[front_rate], zeros]))

    def shifted(self, basis_points: float) -> "ZeroCurve":
        """A parallel shift of the whole curve, for the shock scenarios."""
        return ZeroCurve(self.maturities, self.zero_rates, self.shift + basis_points / 1e4)

    def zero(self, t):
        t = np.asarray(t, dtype=float)
        return np.interp(t, self.maturities, self.zero_rates) + self.shift

    def discount(self, t):
        """Discount factor to time t, in years."""
        t = np.asarray(t, dtype=float)
        return np.exp(-self.zero(t) * t)


def default_curve() -> tuple[ZeroCurve, float]:
    """The bootstrapped snapshot curve and the current 30-year mortgage rate."""
    snap = load()
    maturities, par = par_curve(snap)
    return ZeroCurve.from_par(maturities, par), snap["mortgage30us"] / 100.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="pull fresh data from FRED")
    args = ap.parse_args()

    if args.refresh:
        refresh()
        print(f"wrote {SNAPSHOT}")

    snap = load()
    curve, mortgage = default_curve()
    print(f"as of {snap.get('asof')}   30y mortgage {mortgage:.2%}")
    print(f"{'tenor':>8}  {'par':>7}  {'zero':>7}  {'DF':>7}")
    for t, y in zip(*par_curve(snap)):
        print(f"{t:8.3f}  {y:6.3%}  {curve.zero(t):6.3%}  {curve.discount(t):7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
