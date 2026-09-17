"""Option-adjusted spread by Monte Carlo over Hull-White short-rate paths.

The static spread in `mbs.py` answers a narrow question: what spread makes
this pool price correctly *if* rates follow one path, namely the forwards. The
prepayment option does not care about the forwards, it cares about the whole
distribution, and a static spread prices it at zero. OAS fixes that by
averaging over paths:

    price = mean over paths of  sum_t  CF(path, t) * exp(-(r_path + OAS) * dt)

and solving for the OAS that reproduces a given market price. What is left
after taking out the option is compensation for everything else - liquidity,
credit at the agency level, model error.

The short rate follows one-factor Hull-White:

    dr = (theta(t) - a*r) dt + sigma dW

fitted to the initial curve exactly, using the standard decomposition
r(t) = x(t) + alpha(t) with dx = -a*x dt + sigma dW and x(0) = 0, so every
simulated path is consistent with today's discount factors by construction
rather than by calibration. `test_mbs.py` checks that.

On each path the pool's refinancing incentive is recomputed each month from
the simulated 10-year rate plus a fixed primary-mortgage spread, the CPR
follows from the same S-curve the deterministic model uses, and the cash
flows are built month by month. Prepayment is a function of the rate *at that
month on that path*, which is what makes this a genuinely path-dependent
valuation and not a dressed-up average.

    python oas.py --paths 1000 --price 100.0
"""

from __future__ import annotations

import argparse

import numpy as np

import curves
import mbs

# Gap between the 30-year mortgage rate and the 10-year Treasury. Held fixed,
# which is a simplification: it is itself mean-reverting and widens in stress.
PRIMARY_SPREAD = 0.0176

MEAN_REVERSION = 0.05       # a, roughly a 20-year half-life of shocks
VOLATILITY = 0.0100         # sigma, absolute, about 100bp a year


class HullWhite:
    """One-factor Hull-White fitted to an initial zero curve.

    `alpha(t)` absorbs the drift needed to match today's forwards, so the
    stochastic part `x(t)` is a plain Ornstein-Uhlenbeck process starting at
    zero. Bond prices at any future date stay affine in the short rate, which
    is what lets the 10-year rate on each path be read off in closed form
    instead of by a nested simulation.
    """

    def __init__(self, curve: curves.ZeroCurve, a: float = MEAN_REVERSION,
                 sigma: float = VOLATILITY):
        self.curve, self.a, self.sigma = curve, a, sigma

    def forward(self, t, h: float = 1e-4):
        """Instantaneous forward rate f(0,t) = -d ln P(0,t) / dt."""
        t = np.maximum(np.asarray(t, dtype=float), h)
        return -(np.log(self.curve.discount(t + h)) - np.log(self.curve.discount(t - h))) / (2 * h)

    def alpha(self, t):
        t = np.asarray(t, dtype=float)
        adj = (self.sigma ** 2) / (2 * self.a ** 2) * (1.0 - np.exp(-self.a * t)) ** 2
        return self.forward(t) + adj

    def simulate(self, months: int, paths: int, seed: int = 0) -> np.ndarray:
        """Short rates on a monthly grid, shape (paths, months).

        Column k is the rate over month k+1, i.e. the rate that discounts the
        cash flow paid at the end of that month.
        """
        dt = 1.0 / 12.0
        rng = np.random.default_rng(seed)

        # Exact OU transition, so the step size introduces no discretisation
        # error at all. decay and step_sd are the conditional mean factor and
        # standard deviation over one month.
        decay = np.exp(-self.a * dt)
        step_sd = self.sigma * np.sqrt((1.0 - decay ** 2) / (2.0 * self.a))

        x = np.zeros((paths, months))
        current = np.zeros(paths)
        for k in range(months):
            current = decay * current + step_sd * rng.standard_normal(paths)
            x[:, k] = current

        times = (np.arange(months) + 1) * dt
        return x + self.alpha(times)[None, :]

    def zero_rate(self, t, tenor: float, short_rate: np.ndarray) -> np.ndarray:
        """Zero rate for `tenor` years, seen at time t, given the short rate then.

        From the affine bond price P(t, t+T) = A * exp(-B * r(t)) with
        B = (1 - e^{-aT}) / a, the yield is (B*r - ln A) / T.
        """
        a, sig = self.a, self.sigma
        B = (1.0 - np.exp(-a * tenor)) / a
        ratio = self.curve.discount(t + tenor) / self.curve.discount(t)
        ln_a = (np.log(ratio) + B * self.forward(t)
                - (sig ** 2) / (4 * a) * (1.0 - np.exp(-2 * a * t)) * B ** 2)
        return (B * short_rate - ln_a) / tenor


def path_cash_flows(pool: mbs.Pool, dial: mbs.RefiDial, mortgage_rates: np.ndarray,
                    seasoning_psa: float = 1.0, psa: float = 1.0) -> np.ndarray:
    """Investor cash flows for every path, shape (paths, months held).

    Vectorised across paths but sequential in time, because the balance and
    the burnout state both carry forward. `mortgage_rates` is the simulated
    30-year mortgage rate in each month on each path.
    """
    paths, horizon = mortgage_rates.shape
    n = pool.original_term
    i_gross = pool.mortgage_rate / 12.0
    i_net = pool.pass_through / 12.0

    balance = np.full(paths, pool.original_balance)
    burned = np.zeros(paths)
    flows = np.zeros((paths, n - pool.age))

    for k in range(n):
        remaining = n - k
        annuity = 1.0 - (1.0 + i_gross) ** (-remaining)
        payment = balance * i_gross / annuity
        scheduled = np.minimum(payment - balance * i_gross, balance)

        if k < pool.age:
            # Already happened: one history, at the benchmark speed.
            cpr = np.full(paths, mbs.psa_cpr(k + 1, seasoning_psa))
        else:
            incentive = 100.0 * (pool.mortgage_rate - mortgage_rates[:, k - pool.age])
            cpr = np.clip(mbs.psa_cpr(k + 1, psa) * dial(incentive, burned),
                          0.0, mbs.CPR_CAP)

        prepaid = mbs.smm(cpr) * (balance - scheduled)
        if k >= pool.age:
            flows[:, k - pool.age] = balance * i_net + scheduled + prepaid

        burned = burned + prepaid / pool.original_balance
        balance = np.maximum(balance - scheduled - prepaid, 0.0)

    return flows


def price_paths(flows: np.ndarray, short_rates: np.ndarray, spread: float,
                current_face: float) -> float:
    """Mean discounted value across paths, per 100 of current face.

    Discounting uses each path's own realised short rate compounded month by
    month, which is what makes the spread an option-adjusted one: the option
    is already inside `flows`.
    """
    dt = 1.0 / 12.0
    log_df = -np.cumsum((short_rates + spread) * dt, axis=1)
    return 100.0 * float(np.mean(np.sum(flows * np.exp(log_df), axis=1))) / current_face


def solve_oas(flows, short_rates, current_face, target_price: float,
              lo: float = -0.05, hi: float = 0.30) -> float:
    """Bisection on the OAS. Price falls monotonically in the spread."""
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if price_paths(flows, short_rates, mid, current_face) > target_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run(paths: int = 1000, psa: float = 2.0, target_price: float = 100.0,
        seed: int = 0, sigma: float = VOLATILITY, a: float = MEAN_REVERSION):
    curve, mortgage_rate = curves.default_curve()
    pool = mbs.Pool()
    dial = mbs.make_dial(pool, mortgage_rate)

    horizon = pool.original_term - pool.age
    model = HullWhite(curve, a=a, sigma=sigma)
    short = model.simulate(horizon, paths, seed=seed)

    # The mortgage rate a borrower faces in month k on this path: the
    # simulated 10-year rate plus the primary-secondary spread.
    times = (np.arange(horizon) + 1) / 12.0
    ten_year = model.zero_rate(times[None, :], 10.0, short)
    mortgage_paths = ten_year + PRIMARY_SPREAD

    flows = path_cash_flows(pool, dial, mortgage_paths, psa=psa)
    face = mbs.project(pool, psa=psa, dial=dial,
                       incentive=dial.base_incentive).current_face

    oas = solve_oas(flows, short, face, target_price)
    static = mbs.solve_spread(
        mbs.project(pool, psa=psa, dial=dial, incentive=dial.base_incentive),
        curve, target_price)

    return {
        "paths": paths, "psa": psa, "target_price": target_price,
        "oas_bp": oas * 1e4, "static_spread_bp": static * 1e4,
        "option_cost_bp": (static - oas) * 1e4,
        "zero_oas_price": price_paths(flows, short, 0.0, face),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--psa", type=float, default=2.0)
    ap.add_argument("--price", type=float, default=100.0, help="market price to match")
    ap.add_argument("--sigma", type=float, default=VOLATILITY)
    ap.add_argument("--mean-reversion", type=float, default=MEAN_REVERSION)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = run(paths=args.paths, psa=args.psa, target_price=args.price,
              seed=args.seed, sigma=args.sigma, a=args.mean_reversion)

    print(f"{args.paths} Hull-White paths, a={args.mean_reversion}, "
          f"sigma={args.sigma:.2%}, base speed {args.psa*100:.0f} PSA")
    print(f"  market price     {out['target_price']:.3f}")
    print(f"  static spread    {out['static_spread_bp']:.1f}bp   "
          f"(one path: the forwards)")
    print(f"  OAS              {out['oas_bp']:.1f}bp")
    print(f"  option cost      {out['option_cost_bp']:.1f}bp   "
          f"(what the borrower's right to refinance is worth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
