"""Cash flows, pricing, and risk for a 30-year agency mortgage pass-through.

The pool amortizes like an ordinary mortgage and then loses balance faster
than that, because borrowers prepay. Everything interesting about an MBS lives
in that second part.

Two prepayment models share one engine:

    PSA         a fixed seasoning ramp. CPR rises 0.2% a month for 30 months
                to 6%, then holds. A "200 PSA" pool runs at twice that.
    Refi-aware  the PSA ramp multiplied by an S-curve in the refinancing
                incentive, so the speed responds to where mortgage rates are
                rather than sitting still.

Only the second one produces the behaviour the asset is actually known for.
Under a static speed the pool is a slightly odd amortizing bond with positive
convexity. Turn the S-curve on and a rally shortens it while a selloff extends
it, both against the holder, and convexity goes negative.

Conventions, applied throughout:

    balances are in dollars, rates are decimals, time is in months
    month t runs from balance B[t-1] to B[t] and pays at the end of t
    loan age in month t is t, counted from origination
    prepayment applies to the balance left *after* scheduled principal
    the investor earns the pass-through coupon; the gap to the mortgage
      rate is the servicing and guarantee strip and never reaches them
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

import curves

PSA_RAMP_MONTHS = 30        # months for the benchmark ramp to reach its plateau
PSA_TERMINAL_CPR = 0.06     # the plateau, under 100 PSA
CPR_CAP = 0.80              # nobody prepays faster than this, whatever the model says


# --------------------------------------------------------------------------
# prepayment
# --------------------------------------------------------------------------

def psa_cpr(loan_age: np.ndarray, psa: float = 1.0) -> np.ndarray:
    """Annual CPR from the PSA benchmark ramp.

    `psa` is a multiple, not a percent: 100 PSA is 1.0, 200 PSA is 2.0.
    """
    age = np.asarray(loan_age, dtype=float)
    ramp = np.minimum(age / PSA_RAMP_MONTHS, 1.0)
    return np.clip(psa * PSA_TERMINAL_CPR * ramp, 0.0, CPR_CAP)


def smm(cpr):
    """Single monthly mortality from an annual CPR.

    The twelfth root, not a twelfth: prepayment compounds on a shrinking
    balance, so 6% CPR is 0.514% a month rather than 0.5%.
    """
    return 1.0 - (1.0 - np.asarray(cpr, dtype=float)) ** (1.0 / 12.0)


def cpr_from_smm(s):
    """Inverse of `smm`, for reading a monthly factor back as an annual rate."""
    return 1.0 - (1.0 - np.asarray(s, dtype=float)) ** 12


@dataclass(frozen=True)
class RefiDial:
    """S-curve linking the refinancing incentive to a speed multiplier.

    The incentive is the borrower's own note rate less the rate they could
    refinance into, in percentage points, so a positive number means the loan
    is in the money. The response is deliberately not linear:

        deeply out of the money   only turnover prepays - moving house,
                                  divorce, death. The curve flattens out.
        near zero                 a threshold region; the closing costs have
                                  to be earned back before anyone bothers
        deeply in the money       everyone who can refinance already has, so
                                  the curve flattens again at the top

    The raw logistic is normalised so the multiplier is exactly 1.0 at
    `base_incentive`, the incentive prevailing on the pricing date. That keeps
    the scenario labels honest - a "200 PSA" pool really does run at 200 PSA
    in the base case, and the dial only expresses the *response* to a rate
    move. Without the normalisation every quoted speed would silently be
    something else.
    """

    base_incentive: float = 0.0
    floor: float = 0.40         # turnover-only speed, deeply out of the money
    ceiling: float = 4.00       # saturation speed, deeply in the money
    threshold: float = 0.75     # incentive, in pp, at the curve's midpoint
    steepness: float = 2.60     # how sharply borrowers react around it
    burnout: float = 1.50       # decay of the response as the pool refis away

    def _raw(self, incentive):
        x = np.asarray(incentive, dtype=float)
        span = self.ceiling - self.floor
        return self.floor + span / (1.0 + np.exp(-self.steepness * (x - self.threshold)))

    def __call__(self, incentive, refi_burned: float = 0.0):
        """Speed multiplier at an incentive, after burnout.

        `refi_burned` is the fraction of the original balance already prepaid
        voluntarily. Burnout decays the *excess* over the base speed and
        leaves the base alone: the borrowers who were going to refinance have
        gone, but the ones moving house still move house.
        """
        dial = self._raw(incentive) / self._raw(self.base_incentive)
        excess = dial - 1.0
        damped = np.where(excess > 0, excess * np.exp(-self.burnout * refi_burned), excess)
        return 1.0 + damped


# --------------------------------------------------------------------------
# the pool
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Pool:
    """A generic 30-year fixed-rate agency pass-through."""

    original_balance: float = 100_000_000.0
    mortgage_rate: float = 0.065        # what the borrower pays
    pass_through: float = 0.060         # what the investor receives
    original_term: int = 360
    age: int = 12                       # months since origination at settlement

    @property
    def servicing_strip(self) -> float:
        """The annual fee skimmed between the two rates."""
        return self.mortgage_rate - self.pass_through


@dataclass
class CashFlows:
    """Month-by-month output of `project`, from origination.

    Every array is length `original_term` and indexed by loan age minus one.
    `settle` marks the first month the investor owns, so the pricing views are
    all `[settle:]` slices of these.
    """

    month: np.ndarray            # loan age, 1-based
    begin_balance: np.ndarray
    end_balance: np.ndarray
    scheduled_principal: np.ndarray
    prepayment: np.ndarray
    net_interest: np.ndarray     # to the investor, at the pass-through coupon
    servicing: np.ndarray        # to the servicer and the guarantor
    cpr: np.ndarray
    settle: int

    @property
    def principal(self) -> np.ndarray:
        return self.scheduled_principal + self.prepayment

    @property
    def total(self) -> np.ndarray:
        """Investor cash flow: coupon interest plus all principal."""
        return self.net_interest + self.principal

    def held(self) -> "CashFlows":
        """The part of the schedule the investor actually receives."""
        s = self.settle
        return CashFlows(
            month=self.month[s:] - s, begin_balance=self.begin_balance[s:],
            end_balance=self.end_balance[s:],
            scheduled_principal=self.scheduled_principal[s:],
            prepayment=self.prepayment[s:], net_interest=self.net_interest[s:],
            servicing=self.servicing[s:], cpr=self.cpr[s:], settle=0,
        )

    @property
    def current_face(self) -> float:
        """Balance outstanding at settlement, the base for a quoted price."""
        return float(self.begin_balance[self.settle])


def project(pool: Pool, psa: float = 1.0, dial: RefiDial | None = None,
            incentive: float = 0.0, seasoning_psa: float = 1.0) -> CashFlows:
    """Run the pool from origination to maturity under one speed assumption.

    The months before settlement already happened, so they run at
    `seasoning_psa` and ignore the dial whatever scenario is being projected.
    That keeps the current face a single fact about the pool instead of a
    different number in every column of the results table, which is what a
    price per 100 of current face has to be quoted against.

    The loop is sequential rather than vectorised because burnout makes the
    speed depend on how much has already prepaid. With no dial the recursion
    is still needed for the balance, and 360 iterations costs nothing.

    The scheduled payment is recomputed every month from the balance actually
    outstanding and the term actually remaining. That is what keeps the
    amortization consistent after a prepayment: the survivors carry on paying
    the same loans, and the pool-level payment just scales down.
    """
    n = pool.original_term
    i_gross = pool.mortgage_rate / 12.0
    i_net = pool.pass_through / 12.0
    i_strip = pool.servicing_strip / 12.0

    begin = np.zeros(n); end = np.zeros(n)
    sched = np.zeros(n); prepaid = np.zeros(n)
    net_int = np.zeros(n); strip = np.zeros(n); cpr_out = np.zeros(n)

    balance = pool.original_balance
    burned = 0.0

    for k in range(n):
        remaining = n - k                      # months left, including this one
        begin[k] = balance

        # Level payment on the current balance over the remaining term.
        annuity = 1.0 - (1.0 + i_gross) ** (-remaining)
        payment = balance * i_gross / annuity if annuity > 0 else balance * (1 + i_gross)

        gross_interest = balance * i_gross
        sched[k] = min(payment - gross_interest, balance)

        historical = k < pool.age
        base_cpr = psa_cpr(k + 1, seasoning_psa if historical else psa)
        multiplier = 1.0 if (historical or dial is None) else float(dial(incentive, burned))
        cpr = float(np.clip(base_cpr * multiplier, 0.0, CPR_CAP))
        cpr_out[k] = cpr

        # Prepayment hits what is left once the scheduled payment has landed.
        prepaid[k] = smm(cpr) * (balance - sched[k])

        net_int[k] = balance * i_net
        strip[k] = balance * i_strip

        balance = balance - sched[k] - prepaid[k]
        end[k] = balance
        burned += prepaid[k] / pool.original_balance

        if balance <= 1e-9:
            balance = 0.0

    return CashFlows(
        month=np.arange(1, n + 1), begin_balance=begin, end_balance=end,
        scheduled_principal=sched, prepayment=prepaid, net_interest=net_int,
        servicing=strip, cpr=cpr_out, settle=pool.age,
    )


# --------------------------------------------------------------------------
# valuation
# --------------------------------------------------------------------------

def price(cf: CashFlows, curve: curves.ZeroCurve, spread: float) -> float:
    """Price per 100 of current face, discounting at the zero curve plus spread.

    The spread is a flat add-on to the continuously compounded zero rate at
    each cash flow's own maturity - a static spread, not an OAS. It is held
    fixed across the shock scenarios, which is what makes the duration below
    an *effective* duration of this cash-flow model rather than a
    prepayment-blind modified duration.
    """
    held = cf.held()
    t = held.month / 12.0
    discounts = np.exp(-(curve.zero(t) + spread) * t)
    return 100.0 * float(held.total @ discounts) / cf.current_face


def wal(cf: CashFlows) -> float:
    """Weighted-average life in years: principal-weighted mean time to repayment."""
    held = cf.held()
    principal = held.principal
    total = principal.sum()
    if total <= 0:
        return float("nan")
    return float((held.month / 12.0) @ principal / total)


def solve_spread(cf: CashFlows, curve: curves.ZeroCurve, target_price: float,
                 lo: float = -0.05, hi: float = 0.30) -> float:
    """Static spread that reproduces a given price. Monotone, so bisection is enough."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if price(cf, curve, mid) > target_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass
class RiskResult:
    price: float
    wal: float
    effective_duration: float
    effective_convexity: float
    price_up: float
    price_down: float
    wal_up: float
    wal_down: float


def shock(pool: Pool, curve: curves.ZeroCurve, psa: float, spread: float,
          basis_points: float, dial: RefiDial | None, mortgage_rate: float,
          seasoning_psa: float = 1.0) -> CashFlows:
    """Reprice the pool with the curve and the mortgage rate both moved.

    The link between the two is the whole point. A parallel shift in Treasuries
    drags the primary mortgage rate with it one-for-one, which moves the
    refinancing incentive, which moves the speed. Shocking the discount curve
    while leaving the mortgage rate alone would hide the feature being measured.
    """
    shifted_mortgage = mortgage_rate + basis_points / 1e4
    incentive = 100.0 * (pool.mortgage_rate - shifted_mortgage)   # percentage points
    return project(pool, psa=psa, dial=dial, incentive=incentive,
                   seasoning_psa=seasoning_psa)


def risk(pool: Pool, curve: curves.ZeroCurve, psa: float, spread: float,
         mortgage_rate: float, dial: RefiDial | None = None,
         bump_bp: float = 50.0) -> RiskResult:
    """Price, WAL, and effective duration and convexity from +/- `bump_bp` shocks.

        D_eff = (P- - P+) / (2 * P0 * dy)
        C_eff = (P+ + P- - 2*P0) / (P0 * dy^2)

    Both are computed by full revaluation: the cash flows are rebuilt from
    scratch in each scenario rather than held fixed and rediscounted.
    """
    dy = bump_bp / 1e4

    base = shock(pool, curve, psa, spread, 0.0, dial, mortgage_rate)
    up = shock(pool, curve, psa, spread, +bump_bp, dial, mortgage_rate)
    down = shock(pool, curve, psa, spread, -bump_bp, dial, mortgage_rate)

    p0 = price(base, curve, spread)
    p_up = price(up, curve.shifted(+bump_bp), spread)
    p_dn = price(down, curve.shifted(-bump_bp), spread)

    return RiskResult(
        price=p0, wal=wal(base),
        effective_duration=(p_dn - p_up) / (2.0 * p0 * dy),
        effective_convexity=(p_up + p_dn - 2.0 * p0) / (p0 * dy * dy),
        price_up=p_up, price_down=p_dn, wal_up=wal(up), wal_down=wal(down),
    )


def price_ladder(pool: Pool, curve: curves.ZeroCurve, psa: float, spread: float,
                 mortgage_rate: float, shocks_bp: np.ndarray,
                 dial: RefiDial | None = None) -> np.ndarray:
    """Price at each shock in `shocks_bp`, for the convexity chart."""
    out = []
    for bp in shocks_bp:
        cf = shock(pool, curve, psa, spread, float(bp), dial, mortgage_rate)
        out.append(price(cf, curve.shifted(float(bp)), spread))
    return np.array(out)


def make_dial(pool: Pool, mortgage_rate: float, **kwargs) -> RefiDial:
    """A dial normalised to today's incentive for this pool."""
    base = 100.0 * (pool.mortgage_rate - mortgage_rate)
    return replace(RefiDial(**kwargs), base_incentive=base)
