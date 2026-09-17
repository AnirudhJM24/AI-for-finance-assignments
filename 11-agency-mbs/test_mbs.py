"""Checks on the cash-flow engine, the curve, and the Hull-White simulation.

These are arithmetic identities and known closed forms, not regression tests
against last night's numbers. Each one would catch a specific way of getting
the model wrong:

    balances that do not reconcile with what was paid
    interest paid on the wrong balance, or the servicing strip leaking to
      the investor
    SMM taken as CPR/12 instead of the twelfth root
    a par bond that does not reprice at par off the bootstrapped curve
    a simulated short rate that quietly stops matching today's discount curve

Run with `python test_mbs.py`.
"""

import numpy as np

import curves
import mbs
import oas


# --------------------------------------------------------------------------
# prepayment arithmetic
# --------------------------------------------------------------------------

def test_psa_ramp_hits_its_documented_points():
    """100 PSA: 0.2% a month for 30 months, then flat at 6%."""
    assert abs(mbs.psa_cpr(1, 1.0) - 0.002) < 1e-12
    assert abs(mbs.psa_cpr(15, 1.0) - 0.030) < 1e-12
    assert abs(mbs.psa_cpr(30, 1.0) - 0.060) < 1e-12
    assert abs(mbs.psa_cpr(200, 1.0) - 0.060) < 1e-12
    assert abs(mbs.psa_cpr(30, 2.0) - 0.120) < 1e-12
    print("ok  PSA ramp matches the benchmark at 1, 15, 30, and 200 months")


def test_smm_compounds_rather_than_divides():
    """The twelfth root, not a twelfth - the difference is the whole point."""
    assert abs(mbs.smm(0.06) - 0.005143) < 1e-6
    # Higher than CPR/12, not lower: each month's prepayment applies to a
    # balance the earlier months have already shrunk, so reaching 6% over the
    # year takes slightly more than 0.5% a month.
    assert mbs.smm(0.06) > 0.06 / 12
    for cpr in (0.0, 0.02, 0.06, 0.25, 0.60):
        assert abs(mbs.cpr_from_smm(mbs.smm(cpr)) - cpr) < 1e-12
    print("ok  SMM compounds to CPR and inverts cleanly")


def test_refi_dial_is_one_at_its_base_and_rises_with_incentive():
    dial = mbs.RefiDial(base_incentive=-0.25)
    assert abs(float(dial(-0.25)) - 1.0) < 1e-12, "base case must be exactly the quoted speed"

    grid = np.linspace(-3.0, 3.0, 61)
    values = np.array([float(dial(x)) for x in grid])
    assert np.all(np.diff(values) > 0), "speed must rise monotonically with incentive"
    assert values[0] < 1.0 < values[-1]
    print(f"ok  refi dial spans {values[0]:.2f}x to {values[-1]:.2f}x, monotone, 1.00x at base")


def test_burnout_only_damps_the_excess_speed():
    """Borrowers who would refinance leave; borrowers who move house still move."""
    dial = mbs.RefiDial(base_incentive=0.0)
    fast_fresh, fast_burned = float(dial(1.5, 0.0)), float(dial(1.5, 0.5))
    assert fast_burned < fast_fresh
    assert fast_burned > 1.0, "burnout must not push a fast pool below its base speed"

    slow_fresh, slow_burned = float(dial(-1.5, 0.0)), float(dial(-1.5, 0.5))
    assert abs(slow_fresh - slow_burned) < 1e-12, "burnout must not touch turnover"
    print(f"ok  burnout cuts {fast_fresh:.2f}x to {fast_burned:.2f}x and leaves turnover alone")


# --------------------------------------------------------------------------
# cash flows
# --------------------------------------------------------------------------

def test_balances_reconcile_every_month():
    cf = mbs.project(mbs.Pool(), psa=2.0)
    closing = cf.begin_balance - cf.scheduled_principal - cf.prepayment
    assert np.allclose(closing, cf.end_balance, atol=1e-6)
    assert np.allclose(cf.begin_balance[1:], cf.end_balance[:-1], atol=1e-6)
    assert cf.begin_balance[0] == mbs.Pool().original_balance
    assert cf.end_balance[-1] < 1e-6, "the pool must fully retire by its final month"
    assert np.all(cf.prepayment >= -1e-9) and np.all(cf.scheduled_principal >= -1e-9)
    print("ok  balances roll forward and the pool retires to zero")


def test_all_principal_is_repaid_exactly_once():
    for psa in (0.0, 1.0, 3.0):
        cf = mbs.project(mbs.Pool(), psa=psa)
        assert abs(cf.principal.sum() - mbs.Pool().original_balance) < 1e-4
    print("ok  principal sums to the original balance at 0, 100, and 300 PSA")


def test_zero_prepayment_reproduces_the_textbook_annuity():
    """With no prepayment the pool is a plain 30-year mortgage.

    The seasoning speed has to be switched off too, or the first twelve months
    prepay at the benchmark and the level payment steps down after them.
    """
    pool = mbs.Pool()
    cf = mbs.project(pool, psa=0.0, seasoning_psa=0.0)

    i = pool.mortgage_rate / 12.0
    payment = pool.original_balance * i / (1.0 - (1.0 + i) ** -pool.original_term)
    gross = cf.begin_balance * i
    assert np.allclose(cf.scheduled_principal + gross, payment, atol=1e-6)

    # Closed-form remaining balance after k payments.
    for k in (12, 120, 359):
        expected = pool.original_balance * (
            (1 + i) ** pool.original_term - (1 + i) ** k
        ) / ((1 + i) ** pool.original_term - 1)
        assert abs(cf.end_balance[k - 1] - expected) < 1e-4
    print(f"ok  level payment ${payment:,.2f}/mo and balances match the annuity formula")


def test_the_investor_receives_the_coupon_and_the_servicer_keeps_the_strip():
    pool = mbs.Pool()
    cf = mbs.project(pool, psa=2.0)
    assert np.allclose(cf.net_interest, cf.begin_balance * pool.pass_through / 12, atol=1e-6)
    assert np.allclose(cf.servicing, cf.begin_balance * pool.servicing_strip / 12, atol=1e-6)
    gross = cf.begin_balance * pool.mortgage_rate / 12
    assert np.allclose(cf.net_interest + cf.servicing, gross, atol=1e-6)
    assert abs(cf.servicing.sum() / cf.net_interest.sum() - 0.5 / 6.0) < 1e-6
    print("ok  interest splits into a 6.00% coupon and a 50bp strip on the same balance")


def test_realised_cpr_matches_the_assumption():
    """Back out the speed from the balances and check it is what was asked for.

    Also pins the seasoning split: the twelve months before settlement run at
    the benchmark 100 PSA whatever scenario is being projected, and only the
    months from settlement onward run at the scenario speed.
    """
    pool = mbs.Pool()
    cf = mbs.project(pool, psa=2.0, seasoning_psa=1.0)
    surviving = cf.begin_balance - cf.scheduled_principal
    realised = mbs.cpr_from_smm(cf.prepayment[:100] / surviving[:100])

    expected = np.where(cf.month[:100] <= pool.age,
                        mbs.psa_cpr(cf.month[:100], 1.0),
                        mbs.psa_cpr(cf.month[:100], 2.0))
    assert np.allclose(realised, expected, atol=1e-9)
    print("ok  CPR implied by the cash flows is 100 PSA while seasoning, 200 PSA after")


def test_faster_speeds_shorten_the_pool():
    wals = [mbs.wal(mbs.project(mbs.Pool(), psa=p)) for p in (1.0, 2.0, 3.0)]
    assert wals[0] > wals[1] > wals[2]
    assert 9.0 < wals[0] < 12.0, f"100 PSA WAL of {wals[0]:.2f} is outside the plausible range"
    print(f"ok  WAL falls {wals[0]:.2f} -> {wals[1]:.2f} -> {wals[2]:.2f} years as speed doubles and triples")


def test_seasoning_is_the_same_history_in_every_scenario():
    """Current face is a fact about the pool, not about the forward scenario."""
    faces = {p: mbs.project(mbs.Pool(), psa=p).current_face for p in (1.0, 2.0, 3.0)}
    assert len(set(round(f, 6) for f in faces.values())) == 1
    print(f"ok  current face is ${list(faces.values())[0]/1e6:.2f}mm in all three scenarios")


# --------------------------------------------------------------------------
# curve and pricing
# --------------------------------------------------------------------------

def test_bootstrapped_curve_reprices_par_bonds_at_par():
    maturities, par = curves.par_curve()
    curve = curves.ZeroCurve.from_par(maturities, par)

    for maturity, yld in zip(maturities, par):
        if maturity < 0.5:
            continue
        times = np.arange(0.5, maturity + 1e-9, 0.5)
        value = (yld / 2) * curve.discount(times).sum() + curve.discount(maturity)
        assert abs(value - 1.0) < 1e-8, f"{maturity}y par bond prices at {value:.6f}"
    print("ok  every quoted par bond reprices to 1.000000 off the bootstrapped zeros")


def test_discount_factors_fall_with_maturity():
    curve, _ = curves.default_curve()
    df = curve.discount(np.arange(0.5, 30.5, 0.5))
    assert np.all(np.diff(df) < 0) and np.all(df > 0) and df[0] < 1.0
    print("ok  discount factors are positive and strictly decreasing out to 30 years")


def test_solved_spread_reproduces_its_target_price():
    curve, _ = curves.default_curve()
    cf = mbs.project(mbs.Pool(), psa=2.0)
    for target in (95.0, 100.0, 104.0):
        spread = mbs.solve_spread(cf, curve, target)
        assert abs(mbs.price(cf, curve, spread) - target) < 1e-6
    print("ok  the solved static spread reprices the pool to 95, 100, and 104")


def test_price_falls_as_the_spread_widens():
    curve, _ = curves.default_curve()
    cf = mbs.project(mbs.Pool(), psa=2.0)
    prices = [mbs.price(cf, curve, s) for s in (0.00, 0.01, 0.02, 0.04)]
    assert np.all(np.diff(prices) < 0)
    print("ok  price is monotonically decreasing in the discount spread")


# --------------------------------------------------------------------------
# the result the model exists to produce
# --------------------------------------------------------------------------

def test_a_fixed_speed_pool_has_positive_convexity():
    """Turn the prepayment option off and an MBS is an ordinary bond."""
    curve, mortgage_rate = curves.default_curve()
    pool = mbs.Pool()
    for psa in (1.0, 2.0, 3.0):
        r = mbs.risk(pool, curve, psa, 0.01, mortgage_rate, dial=None)
        assert r.effective_duration > 0
        assert r.effective_convexity > 0, f"{psa*100:.0f} PSA: C = {r.effective_convexity:.1f}"
    print("ok  with prepayment held fixed, convexity is positive at every speed")


def test_rate_driven_prepayment_turns_convexity_negative():
    """The headline: link the speed to rates and the sign flips."""
    curve, mortgage_rate = curves.default_curve()
    pool = mbs.Pool()
    dial = mbs.make_dial(pool, mortgage_rate)

    for psa in (1.0, 2.0, 3.0):
        live = mbs.risk(pool, curve, psa, 0.01, mortgage_rate, dial)
        fixed = mbs.risk(pool, curve, psa, 0.01, mortgage_rate, None)
        assert live.effective_convexity < 0, f"{psa*100:.0f} PSA: C = {live.effective_convexity:.1f}"
        assert live.effective_convexity < fixed.effective_convexity

        # Contraction on the way down, extension on the way up.
        assert live.wal_down < live.wal < live.wal_up
        # The holder loses on both sides relative to a symmetric response.
        gain = live.price_down - live.price
        loss = live.price - live.price_up
        assert gain < loss, f"{psa*100:.0f} PSA: rally gain {gain:.3f} >= selloff loss {loss:.3f}"
    print("ok  responsive prepayment gives negative convexity, contraction, and extension")


def test_the_pool_prices_below_a_comparable_non_callable_bond():
    """The prepayment option is worth something, and the holder is short it."""
    curve, mortgage_rate = curves.default_curve()
    pool = mbs.Pool()
    dial = mbs.make_dial(pool, mortgage_rate)
    cf = mbs.project(pool, psa=2.0, dial=dial, incentive=dial.base_incentive)
    spread = mbs.solve_spread(cf, curve, 100.0)

    ladder = mbs.price_ladder(pool, curve, 2.0, spread, mortgage_rate,
                              np.array([-300, -200, -100, 0]), dial)
    fixed = mbs.price_ladder(pool, curve, 2.0, spread, mortgage_rate,
                             np.array([-300, -200, -100, 0]), None)
    assert np.all(ladder[:-1] < fixed[:-1]), "a rally must be worth less with live prepayment"
    print(f"ok  at -300bp the option costs {fixed[0]-ladder[0]:.2f} points of price")


# --------------------------------------------------------------------------
# Hull-White
# --------------------------------------------------------------------------

def test_hull_white_reproduces_the_initial_curve():
    """The decisive check on the simulation.

    Under the risk-neutral measure the expected discounted value of $1 paid at
    T must equal today's discount factor P(0,T). If the drift fit were wrong,
    the simulated economy would price today's Treasuries wrong and every OAS
    off it would be meaningless.
    """
    curve, _ = curves.default_curve()
    model = oas.HullWhite(curve)
    months, paths = 120, 20_000
    short = model.simulate(months, paths, seed=7)

    simulated = np.mean(np.exp(-np.cumsum(short / 12.0, axis=1)), axis=0)
    actual = curve.discount((np.arange(months) + 1) / 12.0)
    error = np.abs(simulated / actual - 1.0)

    # Monte Carlo error at 20k paths is a few basis points of price.
    assert error.max() < 3e-3, f"worst discount-factor error {error.max():.2e}"
    print(f"ok  {paths:,} simulated paths reprice the zero curve to "
          f"{error.max()*1e4:.1f}bp of price at worst")


def test_hull_white_zero_rates_are_affine_and_finite():
    curve, _ = curves.default_curve()
    model = oas.HullWhite(curve)
    short = model.simulate(60, 500, seed=3)
    times = (np.arange(60) + 1) / 12.0
    rates = model.zero_rate(times[None, :], 10.0, short)

    assert np.all(np.isfinite(rates))
    # At t -> 0 with the short rate at its fitted mean, the model's 10-year
    # rate must agree with the curve it was fitted to.
    today = float(model.zero_rate(1e-6, 10.0, np.array([model.alpha(1e-6)]))[0])
    expected = -np.log(curve.discount(10.0)) / 10.0
    assert abs(today - expected) < 1e-6, f"{today:.6f} vs {expected:.6f}"
    print(f"ok  the model's 10-year rate starts at the curve's {expected:.3%}")


def test_oas_is_below_the_static_spread():
    """Averaging over paths prices the option, so less spread is left over."""
    result = oas.run(paths=400, psa=2.0, target_price=100.0, seed=11)
    assert result["oas_bp"] < result["static_spread_bp"]
    assert 0.0 < result["option_cost_bp"] < 200.0
    print(f"ok  OAS {result['oas_bp']:.1f}bp sits {result['option_cost_bp']:.1f}bp "
          f"below the {result['static_spread_bp']:.1f}bp static spread")


def test_more_rate_volatility_costs_the_holder_more():
    """The option the borrower holds is worth more when rates move more."""
    quiet = oas.run(paths=400, psa=2.0, target_price=100.0, seed=11, sigma=0.005)
    loud = oas.run(paths=400, psa=2.0, target_price=100.0, seed=11, sigma=0.015)
    assert loud["option_cost_bp"] > quiet["option_cost_bp"]
    print(f"ok  option cost rises {quiet['option_cost_bp']:.1f}bp -> "
          f"{loud['option_cost_bp']:.1f}bp as vol goes 50bp -> 150bp")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
    print(f"\n{len(tests)} checks passed")
