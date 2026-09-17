"""Price the pool under 100, 200, and 300 PSA, and draw the three figures.

    python run.py                       # snapshot curve, price to par at 200 PSA
    python run.py --refresh             # pull today's curve from FRED first
    python run.py --target-price 101.5  # solve the spread off a different price

The spread is not an input. It is solved once, so that the pool prices at
`--target-price` under the base scenario, and then held fixed everywhere else.
That is what makes the other two scenarios and all the shocks comparable: the
only thing moving between them is the prepayment assumption.
"""

from __future__ import annotations

import argparse

import numpy as np

import charts
import curves
import mbs

SCENARIOS = [(1.0, "100 PSA"), (2.0, "200 PSA"), (3.0, "300 PSA")]


def md_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    rule = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([line(header), rule] + [line(r) for r in rows])


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def chart_balance(pool, dial, path="balance.png") -> None:
    """Remaining balance under the three speeds - the contraction picture."""
    fig, ax = charts.plt.subplots(figsize=(8.2, 4.8))

    for (psa, label), colour in zip(SCENARIOS, charts.SERIES):
        cf = mbs.project(pool, psa=psa, dial=dial,
                         incentive=dial.base_incentive).held()
        years = cf.month / 12.0
        pct = 100.0 * cf.end_balance / pool.original_balance
        ax.plot(years, pct, color=colour, label=label)
        charts.label_at(ax, years, pct, 12.0, label, dy=6)

    ax.set_title("Faster prepayment retires the pool years earlier")
    charts.tidy(ax, "Remaining balance, % of original face, from settlement at age 12 months")
    ax.set_xlabel("Years from settlement")
    ax.set_ylabel("% of original balance")
    ax.set_xlim(0, 29)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", ncols=3)
    fig.tight_layout()
    fig.savefig(path)
    charts.plt.close(fig)


def chart_principal(pool, dial, path="principal.png") -> None:
    """Monthly principal, and the scheduled part of it.

    The gap between a scenario's line and the dashed reference is prepayment.
    At 300 PSA almost all of the early cash flow is borrowers leaving, not
    borrowers amortizing.
    """
    fig, ax = charts.plt.subplots(figsize=(8.2, 4.8))

    for (psa, label), colour in zip(SCENARIOS, charts.SERIES):
        cf = mbs.project(pool, psa=psa, dial=dial,
                         incentive=dial.base_incentive).held()
        years = cf.month / 12.0
        ax.plot(years, cf.principal / 1e6, color=colour, label=label)
        if psa == 3.0:
            ax.plot(years, cf.scheduled_principal / 1e6, color=charts.REFERENCE,
                    linewidth=1.4, linestyle=(0, (4, 3)),
                    label="Scheduled only, 300 PSA")

        # Label each curve at its peak: the three peaks are far apart, while
        # the tails converge and leave nowhere for a label to sit.
        peak = int(np.argmax(cf.principal))
        charts.end_label(ax, years[peak], cf.principal[peak] / 1e6, label, dy=12)

    ax.set_title("Prepayment pulls principal forward, then starves the tail")
    charts.tidy(ax, "Monthly principal paid to the investor, $ millions")
    ax.set_xlabel("Years from settlement")
    ax.set_ylabel("$ millions")
    ax.set_xlim(0, 29)
    ax.set_ylim(0, 1.6)     # headroom so the 300 PSA peak label clears the subtitle
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    charts.plt.close(fig)


def chart_shocks(pool, curve, dial, spread, mortgage_rate, path="price_shocks.png") -> None:
    """Price against parallel rate shocks - where the negative convexity shows."""
    shocks = np.arange(-300, 301, 10)
    fig, (left, right) = charts.plt.subplots(1, 2, figsize=(11.6, 4.9))

    for (psa, label), colour in zip(SCENARIOS, charts.SERIES):
        ladder = mbs.price_ladder(pool, curve, psa, spread, mortgage_rate, shocks, dial)
        left.plot(shocks, ladder, color=colour, label=label)
        charts.end_label(left, shocks[-1], ladder[-1], label)

    left.axvline(0, color=charts.GRID, linewidth=1.0, zorder=0)
    left.set_title("Price compresses in a rally")
    charts.tidy(left, "Price per 100 current face, refinancing-aware prepayment")
    left.set_xlabel("Parallel rate shock, bp")
    left.set_ylabel("Price")
    left.set_xlim(-310, 400)
    left.set_xticks(np.arange(-300, 301, 100))
    left.legend(loc="lower left", ncols=3)

    live = mbs.price_ladder(pool, curve, 2.0, spread, mortgage_rate, shocks, dial)
    static = mbs.price_ladder(pool, curve, 2.0, spread, mortgage_rate, shocks, None)
    right.plot(shocks, static, color=charts.REFERENCE, linewidth=1.6,
               linestyle=(0, (4, 3)), label="Prepayment held fixed at 200 PSA")
    right.plot(shocks, live, color=charts.SERIES[1], label="Prepayment responds to rates")
    right.fill_between(shocks, live, static, color=charts.SERIES[1], alpha=0.08, lw=0)
    charts.end_label(right, shocks[-1], static[-1], "Fixed speed", dy=6)
    charts.end_label(right, shocks[-1], live[-1], "Responsive speed", dy=-6)

    right.axvline(0, color=charts.GRID, linewidth=1.0, zorder=0)
    right.set_title("The shaded wedge is the prepayment option")
    charts.tidy(right, "200 PSA, price per 100 current face")
    right.set_xlabel("Parallel rate shock, bp")
    right.set_ylabel("Price")
    right.set_xlim(-310, 470)
    right.set_xticks(np.arange(-300, 301, 100))
    right.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(path)
    charts.plt.close(fig)


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="pull today's curve from FRED")
    ap.add_argument("--target-price", type=float, default=100.0,
                    help="price the base scenario to this, and solve for the spread")
    ap.add_argument("--base-psa", type=float, default=2.0,
                    help="scenario the spread is solved against")
    ap.add_argument("--bump-bp", type=float, default=50.0, help="shock size for the risk measures")
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args()

    if args.refresh:
        curves.refresh()

    curve, mortgage_rate = curves.default_curve()
    pool = mbs.Pool()
    dial = mbs.make_dial(pool, mortgage_rate)

    base = mbs.project(pool, psa=args.base_psa, dial=dial, incentive=dial.base_incentive)
    spread = mbs.solve_spread(base, curve, args.target_price)

    print(f"Pool          ${pool.original_balance/1e6:.0f}mm, {pool.mortgage_rate:.3%} "
          f"gross / {pool.pass_through:.3%} net, {pool.original_term}mo term, "
          f"age {pool.age}mo")
    print(f"Current face  ${base.current_face/1e6:.2f}mm "
          f"({100*base.current_face/pool.original_balance:.2f}% of original)")
    print(f"Market        30y mortgage {mortgage_rate:.3%}, 10y Treasury "
          f"{np.expm1(curve.zero(10.0)):.3%} zero")
    print(f"Incentive     {dial.base_incentive:+.2f}pp "
          f"({'in' if dial.base_incentive > 0 else 'out of'} the money)")
    print(f"Static spread {spread*1e4:.1f}bp, solved to price {args.target_price:.2f} "
          f"at {args.base_psa*100:.0f} PSA\n")

    rows, static_rows = [], []
    for psa, label in SCENARIOS:
        live = mbs.risk(pool, curve, psa, spread, mortgage_rate, dial, args.bump_bp)
        fixed = mbs.risk(pool, curve, psa, spread, mortgage_rate, None, args.bump_bp)
        rows.append([label, f"{live.price:.3f}", f"{live.wal:.2f}",
                     f"{live.effective_duration:.2f}", f"{live.effective_convexity:.0f}"])
        static_rows.append([
            label, f"{fixed.effective_duration:.2f}", f"{fixed.effective_convexity:+.0f}",
            f"{live.effective_duration:.2f}", f"{live.effective_convexity:+.0f}",
            f"{live.wal_down:.2f} / {live.wal:.2f} / {live.wal_up:.2f}",
        ])

    print(md_table(["Scenario", "Price", "WAL", "Effective duration", "Convexity"], rows))
    print()
    print(md_table(
        ["Scenario", "Fixed D", "Fixed C", "Responsive D", "Responsive C",
         f"WAL at -{args.bump_bp:.0f} / 0 / +{args.bump_bp:.0f} bp"],
        static_rows))

    if not args.no_charts:
        charts.setup()
        chart_balance(pool, dial)
        chart_principal(pool, dial)
        chart_shocks(pool, curve, dial, spread, mortgage_rate)
        print("\nwrote balance.png, principal.png, price_shocks.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
