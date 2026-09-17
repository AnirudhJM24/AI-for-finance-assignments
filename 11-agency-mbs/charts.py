"""Chart styling shared by every figure in this assignment.

One palette, one set of mark rules, applied identically everywhere so the
three figures read as one system. The categorical slots are validated for
colour-vision deficiency at every pair, which is why there are three of them
and why the scenarios always take them in the same order: 100 PSA is blue in
every chart, 300 PSA is aqua in every chart. Colour follows the scenario, not
its rank on the page.

Aqua sits below 3:1 contrast on this surface, so every series carries a direct
label at the end of its line as well as a legend entry. Identity is never
colour alone.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e0"

# Categorical slots 1-3, in fixed order. Never cycled, never reassigned.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
REFERENCE = "#8a8880"       # non-categorical: baselines and comparison lines

LINE_WIDTH = 2.0


def setup() -> None:
    """Recessive axes, generous type, no chartjunk."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "savefig.dpi": 200,
        "font.size": 10.5, "font.family": "DejaVu Sans",
        "axes.edgecolor": GRID, "axes.linewidth": 1.0,
        "axes.labelcolor": INK_SOFT, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 30,
        "axes.labelsize": 10, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SOFT, "ytick.labelcolor": INK_SOFT,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "legend.labelcolor": INK_SOFT,
        "legend.fontsize": 10, "legend.handlelength": 1.6,
        "lines.linewidth": LINE_WIDTH, "lines.solid_capstyle": "round",
    })


def tidy(ax, subtitle: str | None = None) -> None:
    """Drop the box, keep only a horizontal grid, add an optional subtitle."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes,
                fontsize=10, color=INK_SOFT, va="bottom", ha="left")


def end_label(ax, x, y, text, dy=0.0, side="right", color=INK_SOFT) -> None:
    """Label a line at its end. Text stays in ink; the line beside it carries colour."""
    dx, align = (7, "left") if side == "right" else (-7, "right")
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", ha=align, fontsize=9.5, color=color)


def label_at(ax, x, y, x_at, text, **kw) -> None:
    """Direct-label a series where its neighbours are furthest away.

    The end of a line is the conventional place for its label, but these
    curves all converge - balances to zero, prices to each other - so the
    label goes where the series are actually separated instead.
    """
    import numpy as np
    k = int(np.searchsorted(np.asarray(x), x_at))
    k = min(max(k, 0), len(x) - 1)
    end_label(ax, x[k], y[k], text, **kw)
