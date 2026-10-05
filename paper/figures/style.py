"""Shared matplotlib style for the paper's forest plots (Computer Modern, Okabe-Ito colours)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ["Qwen3-8B", "Gemma 3 4B"]
COLORS = {"Qwen3-8B": "#0072B2", "Gemma 3 4B": "#D55E00"}  # Okabe-Ito, colorblind-safe
MARKERS = {"Qwen3-8B": "o", "Gemma 3 4B": "s"}
OFFSETS = {"Qwen3-8B": -0.14, "Gemma 3 4B": 0.14}

plt.rcParams.update({
    # Match the paper's Computer Modern text.
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 0,
    "pdf.fonttype": 42,
})


def errorbar(ax, y, est, model):
    """Plot one (mean, lower, upper) estimate as a point with a horizontal interval."""
    mean, lo, hi = est
    ax.errorbar(
        mean, y, xerr=[[mean - lo], [hi - mean]], fmt=MARKERS[model], color=COLORS[model],
        ms=3.2, mew=0, elinewidth=1.1, capsize=0, label=model, zorder=3,
    )


def light_grid(ax):
    ax.grid(axis="both", color="0.88", lw=0.45, ls=(0, (2, 2)), zorder=0)
    ax.set_axisbelow(True)


def zero_line(ax):
    ax.axvline(0, color="0.55", lw=0.6, zorder=1)


def legend_above(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # one entry per model
    ax.legend(unique.values(), unique.keys(), frameon=False, ncol=2, handletextpad=0.3,
              columnspacing=1.2, loc="lower center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.2)
