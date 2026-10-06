#!/usr/bin/env python3
"""Recreate main figures from the compact per-history result tables.

The CSV inputs are exported from sealed analysis JSONs by
paper/data/export_history_estimates.py. This script recomputes plotted means and
95% percentile history-bootstrap intervals (2,000 draws, seed 73021); it does
not require model outputs or a local style module.

Run from this directory with matplotlib available:
    python make_figures.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import style  # noqa: F401  shared fonts and sizes (paper/figures/style.py)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MODELS = ("Qwen3-8B", "Gemma 3 4B")
CONSTRUCTIONS = ("superseded", "other_attribute", "entity_mention",
                 "early_unassigned", "late_unassigned")
LABELS = ("Superseded", "Other attribute", "Entity mention",
          "Early unassigned", "Late unassigned")
ORDERS = ("aligned", "reversed")
ORDER_LABELS = {"aligned": "Aligned", "reversed": "Reversed"}
ORDER_COLORS = {"aligned": "#2878b5", "reversed": "#d97732"}
CONSTRUCTION_LABELS = {"superseded": "Superseded", "entity_mention": "Entity mention"}
CONSTRUCTION_COLORS = {"superseded": "#2878b5", "entity_mention": "#d97732"}


def read_csv(name):
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def estimate(rows, key):
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Missing or invalid history estimates for {key}")
    rng = np.random.default_rng(73021)
    indexes = rng.integers(0, len(values), size=(2000, len(values)))
    means = values[indexes].mean(axis=1)
    return float(values.mean()), *map(float, np.quantile(means, [0.025, 0.975]))


def draw_interval(ax, estimate, y, color, marker="o"):
    mean, lower, upper = estimate
    ax.errorbar(mean, y, xerr=[[mean - lower], [upper - mean]], fmt=marker,
                color=color, ecolor=color, markersize=3.7, capsize=2.2,
                linewidth=1.05, zorder=3)


def draw_vertical_interval(ax, estimate, x, color, marker="o"):
    mean, lower, upper = estimate
    ax.errorbar(x, mean, yerr=[[mean - lower], [upper - mean]], fmt=marker,
                color=color, ecolor=color, markersize=3.7, capsize=2.2,
                linewidth=1.05, zorder=3)


def construction_order_figure():
    data = read_csv("exp1_history_estimates.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), sharex=True, sharey=True)
    offsets = {"aligned": -0.10, "reversed": 0.10}
    for ax, model in zip(axes, MODELS):
        model_rows = [row for row in data if row["model"] == model]
        for yi, condition in enumerate(CONSTRUCTIONS):
            for order in ORDERS:
                rows = model_rows
                estimate_value = estimate(rows, f"R_{condition}_{order}")
                draw_interval(ax, estimate_value, yi + offsets[order], ORDER_COLORS[order])
        ax.axvline(0, color="0.35", linewidth=0.8)
        ax.set_title(model)
        ax.set_xlim(-4.2, 7.6)
        style.light_grid(ax)
    axes[0].set_yticks(range(len(LABELS)), LABELS)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Experiment 1 construction")
    fig.supxlabel("Query relevance $R$ (nats); 95% history-bootstrap intervals")
    handles = [plt.Line2D([0], [0], marker="o", color=ORDER_COLORS[o], linestyle="",
                          label=ORDER_LABELS[o]) for o in ORDERS]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=0.4)
    fig.savefig(HERE / "construction_by_order.pdf", bbox_inches="tight")
    plt.close(fig)


def marker_figure():
    """Experiment 2 by order group: the Gemma interaction has opposite signs in the two groups."""
    data = read_csv("exp2_marker96_history_estimates.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.6), sharey=True, sharex=True)
    constructions = ("superseded", "entity_mention")
    for col, model in enumerate(MODELS):
        model_rows = [row for row in data if row["model"] == model]
        for row_index, order in enumerate(ORDERS):
            ax = axes[row_index, col]
            for construction in constructions:
                estimates = []
                for marker in (0, 1):
                    est = estimate(model_rows, f"R_{construction}_marker{marker}_{order}")
                    estimates.append(est[0])
                    draw_vertical_interval(ax, est, marker, CONSTRUCTION_COLORS[construction])
                ax.plot([0, 1], estimates, color=CONSTRUCTION_COLORS[construction], linewidth=1.1, zorder=2)
            ax.set_title(f"{model}, {ORDER_LABELS[order].lower()} order")
            ax.set_xticks([0, 1], ["Absent", "Present"])
            ax.set_xlim(-0.25, 1.25)
            ax.set_ylim(0, 9.5)
            style.light_grid(ax)
        axes[0, 0].set_ylabel("Query relevance $R$ (nats)")
        axes[1, 0].set_ylabel("Query relevance $R$ (nats)")
    fig.supxlabel(r"Explicit marker $\it{Previously}$")
    handles = [plt.Line2D([0], [0], color=CONSTRUCTION_COLORS[c], marker="o", label=label)
               for c, label in (("superseded", "Superseded"), ("entity_mention", "Entity mention"))]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0.02, 0, 1, 0.95), w_pad=0.5, h_pad=0.8)
    fig.savefig(HERE / "marker_by_construction.pdf", bbox_inches="tight")
    plt.close(fig)


def order_figure():
    """Appendix figure: superseded-minus-unassigned contrasts by order group."""
    data = read_csv("exp1_history_estimates.csv")
    rows = [("early_unassigned", "aligned"), ("early_unassigned", "reversed"),
            ("late_unassigned", "aligned"), ("late_unassigned", "reversed")]
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    for model in MODELS:
        model_rows = [row for row in data if row["model"] == model]
        for i, (control, order) in enumerate(rows):
            style.errorbar(ax, i + style.OFFSETS[model],
                           estimate(model_rows, f"R_superseded_minus_{control}_{order}"), model)
    ax.axhspan(1.5, 3.5, color="0.965", zorder=0, lw=0)
    ax.set_yticks(range(len(rows)), [f"{c.replace('_', ' ')}, {o}" for c, o in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    style.zero_line(ax)
    style.light_grid(ax)
    ax.set_xlabel(r"$R_{\mathrm{superseded}} - R_{\mathrm{unassigned}}$ (nats)")
    style.legend_above(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / "order_stratified_contrasts.pdf")
    plt.close(fig)


if __name__ == "__main__":
    construction_order_figure()
    marker_figure()
    order_figure()
