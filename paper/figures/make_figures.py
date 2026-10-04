#!/usr/bin/env python3
"""Draw the paper's figures from the estimates in the sealed analysis reports.

Every number below is copied from a protocol record, not recomputed:
  - Experiment 1: docs/cross_model_relational_v2.md, "Complete results for the paper writer"
    (condition-level R and superseded-minus-control contrasts).
  - Experiment 2: docs/marker_construction_confirmation.md, "Marker effects and interaction".
Each entry is (mean, lower 95% bound, upper 95% bound) in nats.

Run from this directory:
    uv run --no-project --with matplotlib python make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
MODELS = ["Qwen3-8B", "Gemma 3 4B"]
COLORS = {"Qwen3-8B": "#1f5fa8", "Gemma 3 4B": "#c0502b"}
MARKERS = {"Qwen3-8B": "o", "Gemma 3 4B": "s"}
OFFSETS = {"Qwen3-8B": -0.12, "Gemma 3 4B": 0.12}


def flip(est):
    """Sign-reverse a contrast and its percentile interval (A - B -> B - A)."""
    mean, lo, hi = est
    return (-mean, -hi, -lo)


# Experiment 1, R_superseded - R_control, by order group.
SUPERSEDED_MINUS = {
    "Qwen3-8B": {
        ("Other attribute", "Overall"): (0.314, 0.147, 0.482),
        ("Entity mention", "Overall"): (-3.504, -3.806, -3.198),
        ("Early unassigned", "Aligned"): (-1.108, -1.281, -0.925),
        ("Early unassigned", "Reversed"): (5.130, 4.920, 5.348),
        ("Late unassigned", "Aligned"): (-1.043, -1.281, -0.810),
        ("Late unassigned", "Reversed"): (4.988, 4.724, 5.266),
    },
    "Gemma 3 4B": {
        ("Other attribute", "Overall"): (0.004, -0.104, 0.118),
        ("Entity mention", "Overall"): (-2.589, -2.829, -2.361),
        ("Early unassigned", "Aligned"): (-0.895, -1.084, -0.719),
        ("Early unassigned", "Reversed"): (4.044, 3.810, 4.292),
        ("Late unassigned", "Aligned"): (-1.741, -1.968, -1.505),
        ("Late unassigned", "Reversed"): (4.866, 4.592, 5.155),
    },
}

# Experiment 1, early-unassigned R by order group.
EARLY_UNASSIGNED_R = {
    "Qwen3-8B": {"Aligned": (3.252, 3.046, 3.455), "Reversed": (-3.327, -3.518, -3.150)},
    "Gemma 3 4B": {"Aligned": (2.598, 2.395, 2.798), "Reversed": (-2.372, -2.567, -2.178)},
}

# Experiment 2, marker effect R_present - R_absent, all order cells.
MARKER_EFFECT = {
    "Qwen3-8B": {"Superseded": (-0.750, -0.917, -0.590), "Entity mention": (-2.087, -2.410, -1.772)},
    "Gemma 3 4B": {"Superseded": (-1.085, -1.308, -0.864), "Entity mention": (-0.960, -1.183, -0.727)},
}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def errorbar(ax, y, est, model):
    mean, lo, hi = est
    ax.errorbar(
        mean, y, xerr=[[mean - lo], [hi - mean]], fmt=MARKERS[model], color=COLORS[model],
        ms=4, capsize=2, lw=1, label=model,
    )


def legend_above(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))  # one entry per model
    ax.legend(unique.values(), unique.keys(), frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))


def decomposition_figure():
    rows = [
        ("Relation status\nsuperseded − other attribute",
         {m: SUPERSEDED_MINUS[m][("Other attribute", "Overall")] for m in MODELS}),
        ("Construction\nentity mention − superseded",
         {m: flip(SUPERSEDED_MINUS[m][("Entity mention", "Overall")]) for m in MODELS}),
        ("Position\nearly-unassigned $R$, aligned", {m: EARLY_UNASSIGNED_R[m]["Aligned"] for m in MODELS}),
        ("Position\nearly-unassigned $R$, reversed", {m: EARLY_UNASSIGNED_R[m]["Reversed"] for m in MODELS}),
        ("Temporal marker\nsuperseded", {m: MARKER_EFFECT[m]["Superseded"] for m in MODELS}),
        ("Temporal marker\nentity mention", {m: MARKER_EFFECT[m]["Entity mention"] for m in MODELS}),
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (_, ests) in enumerate(rows):
        for model in MODELS:
            errorbar(ax, i + OFFSETS[model], ests[model], model)
    for boundary in (0.5, 1.5):
        ax.axhline(boundary, color="0.85", lw=0.6)
    ax.axhline(3.5, color="0.4", lw=0.8)  # Experiment 1 above, Experiment 2 below
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("$R$ or difference in $R$ (nats)")
    for y, text in ((-0.45, "Exp. 1, n = 96 histories"), (3.55, "Exp. 2, n = 24 histories")):
        ax.text(1.0, y, text, transform=ax.get_yaxis_transform(), ha="right", va="top",
                fontsize=7, color="0.4")
    legend_above(ax)
    fig.tight_layout()
    fig.savefig(OUT / "decomposition.pdf")
    plt.close(fig)


def order_figure():
    rows = [
        ("Early unassigned", "Aligned"),
        ("Early unassigned", "Reversed"),
        ("Late unassigned", "Aligned"),
        ("Late unassigned", "Reversed"),
    ]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    for model in MODELS:
        for i, row in enumerate(rows):
            errorbar(ax, i + OFFSETS[model], SUPERSEDED_MINUS[model][row], model)
    ax.axvline(0, color="0.5", lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)), [f"{c}, {o.lower()}" for c, o in rows])
    ax.invert_yaxis()
    ax.set_xlabel(r"$R_{\mathrm{superseded}} - R_{\mathrm{unassigned}}$ (nats)")
    legend_above(ax)
    fig.tight_layout()
    fig.savefig(OUT / "order_stratified_contrasts.pdf")
    plt.close(fig)


if __name__ == "__main__":
    decomposition_figure()
    order_figure()
