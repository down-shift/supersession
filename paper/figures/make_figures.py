#!/usr/bin/env python3
"""Draw Figure 2 (order-stratified contrasts) from the estimates in the protocol record.

Figure 1 is drawn from the sealed analyses by scripts/fig_decomposition.py.

Every number below is copied from the protocol record, not recomputed:
  - Experiment 1: docs/cross_model_relational_v2.md, "Complete results for the paper writer"
    (condition-level R and superseded-minus-control contrasts).
  - Experiment 2: docs/marker_construction_confirmation.md, "Marker effects and interaction".
Each entry is (mean, lower 95% bound, upper 95% bound) in nats.

Run from this directory:
    uv run --no-project --with matplotlib python make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from style import MODELS, OFFSETS, errorbar, legend_above, light_grid, zero_line

OUT = Path(__file__).resolve().parent

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


def order_figure():
    rows = [
        ("Early unassigned", "Aligned"),
        ("Early unassigned", "Reversed"),
        ("Late unassigned", "Aligned"),
        ("Late unassigned", "Reversed"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 1.75))
    for model in MODELS:
        for i, row in enumerate(rows):
            errorbar(ax, i + OFFSETS[model], SUPERSEDED_MINUS[model][row], model)
    ax.axhspan(1.5, 3.5, color="0.965", zorder=0, lw=0)
    ax.set_yticks(range(len(rows)), [f"{c.lower()}, {o.lower()}" for c, o in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    zero_line(ax)
    light_grid(ax)
    ax.set_xlabel(r"$R_{\mathrm{superseded}} - R_{\mathrm{unassigned}}$ (nats)")
    legend_above(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "order_stratified_contrasts.pdf")
    plt.close(fig)


if __name__ == "__main__":
    order_figure()
