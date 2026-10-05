#!/usr/bin/env python3
"""Figure 1 of the paper: what drives query relevance, computed from the sealed analyses.

Reads only saved artifacts (no model is loaded, nothing under outputs/ is modified):
  - Experiment 1: extended_analysis_v6/{qwen3_8b,gemma3_4b}/per_history.csv, produced by
    scripts/analyze_relational_confirmation_v2.py from the sealed confirmation scores;
  - Experiment 2: outputs/followups/marker96/{qwen,gemma}_marker_analysis.json (96 histories; the
    earlier 24-history confirmation is reported in Appendix D).
Experiment 1 rows are history-level differences summarized with the frozen history bootstrap
(src.cross_model.robustness_analysis.summary: 2,000 draws, seed 73021). Experiment 2 rows are
the sealed marker effects as saved.

Writes paper/figures/decomposition.pdf and paper/figures/decomposition_estimates.json (the
plotted numbers plus SHA-256 hashes of every input).

Run from the repository root:
    PYTHONPATH=. uv run --extra dev python scripts/fig_decomposition.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP1 = ROOT / ("outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004/"
               "extended_analysis_v6")
EXP2 = ROOT / "outputs/followups"
FIGURES = ROOT / "paper/figures"
MODELS = {"Qwen3-8B": ("qwen3_8b", "qwen"), "Gemma 3 4B": ("gemma3_4b", "gemma")}

# Experiment 1 rows: (group, label, per-history value as a function of one per_history.csv row).
EXP1_ROWS = [
    ("Relation status", "superseded $-$ other attribute",
     lambda r: r["R_superseded"] - r["R_other_attribute"]),
    ("Construction", "entity mention $-$ superseded",
     lambda r: r["R_entity_mention"] - r["R_superseded"]),
    ("Position (aligned $-$ reversed)", "early unassigned",
     lambda r: r["R_early_unassigned_aligned_minus_reversed"]),
    ("Position (aligned $-$ reversed)", "superseded",
     lambda r: r["R_superseded_aligned_minus_reversed"]),
    ("Position (aligned $-$ reversed)", "entity mention",
     lambda r: r["R_entity_mention_aligned_minus_reversed"]),
]
# Experiment 2 rows: (group, label, key in the sealed marker analysis summary).
EXP2_ROWS = [
    ("Temporal marker (with $-$ without)", "superseded", "superseded_marker_effect_all"),
    ("Temporal marker (with $-$ without)", "entity mention", "entity_mention_marker_effect_all"),
]


def read_history_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        return [{k: float(v) for k, v in row.items() if k != "history_id"} for row in csv.DictReader(f)]


def exp1_estimates(history_rows: list[dict[str, float]], summary) -> list[dict]:
    """Summarize each Experiment 1 row over histories with the given bootstrap summary function."""
    out = []
    for group, label, fn in EXP1_ROWS:
        s = summary([fn(r) for r in history_rows])
        lo, hi = s["ci95_cluster_bootstrap"]
        out.append({"group": group, "label": label, "experiment": 1, "n_histories": s["n_histories"],
                    "mean": s["mean"], "ci95": [lo, hi]})
    return out


def exp2_estimates(analysis: dict) -> list[dict]:
    out = []
    for group, label, key in EXP2_ROWS:
        s = analysis["summary"][key]
        out.append({"group": group, "label": label, "experiment": 2, "n_histories": s["n_histories"],
                    "mean": s["mean"], "ci95": list(s["ci95_history_bootstrap"])})
    return out


def compute() -> dict:
    from src.cross_model.robustness_analysis import summary
    from src.data.io import sha256_file

    result = {"bootstrap": "history; 2,000 draws; seed 73021", "inputs": {}, "models": {}}
    for model, (exp1_slug, exp2_slug) in MODELS.items():
        per_history = EXP1 / exp1_slug / "per_history.csv"
        marker = EXP2 / "marker96" / f"{exp2_slug}_marker_analysis.json"
        analysis = json.loads(marker.read_text())
        result["inputs"][str(per_history.relative_to(ROOT))] = sha256_file(per_history)
        result["inputs"][str(marker.relative_to(ROOT))] = sha256_file(marker)
        history_rows = read_history_rows(per_history)
        result["models"][model] = exp1_estimates(history_rows, summary) + exp2_estimates(analysis)
        result.setdefault("paired_order_effects", {})[model] = paired_order_effects(history_rows, summary)
    return result


def paired_order_effects(history_rows: list[dict[str, float]], summary) -> dict:
    """R aligned minus R reversed, paired within history, for the constructions in Appendix B."""
    out = {}
    for condition in ("superseded", "early_unassigned", "late_unassigned"):
        s = summary([r[f"R_{condition}_aligned_minus_reversed"] for r in history_rows])
        out[condition] = {"mean": s["mean"], "ci95": s["ci95_cluster_bootstrap"],
                          "fraction_positive": s["fraction_positive"], "n_histories": s["n_histories"]}
    return out


def plot(result: dict, path: Path) -> None:
    sys.path.insert(0, str(FIGURES))
    import matplotlib.pyplot as plt
    from style import MODELS as ORDER, OFFSETS, errorbar, legend_above, light_grid, zero_line

    rows = result["models"][ORDER[0]]
    groups: list[tuple[str, list[int]]] = []
    for i, r in enumerate(rows):
        if not groups or groups[-1][0] != r["group"]:
            groups.append((r["group"], []))
        groups[-1][1].append(i)

    header, row_h = 0.6, 1.0  # vertical space for a group title and for one row
    fig, ax = plt.subplots(figsize=(4.3, 3.05))
    y, ticks, labels, exp2_top = 0.0, [], [], None
    for k, (name, idx) in enumerate(groups):
        top = y
        if rows[idx[0]]["experiment"] == 2 and exp2_top is None:
            exp2_top = top
            ax.axhline(top, color="0.35", lw=0.6, zorder=2)  # Experiment 1 above, 2 below
        ax.text(0.01, top + 0.08, name, transform=ax.get_yaxis_transform(), ha="left", va="top",
                fontname="cmb10", fontsize=7.5)
        y += header
        for i in idx:
            centre = y + row_h / 2
            for model in ORDER:
                est = result["models"][model][i]
                errorbar(ax, centre + OFFSETS[model], (est["mean"], *est["ci95"]), model)
            ticks.append(centre)
            labels.append(rows[i]["label"])
            y += row_h
        if k % 2 == 1:
            ax.axhspan(top, y, color="0.965", zorder=0, lw=0)
    ax.set_ylim(y, 0)
    zero_line(ax)
    ax.set_yticks(ticks, labels)
    light_grid(ax)
    ax.set_xlabel("Difference in $R$ (nats)")
    trans = ax.get_yaxis_transform()
    n1 = rows[0]["n_histories"]
    n2 = next(r["n_histories"] for r in rows if r["experiment"] == 2)
    for yy, text in ((0.08, f"Exp. 1, $n={n1}$"), (exp2_top + 0.08, f"Exp. 2, $n={n2}$")):
        ax.text(0.99, yy, text, transform=trans, ha="right", va="top", fontsize=6.5,
                color="0.4", style="italic")
    legend_above(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    result = compute()
    (FIGURES / "decomposition_estimates.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, FIGURES / "decomposition.pdf")
    for model, rows in result["models"].items():
        for r in rows:
            lo, hi = r["ci95"]
            print(f"{model:11s} {r['group'][:30]:30s} {r['label'][:32]:32s} {r['mean']:7.3f} [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
