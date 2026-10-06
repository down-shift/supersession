#!/usr/bin/env python3
"""Experiment 1 summary for every screened model, from the sealed confirmation analyses.

Writes models_estimates.json and prints the LaTeX rows of Table tab:models (screened models) and of
Table tab:f-excluded (models that failed the screen, scored descriptively; docs/excluded_models_v1.md).
Models whose confirmation has not finished are skipped. It does not score models.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004"
P2B = ROOT / "outputs/p2b"
OUT = Path(__file__).resolve().parent

MODELS = (
    ("Qwen3-8B", V2 / "qwen3_8b"), ("Gemma 3 4B", V2 / "gemma3_4b"),
    ("Falcon3-7B", P2B / "falcon3_7b"), ("Granite-3.1-8B", P2B / "granite31_8b"),
    ("Mistral-7B", P2B / "mistral7b"), ("Gemma 3 12B", P2B / "gemma3_12b"),
    ("Qwen3-14B", P2B / "qwen3_14b"),
)
EXCLUDED = (
    ("Phi-4-mini", ROOT / "outputs/p2/phi4_mini"), ("Qwen2.5-7B", P2B / "qwen25_7b_bf16"),
)
COLUMNS = (
    ("superseded_minus_other", "R_superseded_minus_R_other_attribute", 1),
    ("entity_minus_superseded", "R_superseded_minus_R_entity_mention", -1),
    ("early_unassigned_aligned", "R_early_unassigned_aligned", 1),
    ("early_unassigned_reversed", "R_early_unassigned_reversed", 1),
)


def paired_gap_minus_relation(history_rows):
    """(entity mention - superseded) - (superseded - other attribute), paired within history, with the
    frozen history bootstrap (src.cross_model.robustness_analysis.summary: 2,000 draws, seed 73021)."""
    sys.path.insert(0, str(ROOT))
    from src.cross_model.robustness_analysis import summary
    s = summary([(r["R_entity_mention"] - r["R_superseded"]) - (r["R_superseded"] - r["R_other_attribute"])
                 for r in history_rows])
    return {"mean": s["mean"], "ci95": list(s["ci95_cluster_bootstrap"]), "n_histories": s["n_histories"]}


def estimate(result, sign):
    lo, hi = result["ci95_cluster_bootstrap"]
    lo, hi = sorted((sign * lo, sign * hi))
    return {"mean": sign * result["mean"], "ci95": [lo, hi], "n_histories": result["n_histories"]}


def fmt(x):
    return f"{x:.3f}".replace("-", "−")


def rows(models, descriptive):
    out, lines = {}, []
    for name, directory in models:
        path = directory / "confirmatory_analysis.json"
        if not path.exists():
            continue
        analysis = json.loads(path.read_text())
        if bool(analysis.get("descriptive_only")) != descriptive:
            raise ValueError(f"{name}: descriptive_only={analysis.get('descriptive_only')} in the wrong table")
        results = analysis["sequence_mass"]["results"]
        out[name] = {key: estimate(results[source], sign) for key, source, sign in COLUMNS}
        out[name]["gap_minus_relation"] = paired_gap_minus_relation(analysis["history_rows"])
        cells = [f"{fmt(e['mean'])} [{fmt(e['ci95'][0])}, {fmt(e['ci95'][1])}]" for e in out[name].values()]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    return out, lines


def main():
    screened, screened_lines = rows(MODELS, False)
    excluded, excluded_lines = rows(EXCLUDED, True)
    (OUT / "models_estimates.json").write_text(json.dumps({"screened": screened, "excluded_descriptive": excluded},
                                                          indent=2) + "\n")
    print("%% tab:models"); print("\n".join(screened_lines))
    print("%% tab:f-excluded"); print("\n".join(excluded_lines))


if __name__ == "__main__":
    main()
