#!/usr/bin/env python3
"""Export compact per-history estimates used for manuscript plots and contrasts.

This reads the sealed analysis JSONs in outputs/ and writes small CSVs that can
be checked in or archived with a manuscript release. It does not score models.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004"
FOLLOWUPS = ROOT / "outputs/followups"
OUT = Path(__file__).resolve().parent

CONDITIONS = (
    "superseded", "other_attribute", "entity_mention",
    "early_unassigned", "late_unassigned",
)
ORDERS = ("aligned", "reversed")


def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_exp1():
    rows = []
    for model, directory in (("Qwen3-8B", "qwen3_8b"), ("Gemma 3 4B", "gemma3_4b")):
        analysis_path = V2 / directory / "confirmatory_analysis.json"
        dataset_path = V2 / directory / "confirmatory.jsonl"
        estimates = json.loads(analysis_path.read_text())["history_rows"]
        histories = {}
        for line in dataset_path.open(encoding="utf-8"):
            member = json.loads(line)
            histories.setdefault(member["history_id"], member)
        for estimate in estimates:
            h = histories[estimate["history_id"]]
            row = {
                "model": model,
                "history_id": estimate["history_id"],
                "attribute": h["attribute"],
                "orientation": h["orientation"],
            }
            for condition in CONDITIONS:
                for order in ORDERS:
                    row[f"R_{condition}_{order}"] = estimate[f"R_{condition}_{order}"]
                row[f"R_{condition}_aligned_minus_reversed"] = estimate[
                    f"R_{condition}_aligned_minus_reversed"
                ]
            for control in ("other_attribute", "entity_mention", "early_unassigned", "late_unassigned"):
                row[f"R_superseded_minus_{control}"] = estimate[f"R_superseded_minus_R_{control}"]
                for order in ORDERS:
                    row[f"R_superseded_minus_{control}_{order}"] = estimate[
                        f"R_superseded_minus_R_{control}_{order}"
                    ]
            rows.append(row)
    fields = list(rows[0])
    write_rows(OUT / "exp1_history_estimates.csv", rows, fields)


def export_exp2(source="{prefix}_marker_confirmatory_analysis.json", name="exp2_marker_history_estimates.csv"):
    rows = []
    for model, prefix in (("Qwen3-8B", "qwen"), ("Gemma 3 4B", "gemma")):
        analysis_path = FOLLOWUPS / source.format(prefix=prefix)
        analysis = json.loads(analysis_path.read_text())
        for estimate in analysis["history_rows"]:
            row = {"model": model, "history_id": estimate["history_id"]}
            for construction in ("superseded", "entity_mention"):
                for marker in (0, 1):
                    for order in ("all", "aligned", "reversed"):
                        row[f"R_{construction}_marker{marker}_{order}"] = estimate[
                            f"{construction}_m{marker}_{order}_R"
                        ]
            for order in ("all", "aligned", "reversed"):
                for construction in ("superseded", "entity_mention"):
                    row[f"marker_effect_{construction}_{order}"] = estimate[
                        f"{construction}_marker_effect_{order}"
                    ]
                row[f"marker_interaction_{order}"] = estimate[
                    f"marker_by_construction_interaction_{order}"
                ]
                for marker in (0, 1):
                    row[f"entity_minus_superseded_marker{marker}_{order}"] = (
                        estimate[f"entity_mention_m{marker}_{order}_R"]
                        - estimate[f"superseded_m{marker}_{order}_R"]
                    )
            rows.append(row)
    write_rows(OUT / name, rows, list(rows[0]))


def export_distance():
    """Name-value distance follow-up (docs/distance_v1.md): per-history R by construction/distance."""
    rows = []
    for model, prefix in (("Qwen3-8B", "qwen"), ("Gemma 3 4B", "gemma")):
        analysis = json.loads((FOLLOWUPS / "distance_v1" / f"{prefix}_distance_analysis.json").read_text())
        for estimate in analysis["history_rows"]:
            row = {"model": model, "history_id": estimate["history_id"]}
            for key, value in estimate.items():
                if key != "history_id" and "component" not in key:
                    row[key] = value
            rows.append(row)
    write_rows(OUT / "distance_history_estimates.csv", rows, list(rows[0]))


def export_chain_scores():
    """Update chains at 96 histories (docs/chain_scores_v1.md): per-history R by depth."""
    rows = []
    for model, prefix in (("Qwen3-8B", "qwen"), ("Gemma 3 4B", "gemma")):
        for depth in (3, 4, 5):
            path = FOLLOWUPS / "chain_scores_v1" / f"confirmatory_d{depth}_{prefix}_analysis.json"
            for estimate in json.loads(path.read_text())["history_rows"]:
                rows.append({"model": model, "depth": depth, "history_id": estimate["history_id"], "R": estimate["R"]})
    write_rows(OUT / "chain_history_estimates.csv", rows, list(rows[0]))


def export_update():
    """Updated other attribute (docs/update_control_v1.md): per-history R and contrasts."""
    rows = []
    for model, prefix in (("Qwen3-8B", "qwen"), ("Gemma 3 4B", "gemma")):
        path = FOLLOWUPS / "update_v1" / f"update_confirmatory_{prefix}_analysis.json"
        for estimate in json.loads(path.read_text())["history_rows"]:
            rows.append({"model": model, **estimate})
    write_rows(OUT / "update_history_estimates.csv", rows, list(rows[0]))


if __name__ == "__main__":
    export_chain_scores()
    export_update()
    export_exp1()
    export_exp2()
    export_exp2("marker96/{prefix}_marker_analysis.json", "exp2_marker96_history_estimates.csv")
    export_distance()
