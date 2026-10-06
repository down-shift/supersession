#!/usr/bin/env python3
"""Numbers added in reviewer round 1, computed only from saved score files and sealed analyses.

No model is loaded and nothing under outputs/ is modified. Writes the sealed JSON
paper/data/review_round1_estimates.json, from which every new number in the paper is copied:

  A2  paired aligned-minus-reversed marker-by-construction interaction (Experiment 2)
  A4  current-candidate rank accuracy of the 96-history update-chain score run, by depth
  A5  (b) + (a) for the excluded models, beside the recorded (b) - (a)
  B7  OLMo-2: score-based R, an error breakdown of generated answers, and a baseline-adjusted
      donor-minus-source generation contrast
  C1  absolute candidate log masses (source, donor, current) in Experiment 1
  C2  current-candidate rank accuracy for every model x experiment

Bootstraps: Experiment 1 quantities use src.cross_model.robustness_analysis.summary (the frozen
history bootstrap, 2,000 draws, seed 73021); follow-up quantities use
src.cross_model.followups._bootstrap (the same draws and seed).

    PYTHONPATH=. uv run --extra dev python paper/data/review_round1.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cross_model.followups import _bootstrap  # noqa: E402
from src.cross_model.protocol import sealed  # noqa: E402
from src.cross_model.robustness_analysis import summary  # noqa: E402
from src.data.io import sha256_file  # noqa: E402

OUT = Path(__file__).resolve().parent / "review_round1_estimates.json"
V2 = ROOT / "outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004"
FU = ROOT / "outputs/followups"
P2B = ROOT / "outputs/p2b"
CONDITIONS = ("live", "superseded", "other_attribute", "entity_mention", "early_unassigned", "late_unassigned")
EXP1_SCORES = {
    "Qwen3-8B": V2 / "qwen3_8b/confirmatory_scores.jsonl",
    "Gemma 3 4B": V2 / "gemma3_4b/confirmatory_scores.jsonl",
    "Falcon3-7B": P2B / "falcon3_7b/confirmatory_scores.jsonl",
    "Granite-3.1-8B": P2B / "granite31_8b/confirmatory_scores.jsonl",
    "Mistral-7B": P2B / "mistral7b/confirmatory_scores.jsonl",
    "Gemma 3 12B": P2B / "gemma3_12b/confirmatory_scores.jsonl",
    "Qwen3-14B": P2B / "qwen3_14b/confirmatory_scores.jsonl",
    "Phi-4-mini (excluded)": ROOT / "outputs/p2/phi4_mini/confirmatory_scores.jsonl",
    "Qwen2.5-7B (excluded)": P2B / "qwen25_7b_bf16/confirmatory_scores.jsonl",
    "OLMo-2-7B (excluded, generation run)": P2B / "olmo2_7b/confirmatory_generation_scores.jsonl",
}
INPUTS: dict[str, str] = {}


def read(path: Path) -> list[dict]:
    INPUTS[str(path.relative_to(ROOT))] = sha256_file(path)
    with path.open() as f:
        return [json.loads(line) for line in f]


def mass(row: dict) -> dict:
    m = row["semantic_log_mass"]
    return ast.literal_eval(m) if isinstance(m, str) else m


def est(stats: dict) -> dict:
    ci = stats.get("ci95_history_bootstrap", stats.get("ci95_cluster_bootstrap"))
    return {"mean": stats["mean"], "ci95": [float(ci[0]), float(ci[1])], "n_histories": stats["n_histories"]}


def accuracy(rows: list[dict], key: str = "condition") -> dict:
    """Strict rank-one accuracy of the current value over all scored prompts (baseline and edited)."""
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        if "semantic_rank" in r:  # Experiment 1 scorer: stored strict rank-one result
            ok = int(r["semantic_rank"] == 1 and r.get("semantic_accuracy", 1) == 1)
        else:  # follow-up scorer stores masses only: current strictly above every other candidate
            m = mass(r)
            ok = int(all(m[r["answer"]] > v for k, v in m.items() if k != r["answer"]))
        out[r[key]][0] += ok
        out[r[key]][1] += 1
    total = [sum(v[0] for v in out.values()), sum(v[1] for v in out.values())]
    worst = min(out, key=lambda k: out[k][0] / out[k][1])
    return {"correct": total[0], "total": total[1], "by_group": {k: {"correct": v[0], "total": v[1]} for k, v in out.items()},
            "lowest_group": worst}


# ---------------------------------------------------------------- A2
def marker_order_contrast() -> dict:
    out = {}
    for model, slug in (("Qwen3-8B", "qwen"), ("Gemma 3 4B", "gemma")):
        path = FU / f"marker96/{slug}_marker_analysis.json"
        INPUTS[str(path.relative_to(ROOT))] = sha256_file(path)
        h = json.loads(path.read_text())["history_rows"]
        out[model] = {
            "interaction_aligned_minus_reversed": est(_bootstrap(
                [r["marker_by_construction_interaction_aligned"] - r["marker_by_construction_interaction_reversed"]
                 for r in h])),
            "check_interaction_all": est(_bootstrap([r["marker_by_construction_interaction_all"] for r in h])),
        }
    return out


# ---------------------------------------------------------------- Mistral accuracy sensitivity
def correct_history_sensitivity(scores: list[dict]) -> dict:
    """Table 2 contrasts for Mistral-7B restricted to histories in which every scored prompt ranks the
    current candidate strictly first; the same restriction as Table B7 (reproduces its 91/95 histories)."""
    path = P2B / "mistral7b/confirmatory_analysis.json"
    INPUTS[str(path.relative_to(ROOT))] = sha256_file(path)
    ok = defaultdict(lambda: True)
    for r in scores:
        ok[r["history_id"]] &= r["semantic_rank"] == 1 and r.get("semantic_accuracy", 1) == 1
    h = [x for x in json.loads(path.read_text())["history_rows"] if ok[x["history_id"]]]
    cols = {"superseded_minus_other": lambda x: x["R_superseded"] - x["R_other_attribute"],
            "entity_minus_superseded": lambda x: x["R_entity_mention"] - x["R_superseded"],
            "early_unassigned_aligned": lambda x: x["R_early_unassigned_aligned"],
            "early_unassigned_reversed": lambda x: x["R_early_unassigned_reversed"],
            "gap_minus_relation": lambda x: (x["R_entity_mention"] - x["R_superseded"])
                                            - (x["R_superseded"] - x["R_other_attribute"])}
    out = {"n_histories": len(h)}
    out.update({k: est(summary([f(x) for x in h])) for k, f in cols.items()})
    return out


# ---------------------------------------------------------------- A5
def excluded_b_plus_a() -> dict:
    out = {}
    for model, path in (("Phi-4-mini", ROOT / "outputs/p2/phi4_mini/confirmatory_analysis.json"),
                        ("Qwen2.5-7B", P2B / "qwen25_7b_bf16/confirmatory_analysis.json")):
        INPUTS[str(path.relative_to(ROOT))] = sha256_file(path)
        h = json.loads(path.read_text())["history_rows"]
        b = [r["R_entity_mention"] - r["R_superseded"] for r in h]
        a = [r["R_superseded"] - r["R_other_attribute"] for r in h]
        out[model] = {"b_minus_a": est(summary([x - y for x, y in zip(b, a)])),
                      "b_plus_a": est(summary([x + y for x, y in zip(b, a)]))}
        out[model]["criterion_b_exceeds_abs_a"] = bool(out[model]["b_minus_a"]["ci95"][0] > 0
                                                       and out[model]["b_plus_a"]["ci95"][0] > 0)
    return out


# ---------------------------------------------------------------- C1
def absolute_masses(scores: dict[str, list[dict]]) -> dict:
    out = {}
    for model in ("Qwen3-8B", "Gemma 3 4B"):
        cells = defaultdict(list)
        for r in scores[model]:
            m = mass(r)
            side = "baseline" if r["pair_direction"] == 0 else "edited"
            for role, value in (("source", r["source_value"]), ("donor", r["replacement_value"]), ("current", r["answer"])):
                cells[(r["condition"], side, role)].append(m[value])
        out[model] = {f"{c}|{s}|{role}": {"mean": float(np.mean(v)), "q25": float(np.quantile(v, .25)),
                                          "median": float(np.median(v)), "q75": float(np.quantile(v, .75)), "n": len(v)}
                      for (c, s, role), v in sorted(cells.items())}
    return out


# ---------------------------------------------------------------- B7
def olmo(rows: list[dict]) -> dict:
    from src.cross_model.robustness_analysis import report
    dataset = read(V2 / "qwen3_8b/confirmatory.jsonl")
    if sha256_file(V2 / "qwen3_8b/confirmatory.jsonl")[:16] != "27c1e9b80e62a0cb":
        raise ValueError("not the sealed Experiment 1 confirmation dataset")
    by_id = {r["example_id"]: r for r in rows}
    scores = [by_id[r["example_id"]] for r in dataset]
    res = report(dataset, scores)["sequence_mass"]["results"]
    score_R = {c: est(res[f"R_{c}"]) for c in CONDITIONS}
    score_R.update({f"{c}_{g}": est(res[f"R_{c}_{g}"]) for c in ("early_unassigned", "entity_mention") for g in ("aligned", "reversed")})

    # error breakdown of generated answers
    cats = defaultdict(lambda: defaultdict(int))
    for r in rows:
        ans, side = r["parsed_answer"], ("baseline" if r["pair_direction"] == 0 else "edited")
        other_current = r["current_z"] if r["query"] == "x" else r["current_x"]
        if ans == r["answer"]:
            k = "correct"
        elif ans is None:
            k = "invalid_format"
        elif ans == r["source_value"]:
            k = "source_value"
        elif ans == r["replacement_value"]:
            k = "donor"
        elif ans == other_current:
            k = "other_entity_current"
        elif ans == r["semantic_values"]["initial_z" if r["edited_variable"] == "x" else "initial_x"]:
            k = "other_entity_earlier"
        else:
            k = "other_candidate"
        cats[f"{r['condition']}|{side}"][k] += 1
    errors = {k: dict(v) for k, v in sorted(cats.items())}

    # baseline-adjusted donor-minus-source generation contrast, symmetric over queries like R
    cell = defaultdict(lambda: defaultdict(list))
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r["pair_id"]][r["pair_direction"]] = r
    for pid, p in pairs.items():
        b, e = p[0], p[1]
        d = ((e["parsed_answer"] == e["replacement_value"]) - (b["parsed_answer"] == b["replacement_value"])) \
            - ((e["parsed_answer"] == e["source_value"]) - (b["parsed_answer"] == b["source_value"]))
        order = "aligned" if b["historical_entity_order"] == b["current_entity_order"] else "reversed"
        for g in ("all", order):
            cell[(b["history_id"], b["condition"], g)][(b["edited_variable"], b["query"])].append(d)
    gen = defaultdict(list)
    for (hid, c, g), v in cell.items():
        m = {k: float(np.mean(x)) for k, x in v.items()}
        gen[(c, g)].append(.5 * ((m[("x", "x")] - m[("x", "z")]) + (m[("z", "z")] - m[("z", "x")])))
    generation_contrast = {f"{c}|{g}": est(summary(v)) for (c, g), v in sorted(gen.items())}
    return {"score_R": score_R, "error_breakdown": errors, "generation_donor_minus_source": generation_contrast,
            "definition": "per history: 1/2[(D_xx - D_xz) + (D_zz - D_zx)], D = change in P(answer = donor) minus "
                          "change in P(answer = source), edited minus baseline, averaged over order cells"}


# ---------------------------------------------------------------- A4 / C2 follow-ups
def followup_accuracy(missing: list) -> dict:
    out = {}
    for model, slug in (("Qwen3-8B", "qwen"), ("Gemma 3 4B", "gemma")):
        files = {"Experiment 2 (marker)": FU / f"marker96/{slug}_marker_scores.jsonl",
                 "Distance": FU / f"distance_v1/{slug}_distance_scores.jsonl",
                 "Updated other attribute": FU / f"update_v1/update_confirmatory_{slug}_scores.jsonl"}
        files.update({f"Chains, depth {d} (96-history score run)": FU / f"chain_scores_v1/confirmatory_d{d}_{slug}_scores.jsonl"
                      for d in (3, 4, 5)})
        out[model] = {}
        for name, path in files.items():
            if path.exists():
                out[model][name] = accuracy(read(path))
            else:
                missing.append(str(path.relative_to(ROOT)))
    return out


def main() -> None:
    scores = {m: read(p) for m, p in EXP1_SCORES.items() if p.exists()}
    missing = [str(p.relative_to(ROOT)) for m, p in EXP1_SCORES.items() if not p.exists()]
    olmo_key = "OLMo-2-7B (excluded, generation run)"
    result = {
        "A2_marker_order_contrast": marker_order_contrast(),
        "A5_excluded_b_plus_a": excluded_b_plus_a(),
        "C1_absolute_log_masses": absolute_masses(scores),
        "C2_exp1_accuracy": {m: accuracy(s) for m, s in scores.items()},
        "C2_followup_accuracy": followup_accuracy(missing),
        "B7_olmo": olmo(scores[olmo_key]) if olmo_key in scores else None,
        "mistral_correct_histories": (correct_history_sensitivity(scores["Mistral-7B"])
                                      if "Mistral-7B" in scores else None),
        "missing_inputs": missing,
        "bootstrap": "history; 2,000 draws; seed 73021",
    }
    result["inputs_sha256"] = dict(sorted(INPUTS.items()))
    OUT.write_text(json.dumps(sealed(result), indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT), "missing:", missing)


if __name__ == "__main__":
    main()
