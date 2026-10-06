#!/usr/bin/env python3
"""Numbers added in reviewer round 2, computed only from saved score files and sealed analyses.

B1: the updated-other-attribute follow-up (docs/update_control_v1.md) restricted to histories in which
every scored prompt ranks the current candidate strictly first (ties count as errors), the restriction
of Table B7 and of the Mistral-7B check in Section 5.9. Uses the unchanged analysis
(src.cross_model.update_control.analyze_update) and its history bootstrap (2,000 draws, seed 73021).

Writes the sealed JSON paper/data/review_round2_estimates.json.

    PYTHONPATH=. uv run --extra dev python paper/data/review_round2.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.cross_model.protocol import sealed  # noqa: E402
from src.cross_model.update_control import analyze_update  # noqa: E402
from src.data.io import read_jsonl, sha256_file  # noqa: E402

FU = ROOT / "outputs/followups/update_v1"
OUT = Path(__file__).resolve().parent / "review_round2_estimates.json"
KEYS = ("superseded_minus_other_updated_all", "other_updated_minus_other_static_all",
        "superseded_minus_other_static_all")


def strictly_first(score: dict, answer: str) -> bool:
    m = score["semantic_log_mass"]
    m = ast.literal_eval(m) if isinstance(m, str) else m
    return all(m[answer] > v for k, v in m.items() if k != answer)


def main() -> None:
    dataset_path = FU / "update_confirmatory.jsonl"
    rows = read_jsonl(dataset_path)
    inputs = {str(dataset_path.relative_to(ROOT)): sha256_file(dataset_path)}
    out = {}
    for model, slug in (("Qwen3-8B", "qwen"), ("Gemma 3 4B", "gemma")):
        path = FU / f"update_confirmatory_{slug}_scores.jsonl"
        inputs[str(path.relative_to(ROOT))] = sha256_file(path)
        scores = read_jsonl(path)
        by_id = {s["example_id"]: s for s in scores}
        ok = defaultdict(lambda: True)
        for r in rows:
            ok[r["history_id"]] &= strictly_first(by_id[r["example_id"]], r["answer"])
        keep = {h for h, v in ok.items() if v}
        full = analyze_update(rows, scores)["summary"]
        sub_rows = [r for r in rows if r["history_id"] in keep]
        sub = analyze_update(sub_rows, [by_id[r["example_id"]] for r in sub_rows])["summary"]
        out[model] = {"n_histories_all": len({r["history_id"] for r in rows}), "n_histories_correct": len(keep),
                      "all": {k: full[k] for k in KEYS}, "correct_histories": {k: sub[k] for k in KEYS}}
    OUT.write_text(json.dumps(sealed({"B1_update_correct_histories": out, "inputs_sha256": inputs,
                                      "bootstrap": "history; 2,000 draws; seed 73021"}), indent=1) + "\n")
    for model, v in out.items():
        print(model, v["n_histories_correct"], "of", v["n_histories_all"])
        for k in KEYS:
            a, c = v["all"][k], v["correct_histories"][k]
            print(f"  {k}: all {a['mean']:.3f} {a['ci95_history_bootstrap']}  correct {c['mean']:.3f} "
                  f"[{c['ci95_history_bootstrap'][0]:.3f}, {c['ci95_history_bootstrap'][1]:.3f}]")


if __name__ == "__main__":
    main()
