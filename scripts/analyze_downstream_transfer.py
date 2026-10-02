#!/usr/bin/env python3
"""Analyze confirmatory downstream derived-code transfer after a passing gate."""
import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_model.progress import configure_logging, progress
from src.data.downstream_transfer import SCHEMA
from src.data.io import read_jsonl, sha256_file
from src.analysis.downstream_transfer import (
    current_answer_stability, history_contrasts, summarize_histories, validate_scores,
)
from src.experiments.downstream_transfer import (
    frozen_metadata, validate_dataset, validate_gate, validate_token_audit, write_json_create,
)
from src.utils import load_config

logger = logging.getLogger(__name__)
STABILITY_METRICS = (
    "baseline_accuracy", "edited_accuracy", "baseline_correct_logprob",
    "edited_correct_logprob", "baseline_margin", "edited_margin",
    "delta_correct_logprob", "delta_margin",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/downstream_transfer_v1.yaml")
    parser.add_argument("--values-json", default="configs/downstream_transfer_values.json")
    parser.add_argument("--codes-json", default="configs/downstream_transfer_codes.json")
    parser.add_argument("--seed", type=int, default=20261033)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args(argv)
    configure_logging()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty output directory")
    if args.bootstrap_draws < 1:
        raise ValueError("bootstrap draws must be positive")
    config = load_config(args.config)
    metadata = frozen_metadata(config, args.config, args.values_json, args.codes_json)
    gate_doc = json.loads(Path(args.gate).read_text())
    logger.info("Validating frozen gate and confirmatory lineage before computing effects")
    gate, _ = validate_gate(args.gate, metadata, gate_doc["token_audit_path"])
    validate_token_audit(gate["token_audit_path"], config, metadata,
                         json.loads(Path(args.codes_json).read_text()))
    rows, dsprov = validate_dataset(args.dataset, metadata)
    scores = read_jsonl(args.scores)
    side = json.loads(Path(args.scores + ".provenance.json").read_text())
    if rows[0]["stage"] != "confirmatory" or side.get("protocol") != SCHEMA or side.get("stage") != "confirmatory":
        raise ValueError("confirmatory dataset and scores are required")
    expected = {
        **metadata, "dataset_sha256": sha256_file(args.dataset),
        "gate_sha256": sha256_file(args.gate), "token_audit_sha256": gate["token_audit_sha256"],
    }
    for key, value in expected.items():
        if side.get(key) != value or dsprov.get(key) != value:
            raise ValueError(f"confirmatory frozen artifact mismatch: {key}")
    if side.get("causal_effects_computed") is not True:
        raise ValueError("confirmatory score provenance has wrong scoring mode")
    validate_scores(rows, scores, "confirmatory")
    logger.info("Validated %d records; aggregating complete histories", len(scores))
    histories = history_contrasts(scores)
    stability = current_answer_stability(rows, scores)
    buckets = defaultdict(lambda: defaultdict(list))
    for row in stability:
        for key in STABILITY_METRICS:
            buckets[row["history_id"]][key].append(row[key])
    stability_histories = [
        {"history_id": hid, **{key: sum(values) / len(values) for key, values in metrics.items()}}
        for hid, metrics in sorted(buckets.items())
    ]
    tasks = [("estimands", key, [row[key] for row in histories])
             for key in ("R_stale_derived", "R_live_derived")]
    tasks += [("current_answer_stability", key, [row[key] for row in stability_histories])
              for key in STABILITY_METRICS]
    summary = {
        "protocol": SCHEMA, "primary": "R_stale_derived", "bootstrap_unit": "history_id",
        "n_histories": len(histories), "seed": args.seed, "bootstrap_draws": args.bootstrap_draws,
        "estimands": {}, "current_answer_stability": {},
        "scores_sha256": sha256_file(args.scores), "dataset_sha256": sha256_file(args.dataset),
        "gate_sha256": sha256_file(args.gate), "analysis_script_sha256": sha256_file(__file__),
    }
    logger.info("Computing bootstrap and sign-flip summaries: %d histories, %d draws",
                len(histories), args.bootstrap_draws)
    for section, key, values in progress(tasks, desc="History statistics", unit="metric"):
        summary[section][key] = summarize_histories(values, args.seed, args.bootstrap_draws)
    output.mkdir(parents=True, exist_ok=True)
    for name, document in (
        ("downstream_transfer_summary.json", summary),
        ("downstream_transfer_per_history.json", histories),
        ("downstream_transfer_answer_stability.json", stability),
        ("downstream_transfer_stability_per_history.json", stability_histories),
    ):
        write_json_create(output / name, document)
    logger.info("Wrote analysis for %d histories to %s", len(histories), output)
    return summary


if __name__ == "__main__":
    main()
