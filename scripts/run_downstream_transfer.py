#!/usr/bin/env python3
"""Resumable complete-sequence scoring; development/gate are competence-only."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm.contrib.logging import logging_redirect_tqdm
from src.cross_model.progress import configure_logging, progress
from src.cross_model.tokens import check_tokenizer
from src.data.downstream_transfer import SCHEMA, render, seal_artifact, verify_sealed_artifact
from src.data.io import read_jsonl, sha256_file
from src.data.progress import prepare_jsonl_progress, append_jsonl_record
from src.analysis.downstream_transfer import (
    score_protocol_record, evaluate_competence_gate, validate_scores,
)
from src.experiments.downstream_transfer import (
    frozen_metadata, score_codes, generate_unrestricted_code, validate_dataset, validate_gate,
    validate_token_audit, write_json_create,
)
from src.models.loader import load_model
from src.utils import load_config

logger = logging.getLogger(__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default="configs/downstream_transfer_v1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gate")
    parser.add_argument("--token-audit", required=True)
    parser.add_argument("--values-json", default="configs/downstream_transfer_values.json")
    parser.add_argument("--codes-json", default="configs/downstream_transfer_codes.json")
    args = parser.parse_args(argv)
    configure_logging()
    config = load_config(args.config)
    metadata = frozen_metadata(config, args.config, args.values_json, args.codes_json)
    logger.info("Validating dataset and frozen artifacts: %s", args.dataset)
    rows, dsprov = validate_dataset(args.dataset, metadata)
    stage = rows[0]["stage"]
    codes = json.loads(Path(args.codes_json).read_text())
    if (rows[0]["code_vocabulary"] != codes or rows[0]["value_vocabulary"] !=
            json.loads(Path(args.values_json).read_text())):
        raise ValueError("dataset vocabularies differ from frozen files")
    token_audit = validate_token_audit(args.token_audit, config, metadata, codes)
    if stage == "confirmatory":
        if not args.gate:
            raise ValueError("confirmatory scoring requires frozen --gate")
        gate, _ = validate_gate(args.gate, metadata, args.token_audit)
        if (dsprov.get("gate_sha256") != sha256_file(args.gate) or
                dsprov.get("token_audit_sha256") != gate["token_audit_sha256"]):
            raise ValueError("dataset is not bound to supplied gate/tokenizer audit")
        logger.info("Verified passing frozen competence gate")
    fingerprint = {**config, "downstream_transfer_v1": {
        **metadata, "stage": stage, "dataset_sha256": sha256_file(args.dataset),
        "token_audit_sha256": sha256_file(args.token_audit),
        "gate_sha256": sha256_file(args.gate) if args.gate else None,
    }}
    side = Path(args.output + ".provenance.json")
    gate_path = Path(args.output + ".gate.json")
    side_data = {
        "protocol": SCHEMA, "stage": stage, **metadata,
        "dataset_path": str(Path(args.dataset).resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "token_audit_sha256": sha256_file(args.token_audit),
        "causal_effects_computed": stage == "confirmatory",
        "gate_sha256": sha256_file(args.gate) if args.gate else None,
    }
    if not args.resume and (side.exists() or gate_path.exists()):
        raise FileExistsError("choose a fresh output path or use --resume")
    if side.exists() and json.loads(side.read_text()) != side_data:
        raise ValueError("score provenance differs; refuse resume")
    done = prepare_jsonl_progress(args.output, args.dataset, args.codes_json,
                                  fingerprint, rows, resume=args.resume)
    completed = read_jsonl(args.output) if Path(args.output).exists() else []
    validate_scores(rows, completed, stage, complete=False)
    cache = {r["pair_id"]: r for r in completed
             if stage == "confirmatory" and r["pair_direction"] == 0}
    for record in completed:
        if stage == "confirmatory" and record["pair_direction"] == 1 and record["pair_id"] not in cache:
            raise ValueError("checkpoint edited member lacks its baseline")
    if not side.exists():
        write_json_create(side, side_data)
    pending = [r for r in rows if r["example_id"] not in done]
    logger.info("Stage=%s; %d total records, %d checkpointed, %d remaining",
                stage, len(rows), len(done), len(pending))
    if pending:
        logger.info("Loading pinned model %s at %s", config["model"]["id"], config["model"]["revision"])
        model, tokenizer = load_model(config)
        check_tokenizer(tokenizer, token_audit)
        forwards_per_record = 1 + sum(e["n_tokens"] > 1 for e in token_audit["codes"].values())
        logger.info("Scoring %d code sequences per record; estimated %d model forwards",
                    len(codes), len(pending) * forwards_per_record)
        with logging_redirect_tqdm(), progress(
                total=len(pending) * forwards_per_record,
                desc="Candidate forwards", unit="forward", leave=False) as forwards:
            def scorer(row):
                prompt = render(row, tokenizer, config["model"].get("chat_template", True))
                lp = score_codes(model, tokenizer, prompt, codes, token_audit,
                                 progress_callback=forwards.update)
                accuracy = int(all(lp[row["answer_code"]] > value
                                   for code, value in lp.items() if code != row["answer_code"]))
                result = {**row, "candidate_logprobs": lp, "current_code_accuracy": accuracy}
                if stage in ("development", "frozen_gate"):
                    generated = generate_unrestricted_code(model, tokenizer, prompt, codes, token_audit)
                    generated["unrestricted_code_accuracy"] = int(
                        generated["unrestricted_generated_code"] == row["answer_code"])
                    result.update(generated)
                return result

            for row in progress(pending, desc=f"Scoring {stage}", unit="record"):
                try:
                    score = score_protocol_record(row, scorer, cache, stage)
                    append_jsonl_record(args.output, score)
                except Exception:
                    logger.exception("Scoring interrupted at %s; completed records can be resumed",
                                     row["example_id"])
                    raise
    else:
        logger.info("Checkpoint is complete; model loading skipped")
    scores = read_jsonl(args.output)
    validate_scores(rows, scores, stage)
    if stage == "frozen_gate":
        result = evaluate_competence_gate(rows, scores)
        artifact = seal_artifact({
            "protocol": SCHEMA, "stage": stage, **metadata, **result,
            "dataset_path": str(Path(args.dataset).resolve()),
            "dataset_sha256": sha256_file(args.dataset),
            "dataset_provenance_sha256": sha256_file(args.dataset + ".provenance.json"),
            "scores_path": str(Path(args.output).resolve()), "scores_sha256": sha256_file(args.output),
            "scores_provenance_sha256": sha256_file(side),
            "token_audit_path": str(Path(args.token_audit).resolve()),
            "token_audit_sha256": sha256_file(args.token_audit),
        })
        if gate_path.exists():
            saved = json.loads(gate_path.read_text())
            verify_sealed_artifact(saved)
            if saved != artifact:
                raise ValueError("existing frozen gate differs from recomputed gate")
            logger.info("Verified existing sealed gate on resume")
        else:
            write_json_create(gate_path, artifact)
        logger.info("Gate pass=%s; current-code accuracy=%.4f (%d trials); artifact=%s",
                    result["pass"], result["current_derived_code_accuracy"], result["n"], gate_path)
    logger.info("Completed %s: %d records at %s", stage, len(rows), args.output)


if __name__ == "__main__":
    main()
