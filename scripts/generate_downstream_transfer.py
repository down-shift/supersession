#!/usr/bin/env python3
"""Create isolated downstream stages without overwriting artifacts."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_model.progress import configure_logging
from src.data.downstream_transfer import COUNTS, SCHEMA, audit, generate
from src.data.io import read_jsonl, sha256_file
from src.experiments.downstream_transfer import (
    frozen_metadata, validate_gate, validate_token_audit, write_json_create,
)
from src.utils import load_config

logger = logging.getLogger(__name__)


def validate_exclusion_dataset(path, values, codes):
    """Validate a prior downstream dataset as a fresh-history exclusion.

    Model/config provenance may differ across model panels; dataset bytes,
    schema, and semantic vocabularies must still be intact.
    """
    rows = read_jsonl(path)
    audit(rows)
    provenance_path = Path(str(path) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get("protocol") != SCHEMA or
            provenance.get("stage") != rows[0]["stage"] or
            provenance.get("dataset_sha256") != sha256_file(path)):
        raise ValueError(f"prior dataset integrity/protocol mismatch: {path}")
    if any(row["value_vocabulary"] != values or row["code_vocabulary"] != codes for row in rows):
        raise ValueError(f"prior dataset uses different value/code vocabularies: {path}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(COUNTS), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior-dataset", action="append", default=[])
    parser.add_argument("--gate")
    parser.add_argument("--token-audit")
    parser.add_argument("--config", default="configs/downstream_transfer_v1.yaml")
    parser.add_argument("--values-json", default="configs/downstream_transfer_values.json")
    parser.add_argument("--codes-json", default="configs/downstream_transfer_codes.json")
    args = parser.parse_args(argv)
    configure_logging()
    output = Path(args.output)
    if any(Path(str(output) + suffix).exists() for suffix in ("", ".provenance.json")):
        raise FileExistsError("preserve prior outputs; choose a new path")
    config = load_config(args.config)
    metadata = frozen_metadata(config, args.config, args.values_json, args.codes_json)
    values = json.loads(Path(args.values_json).read_text())
    codes = json.loads(Path(args.codes_json).read_text())
    priors = list(args.prior_dataset)
    if args.stage == "confirmatory":
        if not args.gate or not args.token_audit:
            raise ValueError("confirmatory generation requires --gate and --token-audit")
        gate, gate_prov = validate_gate(args.gate, metadata, args.token_audit)
        validate_token_audit(args.token_audit, config, metadata, codes)
        priors += gate_prov["prior_dataset_paths"] + [gate["dataset_path"]]
        logger.info("Verified passing frozen gate: %s", args.gate)
    priors = list(dict.fromkeys(str(Path(p).resolve()) for p in priors))
    excluded = []
    has_development = False
    for path in priors:
        logger.info("Loading prior histories: %s", path)
        prior_rows = read_jsonl(path)
        if prior_rows and prior_rows[0].get("schema") == SCHEMA:
            prior_rows = validate_exclusion_dataset(path, values, codes)
            has_development |= prior_rows[0]["stage"] == "development"
        excluded.extend(prior_rows)
    if args.stage == "frozen_gate" and not has_development:
        raise ValueError("frozen gate generation requires a development --prior-dataset")
    rows = generate(args.stage, COUNTS[args.stage], values, codes,
                    excluded=excluded, show_progress=True,
                    seed_profile=config.get("data_seed_profile", "qwen3_8b_v1"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    provenance = {
        "protocol": SCHEMA, "stage": args.stage, **metadata,
        "dataset_path": str(output.resolve()), "dataset_sha256": sha256_file(output),
        "token_audit_sha256": sha256_file(args.token_audit) if args.token_audit else None,
        "prior_dataset_paths": priors,
        "prior_dataset_sha256": [sha256_file(path) for path in priors],
        "gate_path": str(Path(args.gate).resolve()) if args.gate else None,
        "gate_sha256": sha256_file(args.gate) if args.gate else None,
    }
    write_json_create(str(output) + ".provenance.json", provenance)
    logger.info("Wrote %d %s records to %s", len(rows), args.stage, output)


if __name__ == "__main__":
    main()
