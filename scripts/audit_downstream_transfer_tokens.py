#!/usr/bin/env python3
"""Tokenizer-only complete-sequence audit; does not load or run a model."""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cross_model.progress import configure_logging, progress
from src.cross_model.protocol import digest
from src.data.downstream_transfer import SCHEMA, generate, render, seal_artifact
from src.experiments.downstream_transfer import candidate_events, frozen_metadata, write_json_create
from src.utils import load_config

logger = logging.getLogger(__name__)


def audit_tokenizer(tokenizer, rows, codes, chat=True):
    canonical = None
    for row in progress(rows, desc="Auditing code continuations", unit="prompt"):
        events = candidate_events(tokenizer, render(row, tokenizer, chat), codes)
        sequences = {code: {"token_ids": members[0]["ids"], "n_tokens": len(members[0]["ids"])}
                     for code, members in events.items()}
        if canonical is not None and sequences != canonical:
            raise ValueError("code sequences change across prompt/query/edit contexts")
        canonical = sequences
    return canonical


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/downstream_transfer_v1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--values-json", default="configs/downstream_transfer_values.json")
    parser.add_argument("--codes-json", default="configs/downstream_transfer_codes.json")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    configure_logging()
    if Path(args.output).exists():
        raise FileExistsError("preserve existing tokenizer audit; choose a new path")
    config = load_config(args.config)
    metadata = frozen_metadata(config, args.config, args.values_json, args.codes_json)
    model = config["model"]
    ident = model.get("tokenizer_id") or model["id"]
    logger.info("Loading tokenizer only: %s at %s", ident, model["tokenizer_revision"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        ident, revision=model["tokenizer_revision"], local_files_only=args.local_files_only,
    )
    values = json.loads(Path(args.values_json).read_text())
    codes = json.loads(Path(args.codes_json).read_text())
    rows = generate("development", 24, values, codes)
    # Offline edited representatives cover both entity orientations and both
    # queries without generating a confirmatory dataset or evaluating a model.
    edits = []
    for row in rows[:4]:
        replacement = next(value for value in values if value not in row["matching_values"].values())
        for binding in ("old_x", "old_z", "current_x", "current_z"):
            edits.append({**row, "stage": "confirmatory", "edited_binding": binding,
                          "replacement_value": replacement, "pair_direction": 1})
    representatives = rows + edits
    logger.info("Auditing %d code sequences across %d rendered prefixes", len(codes), len(representatives))
    code_audit = audit_tokenizer(tokenizer, representatives, codes, model.get("chat_template", True))
    document = seal_artifact({
        "protocol": SCHEMA, "audit": "tokenizer_only_no_model_inference", **metadata,
        "tokenizer_id": ident, "tokenizer_revision": model["tokenizer_revision"],
        "chat_template": model.get("chat_template", True),
        "continuation_prefix_stable": True, "prompts_checked": len(representatives),
        "codes": code_audit,
        "tokenizer_sha256": digest(tokenizer.backend_tokenizer.to_str()),
        "chat_template_sha256": digest(getattr(tokenizer, "chat_template", None)),
        "scoring_note": "complete spaced code-sequence prefix probability; no termination or length normalization",
    })
    write_json_create(args.output, document)
    logger.info("Tokenizer audit passed: token lengths=%s; artifact=%s",
                sorted({entry["n_tokens"] for entry in code_audit.values()}), args.output)


if __name__ == "__main__":
    main()
