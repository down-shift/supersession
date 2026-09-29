#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/behavioral_replication.yaml}
OUT=${2:-outputs/behavioral_replication}
uv run python scripts/validate_tokens.py --config "$CONFIG" --output "$OUT/token_ids.json"
uv run python scripts/generate_dataset.py --config "$CONFIG" --token-ids "$OUT/token_ids.json" --output "$OUT/dataset.jsonl"
uv run python scripts/generate_direct.py --config "$CONFIG" --token-ids "$OUT/token_ids.json" --output "$OUT/direct.jsonl"
uv run python scripts/generate_pairs.py --config "$CONFIG" --token-ids "$OUT/token_ids.json" --output "$OUT/pairs.jsonl"
uv run python scripts/audit_dataset.py "$OUT/dataset.jsonl"
uv run python scripts/run_competence.py --config "$CONFIG" --direct "$OUT/direct.jsonl" --overwrite "$OUT/dataset.jsonl" --token-ids "$OUT/token_ids.json" --output "$OUT/competence.json"
uv run python scripts/run_behavior.py --config "$CONFIG" --dataset "$OUT/pairs.jsonl" --token-ids "$OUT/token_ids.json" --output "$OUT/pair_behavior.jsonl"
uv run python scripts/analyze.py --behavior "$OUT/competence.json.overwrite.jsonl" --pairs "$OUT/pair_behavior.jsonl" --output-dir "$OUT/analysis"
