#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/primary.yaml}
OUT=${2:-outputs/primary}
uv run python scripts/validate_tokens.py --config "$CONFIG" --output "$OUT/token_ids.json"
uv run python scripts/generate_dataset.py --config "$CONFIG" --token-ids "$OUT/token_ids.json" --output "$OUT/dataset.jsonl"
uv run python scripts/generate_direct.py --config "$CONFIG" --token-ids "$OUT/token_ids.json" --output "$OUT/direct.jsonl"
uv run python scripts/generate_pairs.py --config "$CONFIG" --token-ids "$OUT/token_ids.json" --output "$OUT/pairs.jsonl"
uv run python scripts/audit_dataset.py "$OUT/dataset.jsonl"
uv run python scripts/run_competence.py --config "$CONFIG" --direct "$OUT/direct.jsonl" --overwrite "$OUT/dataset.jsonl" --token-ids "$OUT/token_ids.json" --output "$OUT/competence.json"
uv run python scripts/run_behavior.py --config "$CONFIG" --dataset "$OUT/dataset.jsonl" --token-ids "$OUT/token_ids.json" --output "$OUT/behavior.jsonl"
uv run python scripts/run_behavior.py --config "$CONFIG" --dataset "$OUT/pairs.jsonl" --token-ids "$OUT/token_ids.json" --output "$OUT/pair_behavior.jsonl"
uv run python scripts/analyze.py --behavior "$OUT/behavior.jsonl" --pairs "$OUT/pair_behavior.jsonl" --output-dir "$OUT/analysis"
uv run python scripts/run_extraction.py --config "$CONFIG" --dataset "$OUT/dataset.jsonl" --token-ids "$OUT/token_ids.json" --output "$OUT/activations"
uv run python scripts/run_probes.py --dataset "$OUT/dataset.jsonl" --activations "$OUT/activations" --token-ids "$OUT/token_ids.json" --output "$OUT/probes.json"
uv run python scripts/plot_probes.py "$OUT/probes.json" --output-dir "$OUT/analysis"
uv run python scripts/run_patching.py --config "$CONFIG" --dataset "$OUT/dataset.jsonl" --token-ids "$OUT/token_ids.json" --output "$OUT/patching.jsonl"
