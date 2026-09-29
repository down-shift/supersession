#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="${OUT:-outputs/four_query_288}"
CFG="${CFG:-configs/four_query_288.yaml}"
TOK="${TOK:-$OUT/token_ids.json}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable. Install/use an Ubuntu host with the NVIDIA driver and GPU." >&2
  exit 2
fi
nvidia-smi

uv sync --locked --extra model --extra dev
uv run python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot access CUDA; calibration requires the configured CUDA INT8 model.")
print("CUDA device:", torch.cuda.get_device_name(0))
print("PyTorch CUDA runtime:", torch.version.cuda)
import bitsandbytes  # noqa: F401
PY

mkdir -p "$OUT"
if [[ ! -s "$TOK" || "${REVALIDATE_TOKENS:-0}" == "1" ]]; then
  uv run python scripts/validate_tokens.py --config "$CFG" --design four-query --output "$TOK"
fi
uv run python - "$CFG" "$TOK" <<'PY'
import json, sys, yaml
config = yaml.safe_load(open(sys.argv[1], encoding="utf8"))
token_ids = json.load(open(sys.argv[2], encoding="utf8"))["token_ids"]
expected = int(config["dataset"]["candidate_count"])
if len(token_ids) != expected:
    raise SystemExit(f"token_ids.json has {len(token_ids)} values; config requires {expected}")
print(f"Using {len(token_ids)} validated candidate tokens")
PY

uv run python scripts/generate_four_query.py --config "$CFG" --token-ids "$TOK" --calibration --kind queries --output "$OUT/calibration.jsonl"
uv run python scripts/audit_four_query.py "$OUT/calibration.jsonl"
CAL_ARGS=(--config "$CFG" --dataset "$OUT/calibration.jsonl" --token-ids "$TOK" --output "$OUT/calibration.json")
if [[ -f "$OUT/calibration.json.records.jsonl.run.json" ]]; then
  CAL_ARGS+=(--resume)
fi
uv run python scripts/run_four_query_competence.py "${CAL_ARGS[@]}"

uv run python scripts/generate_four_query.py --config "$CFG" --token-ids "$TOK" --kind queries --output "$OUT/behavior_inputs.jsonl"
uv run python scripts/generate_four_query.py --config "$CFG" --token-ids "$TOK" --kind pairs --output "$OUT/pairs.jsonl"
uv run python scripts/audit_four_query.py "$OUT/behavior_inputs.jsonl" --exclude-dataset "$OUT/calibration.jsonl"
uv run python scripts/audit_four_query.py "$OUT/pairs.jsonl" --exclude-dataset "$OUT/calibration.jsonl"

BEHAVIOR_ARGS=(--config "$CFG" --dataset "$OUT/behavior_inputs.jsonl" --token-ids "$TOK" --output "$OUT/behavior.jsonl")
if [[ -f "$OUT/behavior.jsonl.run.json" ]]; then BEHAVIOR_ARGS+=(--resume); fi
uv run python scripts/run_behavior.py "${BEHAVIOR_ARGS[@]}"

PAIRS_ARGS=(--config "$CFG" --dataset "$OUT/pairs.jsonl" --token-ids "$TOK" --output "$OUT/pair_behavior.jsonl")
if [[ -f "$OUT/pair_behavior.jsonl.run.json" ]]; then PAIRS_ARGS+=(--resume); fi
uv run python scripts/run_behavior.py "${PAIRS_ARGS[@]}"

uv run python scripts/analyze_four_query.py --behavior "$OUT/behavior.jsonl" --pairs "$OUT/pair_behavior.jsonl" --output-dir "$OUT/analysis"
echo "Completed four-query 288-history behavior run. Results: $OUT/analysis/four_query_summary.json"
