#!/usr/bin/env bash
set -Eeuo pipefail
trap 'status=$?; printf "RUN FAILED (exit %s) at line %s: %s\n" "$status" "$LINENO" "$BASH_COMMAND" >&2' ERR
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
OUT="${OUT:-outputs/four_query_288}"; CFG="${CFG:-configs/four_query_288.yaml}"; TOK="$OUT/token_ids.json"
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi is unavailable" >&2; exit 2; }; nvidia-smi
uv sync --locked --extra model --extra dev
uv run python - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("PyTorch cannot access CUDA")
import bitsandbytes
print("CUDA device:",torch.cuda.get_device_name(0))
PY
mkdir -p "$OUT"
uv run python scripts/validate_tokens.py --config "$CFG" --design four-query --output "$TOK"
for variant in first_latest initial_update timestamped; do
  uv run python scripts/generate_four_query.py --config "$CFG" --token-ids "$TOK" --partition prompt_dev --prompt-variant "$variant" --kind queries --output "$OUT/prompt_dev_${variant}.jsonl"
  uv run python scripts/audit_four_query.py "$OUT/prompt_dev_${variant}.jsonl"
  args=(--config "$CFG" --dataset "$OUT/prompt_dev_${variant}.jsonl" --token-ids "$TOK" --output "$OUT/prompt_dev_${variant}.json" --diagnostic-only)
  [[ -f "$OUT/prompt_dev_${variant}.json.records.jsonl.run.json" ]] && args+=(--resume)
  uv run python scripts/run_four_query_competence.py "${args[@]}"
done
uv run python scripts/analyze_prompt_development.py --config "$CFG" --token-ids "$TOK" --provenance "$OUT/prompt_dev_first_latest.json" \
  --records "first_latest=$OUT/prompt_dev_first_latest.json.records.jsonl" \
  --records "initial_update=$OUT/prompt_dev_initial_update.json.records.jsonl" \
  --records "timestamped=$OUT/prompt_dev_timestamped.json.records.jsonl" --output "$OUT/prompt_selection.json"
uv run python - "$OUT/prompt_selection.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print("Frozen prompt variant:",x["selected_variant"])
if not x.get("confirmatory_permitted"): raise SystemExit("prompt development did not authorize confirmatory use")
json.dump({"token_ids":x["token_ids"],"prompt_variant":x["selected_variant"],"selection_artifact":"prompt_selection.json","model_revision":x["model_revision"],"tokenizer_revision":x["tokenizer_revision"]},open(sys.argv[1].replace("prompt_selection.json","frozen_token_ids.json"),"w"),indent=2)
PY
TOK="$OUT/frozen_token_ids.json"
uv run python scripts/generate_four_query.py --config "$CFG" --token-ids "$TOK" --partition gate --prompt-variant "$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_variant"])' "$OUT/prompt_selection.json")" --kind queries --output "$OUT/gate.jsonl"
uv run python scripts/audit_four_query.py "$OUT/gate.jsonl" --exclude-dataset "$OUT/prompt_dev_first_latest.jsonl" --exclude-dataset "$OUT/prompt_dev_initial_update.jsonl" --exclude-dataset "$OUT/prompt_dev_timestamped.jsonl"
args=(--config "$CFG" --dataset "$OUT/gate.jsonl" --token-ids "$TOK" --output "$OUT/gate.json")
[[ -f "$OUT/gate.json.records.jsonl.run.json" ]] && args+=(--resume)
uv run python scripts/run_four_query_competence.py "${args[@]}"
VARIANT="$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["selected_variant"])' "$OUT/prompt_selection.json")"
uv run python scripts/generate_four_query.py --config "$CFG" --token-ids "$TOK" --partition confirmatory --prompt-variant "$VARIANT" --kind queries --output "$OUT/behavior_inputs.jsonl"
uv run python scripts/generate_four_query.py --config "$CFG" --token-ids "$TOK" --partition confirmatory --prompt-variant "$VARIANT" --kind pairs --output "$OUT/pairs.jsonl"
for data in "$OUT/behavior_inputs.jsonl" "$OUT/pairs.jsonl"; do
  uv run python scripts/audit_four_query.py "$data" --exclude-dataset "$OUT/prompt_dev_first_latest.jsonl" --exclude-dataset "$OUT/prompt_dev_initial_update.jsonl" --exclude-dataset "$OUT/prompt_dev_timestamped.jsonl" --exclude-dataset "$OUT/gate.jsonl"
done
args=(--config "$CFG" --dataset "$OUT/behavior_inputs.jsonl" --token-ids "$TOK" --output "$OUT/behavior.jsonl"); [[ -f "$OUT/behavior.jsonl.run.json" ]] && args+=(--resume); uv run python scripts/run_behavior.py "${args[@]}"
args=(--config "$CFG" --dataset "$OUT/pairs.jsonl" --token-ids "$TOK" --output "$OUT/pair_behavior.jsonl"); [[ -f "$OUT/pair_behavior.jsonl.run.json" ]] && args+=(--resume); uv run python scripts/run_behavior.py "${args[@]}"
uv run python scripts/analyze_four_query.py --behavior "$OUT/behavior.jsonl" --pairs "$OUT/pair_behavior.jsonl" --output-dir "$OUT/analysis"
