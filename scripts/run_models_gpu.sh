#!/usr/bin/env bash
# Additional models on Experiment 1's sealed datasets (docs/models_v2.md), for one machine.
#   bash scripts/run_models_gpu.sh GATE.jsonl CONF.jsonl VENV model_key [model_key ...]
# Screens every listed model first, then runs the confirmation for each model that passed.
# Waits for a free GPU before each job; never overwrites completed artifacts.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore::UserWarning
GATE=$1 CONF=$2 VENV=$3; shift 3; MODELS=("$@")
log() { echo "$(date -u +%FT%TZ) $(hostname) $*"; }
wait_for_free_gpu() { while [[ -n "$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | awk -F', ' '$2+0 > 1000')" ]]; do sleep 120; done; }
py() { env UV_PROJECT_ENVIRONMENT="$VENV" PYTHONPATH=. uv run --frozen --python 3.13.5 --extra model --extra dev python "$@"; }
pass_of() { python3 -c "import json;print(json.load(open('outputs/p2b/$1/gate_report.json'))['evaluation']['pass'])"; }
for key in "${MODELS[@]}"; do
  out=outputs/p2b/$key; mkdir -p "$out"
  if [[ ! -f "$out/gate_report.json" ]]; then
    wait_for_free_gpu; log "screen $key"
    py scripts/score_exp1_model.py --config "configs/p2/$key.yaml" --stage gate --dataset "$GATE" --out "$out" \
      > "$out/gate.log" 2>&1 || { log "screen FAILED to run for $key"; continue; }
  fi
  log "$key screen pass=$(pass_of "$key")"
done
for key in "${MODELS[@]}"; do
  out=outputs/p2b/$key
  [[ -f "$out/gate_report.json" && "$(pass_of "$key")" == True ]] || continue
  if [[ ! -f "$out/confirmatory_analysis.json" ]]; then
    wait_for_free_gpu; log "confirm $key"
    py scripts/score_exp1_model.py --config "configs/p2/$key.yaml" --stage confirmatory --dataset "$CONF" --out "$out" \
      > "$out/confirmatory.log" 2>&1 || log "confirmation FAILED for $key"
  fi
  log "done $key"
done
log "models complete: ${MODELS[*]}"
