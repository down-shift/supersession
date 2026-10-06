#!/usr/bin/env bash
# Round 2, reordered (2026-10-05): the Granite screen runs ~10x slower than the other models, so the
# remaining screens go first in order of expected speed, and full runs follow (Granite last).
# Same scoring code, rules and configs as scripts/run_p3_gpu.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1 PYTHONWARNINGS=ignore::UserWarning
R=/home/danya/supersession/outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004
GATE=$R/qwen3_8b/migrated/gate.jsonl CONF=$R/qwen3_8b/confirmatory.jsonl
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_for_free_gpu() { while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 120; done; }
py() { env UV_PROJECT_ENVIRONMENT=.venv-p2-qwen PYTHONPATH=. uv run --frozen --python 3.13.5 --extra model --extra dev python "$@"; }
passed() { [[ -f "outputs/p2b/$1/gate_report.json" ]] && [[ "$(python3 -c "import json;print(json.load(open('outputs/p2b/$1/gate_report.json'))['evaluation']['pass'])")" == True ]]; }
screen() {  # key
  local key=$1 out=outputs/p2b/$1; mkdir -p "$out"
  if [[ ! -f "$out/gate_report.json" ]]; then
    wait_for_free_gpu; log "screen $key"
    py scripts/score_exp1_model.py --config "configs/p2/$key.yaml" --stage gate --dataset "$GATE" --out "$out" \
      > "$out/gate.log" 2>&1 || { log "screen FAILED to run for $key"; return 1; }
  fi
  log "$key screen pass=$(python3 -c "import json;print(json.load(open('$out/gate_report.json'))['evaluation']['pass'])")"
}
confirm() {  # key
  local key=$1 out=outputs/p2b/$1
  passed "$key" || return 0
  if [[ ! -f "$out/confirmatory_analysis.json" ]]; then
    wait_for_free_gpu; log "confirm $key"
    py scripts/score_exp1_model.py --config "configs/p2/$key.yaml" --stage confirmatory --dataset "$CONF" --out "$out" \
      > "$out/confirmatory.log" 2>&1 || log "confirmation FAILED for $key"
  fi
  log "done $key"
}
screen qwen25_7b
screen falcon3_7b
screen gemma3_12b || screen gemma3_12b_nf4
screen qwen3_14b_nf4
screen granite31_8b   # normally already finished by the earlier runner's orphaned scorer
for m in qwen25_7b falcon3_7b gemma3_12b gemma3_12b_nf4 qwen3_14b_nf4 granite31_8b; do confirm "$m"; done
log "round 2 (reordered) complete"
