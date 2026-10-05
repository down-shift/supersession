#!/usr/bin/env bash
# Plan P2: additional models on Experiment 1's sealed datasets (scripts/score_exp1_model.py).
# Starts after the P1.3 chain runner exits and the GPU is free. Order: OLMo-2-7B (Llama access was rejected), Phi-4-mini
# (descriptive confirmation; it failed the screen), Qwen3-14B (may not fit in 16 GB; failure recorded).
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
R=/home/danya/supersession/outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004
GATE=$R/qwen3_8b/migrated/gate.jsonl CONF=$R/qwen3_8b/confirmatory.jsonl
while pgrep -f run_p1c_gpu.sh >/dev/null || pgrep -f run_p1_gpu.sh >/dev/null; do sleep 300; done
wait_for_free_gpu() { while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 120; done; }
py() { env UV_PROJECT_ENVIRONMENT=.venv-p1c-qwen PYTHONPATH=. uv run --frozen --python 3.13.5 --extra model --extra dev python "$@"; }

model() {  # key config confirm_flag
  local key=$1 config=$2 flag=${3:-}; local out=outputs/p2/$key
  mkdir -p "$out"
  if [[ ! -f "$out/gate_report.json" ]]; then
    wait_for_free_gpu; echo "$(date -u +%FT%TZ) screen $key"
    py scripts/score_exp1_model.py --config "$config" --stage gate --dataset "$GATE" --out "$out" > "$out/gate.log" 2>&1 \
      || { echo "$(date -u +%FT%TZ) screen FAILED to run for $key (see $out/gate.log)"; return; }
  fi
  local pass; pass=$(python3 -c "import json;print(json.load(open('$out/gate_report.json'))['evaluation']['pass'])")
  echo "$(date -u +%FT%TZ) $key screen pass=$pass"
  if [[ "$pass" != True && "$flag" != --descriptive ]]; then return; fi
  if [[ ! -f "$out/confirmatory_analysis.json" ]]; then
    wait_for_free_gpu; echo "$(date -u +%FT%TZ) confirm $key $flag"
    py scripts/score_exp1_model.py --config "$config" --stage confirmatory --dataset "$CONF" --out "$out" $flag \
      > "$out/confirmatory.log" 2>&1 || echo "$(date -u +%FT%TZ) confirmation FAILED for $key"
  fi
  echo "$(date -u +%FT%TZ) done $key"
}
model olmo2_7b configs/p2/olmo2_7b.yaml
model phi4_mini configs/cross_model_relational_v2/phi4_mini.yaml --descriptive
model qwen3_14b configs/p2/qwen3_14b.yaml
echo "$(date -u +%FT%TZ) P2 complete"
