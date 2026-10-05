#!/usr/bin/env bash
# Review round 2 (docs/update_control_v1.md, docs/chain_scores_v1.md, docs/olmo_generation_v1.md,
# docs/models_v2.md) on the experiment GPU. Datasets and design records must already exist; waits for
# a free GPU before each scoring job; never overwrites completed artifacts; continues past failures.
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
QWEN=configs/cross_model_relational_v2/qwen3_8b.yaml GEMMA=configs/cross_model_relational_v2/gemma3_4b.yaml
R=/home/danya/supersession/outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004
GATE=$R/qwen3_8b/migrated/gate.jsonl CONF=$R/qwen3_8b/confirmatory.jsonl
log() { echo "$(date -u +%FT%TZ) $*"; }
wait_for_free_gpu() { while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 120; done; }
py() { local env=$1; shift; env UV_PROJECT_ENVIRONMENT=".venv-p2-$env" PYTHONPATH=. uv run --frozen --python 3.13.5 --extra model --extra dev python "$@"; }

followup() {  # experiment stage dataset key env config
  local exp=$1 stage=$2 dataset=$3 key=$4 env=$5 config=$6 stem="${3%.jsonl}_$4"
  if [[ ! -f "${stem}_scores.jsonl.provenance.json" ]]; then
    local -a resume=(); [[ -f "${stem}_scores.jsonl.run.json" ]] && resume=(--resume)
    wait_for_free_gpu; log "scoring $exp $dataset / $key"
    py "$env" -m scripts.run_followups score --experiment "$exp" --mode candidate --config "$config" \
      --dataset "$dataset" --dataset-report "${dataset%.jsonl}_report.json" --output "${stem}_scores.jsonl" \
      "${resume[@]}" > "${stem}_scores.log" 2>&1 || { log "FAILED scoring $exp $key"; return; }
  fi
  [[ -f "${stem}_analysis.json" ]] || py "$env" -m scripts.run_followups analyze --experiment "$exp" \
    --stage "$stage" --dataset "$dataset" --scores "${stem}_scores.jsonl" --output "${stem}_analysis.json" \
    || log "FAILED analysis $exp $key"
  log "done $exp $dataset / $key"
}

exp1_model() {  # key config [extra flags...]
  local key=$1 config=$2; shift 2; local out=outputs/p2b/$key
  mkdir -p "$out"
  if [[ ! -f "$out/gate_report.json" ]]; then
    wait_for_free_gpu; log "screen $key"
    py qwen scripts/score_exp1_model.py --config "$config" --stage gate --dataset "$GATE" --out "$out" \
      > "$out/gate.log" 2>&1 || { log "screen FAILED to run for $key"; return 1; }
  fi
  log "$key screen pass=$(python3 -c "import json;print(json.load(open('$out/gate_report.json'))['evaluation']['pass'])")"
}

confirm_if_passed() {  # key config
  local key=$1 config=$2 out=outputs/p2b/$1
  [[ -f "$out/gate_report.json" ]] || return
  [[ "$(python3 -c "import json;print(json.load(open('$out/gate_report.json'))['evaluation']['pass'])")" == True ]] || return
  if [[ ! -f "$out/confirmatory_analysis.json" ]]; then
    wait_for_free_gpu; log "confirm $key"
    py qwen scripts/score_exp1_model.py --config "$config" --stage confirmatory --dataset "$CONF" --out "$out" \
      > "$out/confirmatory.log" 2>&1 || log "confirmation FAILED for $key"
  fi
  log "done $key"
}

U=outputs/followups/update_v1/update_confirmatory.jsonl
followup update confirmatory "$U" gemma gemma "$GEMMA"
followup update confirmatory "$U" qwen qwen "$QWEN"
for d in 3 4 5; do
  C=outputs/followups/chain_scores_v1/confirmatory_d$d.jsonl
  followup chain confirmatory "$C" gemma gemma "$GEMMA"
  followup chain confirmatory "$C" qwen qwen "$QWEN"
done

O=outputs/p2b/olmo2_7b
mkdir -p "$O"
if [[ ! -f "$O/generation_analysis.json" ]]; then
  cp -n /home/danya/supersession-p1c/outputs/p2/olmo2_7b/gate_report.json "$O/gate_report.json"
  wait_for_free_gpu; log "OLMo-2 generation study"
  py qwen scripts/score_exp1_model.py --config configs/p2/olmo2_7b.yaml --stage confirmatory --dataset "$CONF" \
    --out "$O" --descriptive --generate > "$O/generation.log" 2>&1 || log "OLMo generation FAILED"
fi

exp1_model granite31_8b configs/p2/granite31_8b.yaml
exp1_model falcon3_7b configs/p2/falcon3_7b.yaml
exp1_model qwen25_7b configs/p2/qwen25_7b.yaml
exp1_model gemma3_12b configs/p2/gemma3_12b.yaml || exp1_model gemma3_12b_nf4 configs/p2/gemma3_12b_nf4.yaml
exp1_model qwen3_14b_nf4 configs/p2/qwen3_14b_nf4.yaml
for m in granite31_8b falcon3_7b qwen25_7b gemma3_12b gemma3_12b_nf4 qwen3_14b_nf4; do
  confirm_if_passed "$m" "configs/p2/$m.yaml"
done
log "round 2 complete"
