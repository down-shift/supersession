#!/usr/bin/env bash
# P1.3 update-chain study (docs/chain_v1.md) on the experiment GPU: development at depths 3/4/5,
# the recorded depth-selection rule, then confirmation at the selected depth. Starts only after the
# P1.1/P1.2 runner has exited and the GPU is free; never overwrites completed artifacts.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1
OUT=outputs/followups/chain_v1
QWEN=configs/cross_model_relational_v2/qwen3_8b.yaml GEMMA=configs/cross_model_relational_v2/gemma3_4b.yaml

while pgrep -f run_p1_gpu.sh >/dev/null; do echo "$(date -u +%FT%TZ) waiting for P1.1/P1.2 runner"; sleep 300; done

py() { local env=$1; shift; env UV_PROJECT_ENVIRONMENT=".venv-p1c-$env" PYTHONPATH=. uv run --frozen --python 3.13.5 --extra model --extra dev python "$@"; }

wait_for_free_gpu() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "$(date -u +%FT%TZ) GPU busy; waiting"; sleep 120
  done
}

score_and_analyze() {  # stage dataset key env config
  local stage=$1 dataset=$2 key=$3 env=$4 config=$5
  local stem="${dataset%.jsonl}_${key}"
  if [[ ! -f "${stem}_scores.jsonl.provenance.json" ]]; then
    local -a resume=(); [[ -f "${stem}_scores.jsonl.run.json" ]] && resume=(--resume)
    wait_for_free_gpu
    echo "$(date -u +%FT%TZ) scoring $dataset / $key"
    py "$env" -m scripts.run_followups score --experiment chain --mode both --config "$config" \
      --dataset "$dataset" --dataset-report "${dataset%.jsonl}_report.json" \
      --output "${stem}_scores.jsonl" "${resume[@]}" > "${stem}_scores.log" 2>&1
  fi
  [[ -f "${stem}_analysis.json" ]] || py "$env" -m scripts.run_followups analyze --experiment chain \
    --stage "$stage" --dataset "$dataset" --scores "${stem}_scores.jsonl" --output "${stem}_analysis.json"
  echo "$(date -u +%FT%TZ) done $dataset / $key"
}

for d in 3 4 5; do
  score_and_analyze development "$OUT/development_d$d.jsonl" qwen qwen "$QWEN"
  score_and_analyze development "$OUT/development_d$d.jsonl" gemma gemma "$GEMMA"
done

if [[ ! -f "$OUT/depth_selection.json" ]]; then
  py qwen -m scripts.chains select-depth --analyses "$OUT"/development_d*_analysis.json \
    --output "$OUT/depth_selection.json"
fi
DEPTH=$(python3 -c "import json;d=json.load(open('$OUT/depth_selection.json'));print(d['selected_depth'] or '')")
if [[ -z "$DEPTH" ]]; then echo "$(date -u +%FT%TZ) no qualifying depth; study stops (recorded rule)"; exit 0; fi

C="$OUT/confirmatory_d$DEPTH.jsonl"
if [[ ! -f "$C" ]]; then
  py qwen -m scripts.chains generate --stage confirmatory --depth "$DEPTH" --histories 96 --seed 20261108 \
    --vocabulary-audit "$OUT/vocabulary_audit.json" \
    --prior-dataset "$OUT/development_d3.jsonl" --prior-dataset "$OUT/development_d4.jsonl" \
    --prior-dataset "$OUT/development_d5.jsonl" --output "$C" --report "${C%.jsonl}_report.json"
fi
score_and_analyze confirmatory "$C" qwen qwen "$QWEN"
score_and_analyze confirmatory "$C" gemma gemma "$GEMMA"
echo "$(date -u +%FT%TZ) chain study complete"
