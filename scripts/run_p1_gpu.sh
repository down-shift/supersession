#!/usr/bin/env bash
# Score and analyze the P1 follow-ups (docs/distance_v1.md, docs/marker_construction_v2.md) on the
# experiment GPU. Datasets must already exist; design records must predate them. Waits until no other
# process holds the GPU, never overwrites a completed artifact, and resumes partial score files.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH" HF_HUB_OFFLINE=1

wait_for_free_gpu() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do
    echo "$(date -u +%FT%TZ) GPU busy; waiting"; sleep 120
  done
}

run() {  # experiment model_key venv_model config dataset outdir
  local experiment=$1 key=$2 env=$3 config=$4 dataset=$5 out=$6
  local scores="$out/${key}_${experiment}_scores.jsonl" analysis="$out/${key}_${experiment}_analysis.json"
  local py=(env UV_PROJECT_ENVIRONMENT=".venv-p1-$env" PYTHONPATH=. uv run --frozen --python 3.13.5
            --extra model --extra dev python)
  for record in "$out"/design_record*.json; do
    [[ "$record" -ot "$dataset" ]] || { echo "design record $record does not predate $dataset" >&2; exit 1; }
  done
  if [[ ! -f "$scores.provenance.json" ]]; then
    local -a resume=()
    [[ -f "$scores.run.json" ]] && resume=(--resume)
    wait_for_free_gpu
    echo "$(date -u +%FT%TZ) scoring $experiment / $key"
    "${py[@]}" -m scripts.run_followups score --experiment "$experiment" --mode candidate --config "$config" \
      --dataset "$dataset" --output "$scores" "${resume[@]}" > "$scores.log" 2>&1
  fi
  if [[ ! -f "$analysis" ]]; then
    "${py[@]}" -m scripts.run_followups analyze --experiment "$experiment" --stage confirmatory \
      --dataset "$dataset" --scores "$scores" --output "$analysis"
  fi
  echo "$(date -u +%FT%TZ) done $experiment / $key"
}

M=outputs/followups/marker96 D=outputs/followups/distance_v1
QWEN=configs/cross_model_relational_v2/qwen3_8b.yaml GEMMA=configs/cross_model_relational_v2/gemma3_4b.yaml
run marker gemma gemma "$GEMMA" "$M/marker96_confirmatory.jsonl" "$M"
run distance gemma gemma "$GEMMA" "$D/distance_confirmatory.jsonl" "$D"
run marker qwen qwen "$QWEN" "$M/marker96_confirmatory.jsonl" "$M"
run distance qwen qwen "$QWEN" "$D/distance_confirmatory.jsonl" "$D"
echo "$(date -u +%FT%TZ) all P1 runs complete"
