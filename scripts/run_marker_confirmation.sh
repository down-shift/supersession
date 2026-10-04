#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATASET=outputs/followups/marker_confirmatory.jsonl
REPORT=outputs/followups/marker_confirmatory_report.json
PILOT=outputs/followups/pilot_marker_rerun1.jsonl

if [[ ! -s "$PILOT" ]]; then
  echo "Required completed marker pilot is missing: $PILOT" >&2
  exit 1
fi

run_model() {
  local model=$1 env_name=$2 config=$3
  local scores="outputs/followups/${model}_marker_confirmatory_scores.jsonl"
  local analysis="outputs/followups/${model}_marker_confirmatory_analysis.json"

  export UV_PROJECT_ENVIRONMENT="$env_name"
  uv sync --frozen --python 3.13.5 --extra model --extra dev
  uv run --frozen --python 3.13.5 --extra model --extra dev python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.__version__, torch.cuda.get_device_name(0))'

  if [[ -f "$scores.provenance.json" ]]; then
    [[ -s "$scores" ]] || { echo "Provenance exists but scores are missing: $scores" >&2; exit 1; }
    echo "Score complete: $scores"
  else
    local -a resume=()
    if [[ -f "$scores.run.json" ]]; then
      resume+=(--resume)
    elif [[ -e "$scores" ]]; then
      echo "Score file has no resumable run record: $scores; preserve it and use a fresh output name." >&2
      exit 1
    fi
    uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.run_followups score \
      --experiment marker --mode candidate --config "$config" \
      --dataset "$DATASET" --output "$scores" "${resume[@]}"
  fi

  if [[ ! -f "$analysis" ]]; then
    uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.run_followups analyze \
      --experiment marker --stage confirmatory --dataset "$DATASET" \
      --scores "$scores" --output "$analysis"
  else
    echo "Analysis complete: $analysis"
  fi
}

export UV_PROJECT_ENVIRONMENT=.venv-qwen-followups
uv sync --frozen --python 3.13.5 --extra model --extra dev
if [[ -f "$DATASET" && -f "$REPORT" ]]; then
  echo "Using existing marker confirmation dataset: $DATASET"
elif [[ -e "$DATASET" || -e "$REPORT" ]]; then
  echo "Incomplete marker dataset/report pair; preserve it and use fresh output names." >&2
  exit 1
else
  uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.followups marker \
    --stage confirmatory --histories 24 \
    --prior-dataset "$PILOT" --output "$DATASET" --report "$REPORT"
fi

run_model qwen .venv-qwen-followups configs/cross_model_relational_v2/qwen3_8b.yaml
run_model gemma .venv-gemma-followups configs/cross_model_relational_v2/gemma3_4b.yaml

echo "Marker × construction confirmation finished for Qwen and Gemma."
