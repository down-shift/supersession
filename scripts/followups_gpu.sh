#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  echo "Usage: bash scripts/followups_gpu.sh {develop|confirm} {qwen|gemma}" >&2
  exit 2
}

[[ $# == 2 ]] || usage
ACTION=$1
MODEL=$2
[[ "$ACTION" == develop || "$ACTION" == confirm ]] || usage
[[ "$MODEL" == qwen || "$MODEL" == gemma ]] || usage

if [[ "$MODEL" == qwen ]]; then
  export UV_PROJECT_ENVIRONMENT=.venv-qwen-followups
  CONFIG=configs/cross_model_relational_v2/qwen3_8b.yaml
  MODEL_OUT=qwen
else
  export UV_PROJECT_ENVIRONMENT=.venv-gemma-followups
  CONFIG=configs/cross_model_relational_v2/gemma3_4b.yaml
  MODEL_OUT=gemma
fi

uv sync --frozen --python 3.13.5 --extra model --extra dev
if [[ "$ACTION" == develop && "$MODEL" == qwen ]]; then
  uv run --frozen --python 3.13.5 --extra model --extra dev pytest -q tests/test_followups.py
fi
uv run --frozen --python 3.13.5 --extra model --extra dev python -c \
  'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.__version__, torch.cuda.get_device_name(0))'

run_score() {
  local experiment=$1 mode=$2 dataset=$3 output=$4 freeze=${5:-}
  local -a extra=() resume=()
  [[ -z "$freeze" ]] || extra+=(--freeze "$freeze")
  if [[ -f "$output.provenance.json" ]]; then
    echo "Already scored: $output"
    return
  fi
  [[ ! -f "$output.run.json" ]] || resume+=(--resume)
  uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.run_followups score \
    --experiment "$experiment" --mode "$mode" --config "$CONFIG" \
    --dataset "$dataset" --output "$output" "${extra[@]}" "${resume[@]}"
}

run_analysis() {
  local experiment=$1 stage=$2 dataset=$3 scores=$4 output=$5
  [[ -f "$output" ]] && { echo "Already analyzed: $output"; return; }
  uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.run_followups analyze \
    --experiment "$experiment" --stage "$stage" --dataset "$dataset" \
    --scores "$scores" --output "$output"
}

ensure_marker_dataset() {
  local dataset=outputs/followups/marker_confirmatory.jsonl
  local report=outputs/followups/marker_confirmatory_report.json
  if [[ -f "$dataset" && -f "$report" ]]; then return; fi
  if [[ -e "$dataset" || -e "$report" ]]; then
    echo "Incomplete marker dataset/report pair; preserve it and use a fresh output name." >&2
    exit 1
  fi
  uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.followups marker \
    --stage confirmatory --histories 24 \
    --prior-dataset outputs/followups/pilot_marker_rerun1.jsonl \
    --output "$dataset" --report "$report"
}

if [[ "$ACTION" == develop ]]; then
  for N in 2 4 6; do
    DATASET="outputs/followups/development_n${N}.jsonl"
    REPORT="outputs/followups/development_n${N}_report.json"
    if [[ -f "$DATASET" && -f "$REPORT" ]]; then continue; fi
    if [[ -e "$DATASET" || -e "$REPORT" ]]; then
      echo "Incomplete development dataset/report pair for n=${N}; preserve it and use a fresh output name." >&2
      exit 1
    fi
    if [[ "$MODEL" == gemma && ! -f "$DATASET" ]]; then
      echo "Run Qwen development first; its matched development datasets are missing." >&2
      exit 1
    fi
    if [[ "$MODEL" == qwen ]]; then
      uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.followups harder \
        --stage development --histories 12 --distractors "$N" \
        --prior-dataset outputs/followups/pilot_harder_rerun1.jsonl \
        --output "$DATASET" --report "$REPORT"
    fi
  done
  for N in 2 4 6; do
    DATASET="outputs/followups/development_n${N}.jsonl"
    SCORES="outputs/followups/${MODEL_OUT}_development_n${N}_scores.jsonl"
    ANALYSIS="outputs/followups/${MODEL_OUT}_development_n${N}_analysis.json"
    run_score harder both "$DATASET" "$SCORES"
    run_analysis harder development "$DATASET" "$SCORES" "$ANALYSIS"
  done
  if [[ "$MODEL" == qwen ]]; then
    ALL_HIGH=$(uv run --frozen --python 3.13.5 --extra model --extra dev python -c \
      'import json; from pathlib import Path; a=[json.loads(Path(f"outputs/followups/qwen_development_n{n}_analysis.json").read_text())["overall_complete_answer_accuracy"] for n in (2,4,6)]; print(all(x > .90 for x in a))')
    if [[ "$ALL_HIGH" == True ]]; then
      echo "STOP: all Qwen development levels exceed 90%. No fallback selection or harder test. Run: bash scripts/followups_gpu.sh confirm qwen"
      exit 0
    fi
    uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.run_followups select-hard-difficulty \
      --development-reports \
      outputs/followups/qwen_development_n2_analysis.json \
      outputs/followups/qwen_development_n4_analysis.json \
      outputs/followups/qwen_development_n6_analysis.json \
      --output outputs/followups/harder_protocol_freeze.json
    ELIGIBLE=$(uv run --frozen --python 3.13.5 --extra model --extra dev python -c \
      'from src.cross_model.protocol import read_sealed; print(read_sealed("outputs/followups/harder_protocol_freeze.json")["selection_gate_passed"])')
    if [[ "$ELIGIBLE" != True ]]; then
      echo "No development level is in [0.65,0.90]. Freeze records the nearest fallback; harder test is skipped. Run: bash scripts/followups_gpu.sh confirm qwen"
    else
      N=$(uv run --frozen --python 3.13.5 --extra model --extra dev python -c \
        'from src.cross_model.protocol import read_sealed; print(read_sealed("outputs/followups/harder_protocol_freeze.json")["selected_n_distractors"])')
      echo "Selected n=${N}. Gemma development is optional: bash scripts/followups_gpu.sh develop gemma"
    fi
  fi
  exit 0
fi

# Confirm the independent marker factorial even if the harder-task gate failed.
if [[ "$MODEL" == gemma && ! -f outputs/followups/qwen_marker_confirmatory_analysis.json ]]; then
  echo "Run Qwen confirmation first: bash scripts/followups_gpu.sh confirm qwen" >&2
  exit 1
fi
HAS_FREEZE=false
if [[ -f outputs/followups/harder_protocol_freeze.json ]]; then
  HAS_FREEZE=$(uv run --frozen --python 3.13.5 --extra model --extra dev python -c \
    'from src.cross_model.protocol import read_sealed; print(read_sealed("outputs/followups/harder_protocol_freeze.json")["selection_gate_passed"])')
fi
if [[ "$HAS_FREEZE" == True ]]; then
  N=$(uv run --frozen --python 3.13.5 --extra model --extra dev python -c \
    'from src.cross_model.protocol import read_sealed; print(read_sealed("outputs/followups/harder_protocol_freeze.json")["selected_n_distractors"])')
  DATASET="outputs/followups/test_n${N}.jsonl"
  REPORT="outputs/followups/test_n${N}_report.json"
  if [[ ! -f "$DATASET" && ! -f "$REPORT" ]]; then
    uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.followups harder \
      --stage test --histories 24 --distractors "$N" \
      --freeze outputs/followups/harder_protocol_freeze.json \
      --prior-dataset outputs/followups/pilot_harder_rerun1.jsonl \
      --prior-dataset outputs/followups/development_n2.jsonl \
      --prior-dataset outputs/followups/development_n4.jsonl \
      --prior-dataset outputs/followups/development_n6.jsonl \
      --output "$DATASET" --report "$REPORT"
  elif [[ ! -f "$DATASET" || ! -f "$REPORT" ]]; then
    echo "Incomplete test dataset/report pair; preserve it and use a fresh output name." >&2
    exit 1
  fi
  if [[ "$MODEL" == gemma && ! -f "outputs/followups/qwen_test_n${N}_analysis.json" ]]; then
    echo "Run Qwen harder test first: bash scripts/followups_gpu.sh confirm qwen" >&2
    exit 1
  fi
  SCORES="outputs/followups/${MODEL_OUT}_test_n${N}_scores.jsonl"
  ANALYSIS="outputs/followups/${MODEL_OUT}_test_n${N}_analysis.json"
  run_score harder both "$DATASET" "$SCORES" outputs/followups/harder_protocol_freeze.json
  run_analysis harder test "$DATASET" "$SCORES" "$ANALYSIS"
else
  echo "No qualifying Qwen freeze; harder test skipped."
fi

ensure_marker_dataset
MARKER_SCORES="outputs/followups/${MODEL_OUT}_marker_confirmatory_scores.jsonl"
MARKER_ANALYSIS="outputs/followups/${MODEL_OUT}_marker_confirmatory_analysis.json"
run_score marker candidate outputs/followups/marker_confirmatory.jsonl "$MARKER_SCORES"
run_analysis marker confirmatory outputs/followups/marker_confirmatory.jsonl "$MARKER_SCORES" "$MARKER_ANALYSIS"
