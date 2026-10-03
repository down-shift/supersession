#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash scripts/run_relational_robustness.sh MODEL validate|generate-development|development|gate|confirmatory|analyze|plan|status' >&2
  exit 2
fi
MODEL="$1"
ACTION="$2"
case "$MODEL" in qwen3_8b|gemma3_4b|phi4_mini) ;; *) echo "Unknown focal model: $MODEL" >&2; exit 2 ;; esac
CONFIG="configs/cross_model_relational_v2/$MODEL.yaml"
SHARED="outputs/cross_model_relational_v2/corrected_20261003"
RUN="${RELATIONAL_RUN_DIR:-$SHARED/$MODEL}"
FREEZE="${RELATIONAL_FREEZE:-$SHARED/protocol_freeze.json}"
PYTHON=(.venv/bin/python -u)
LOCAL=()
if [[ "${RELATIONAL_LOCAL_FILES_ONLY:-0}" == 1 ]]; then LOCAL=(--local-files-only); fi
cli() { "${PYTHON[@]}" -m scripts.robustness_v2 "$@"; }
common=(--config "$CONFIG" --candidates "$RUN/candidates.json")
score() {
  local dataset="$1" output="$2"
  local resume=()
  if [[ -f "$output.run.json" ]]; then resume=(--resume); fi
  cli score "${common[@]}" --dataset "$dataset" --output "$output" "${resume[@]}"
}

case "$ACTION" in
  validate)
    cli validate --config "$CONFIG" --freeze "$FREEZE" --output "$RUN/candidates.json" "${LOCAL[@]}"
    ;;
  generate-development)
    cli generate "${common[@]}" --stage development --output "$RUN/development.jsonl" "${LOCAL[@]}"
    ;;
  development)
    if [[ ! -f "$RUN/development.jsonl" ]]; then
      cli generate "${common[@]}" --stage development --output "$RUN/development.jsonl" "${LOCAL[@]}"
    fi
    score "$RUN/development.jsonl" "$RUN/development_scores.jsonl"
    cli analyze "${common[@]}" --stage development --dataset "$RUN/development.jsonl" \
      --scores "$RUN/development_scores.jsonl" --output "$RUN/development_report.json"
    ;;
  gate)
    cli generate "${common[@]}" --stage frozen_gate --development-report "$RUN/development_report.json" \
      --output "$RUN/gate.jsonl" "${LOCAL[@]}"
    score "$RUN/gate.jsonl" "$RUN/gate_scores.jsonl"
    cli analyze "${common[@]}" --stage frozen_gate --development-report "$RUN/development_report.json" \
      --dataset "$RUN/gate.jsonl" --scores "$RUN/gate_scores.jsonl" --output "$RUN/gate_report.json"
    cli preflight "${common[@]}" --gate "$RUN/gate_report.json" --output "$RUN/preflight.json"
    ;;
  confirmatory)
    cli generate "${common[@]}" --stage confirmatory --gate "$RUN/gate_report.json" \
      --preflight "$RUN/preflight.json" --output "$RUN/confirmatory.jsonl" "${LOCAL[@]}"
    score "$RUN/confirmatory.jsonl" "$RUN/confirmatory_scores.jsonl"
    cli analyze "${common[@]}" --stage confirmatory --dataset "$RUN/confirmatory.jsonl" \
      --scores "$RUN/confirmatory_scores.jsonl" --output "$RUN/confirmatory_analysis.json"
    ;;
  analyze)
    cli analyze "${common[@]}" --stage confirmatory --dataset "$RUN/confirmatory.jsonl" \
      --scores "$RUN/confirmatory_scores.jsonl" --output "$RUN/confirmatory_reanalysis.json"
    ;;
  plan)
    cli plan "${common[@]}" --dataset "$RUN/development.jsonl" --output "$RUN/development_plan.json" "${LOCAL[@]}"
    ;;
  status)
    echo "Run directory: $RUN"
    if [[ -d "$RUN" ]]; then ls -lh "$RUN"; fi
    ;;
  *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
esac
