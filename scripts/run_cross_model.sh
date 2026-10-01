#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_cross_model.sh MODEL prepare
  bash scripts/run_cross_model.sh MODEL confirmatory
  bash scripts/run_cross_model.sh MODEL mechanism
  bash scripts/run_cross_model.sh MODEL heads
  bash scripts/run_cross_model.sh MODEL sensitivity
  bash scripts/run_cross_model.sh MODEL resume
  bash scripts/run_cross_model.sh MODEL status

MODEL: qwen3_8b | mistral7b | phi4_mini | llama31_8b

Set CROSS_MODEL_RUN_DIR to override the default outputs/MODEL_review2 directory.
prepare stops after writing the frozen gate and preflight report. Review those
files before invoking confirmatory.
EOF
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then usage; exit 0; fi
if [[ $# -ne 2 ]]; then usage >&2; exit 2; fi
MODEL="$1"
ACTION="$2"
case "$MODEL" in qwen3_8b|mistral7b|phi4_mini|llama31_8b) ;; *) usage >&2; exit 2 ;; esac
case "$ACTION" in prepare|confirmatory|mechanism|heads|sensitivity|resume|status) ;; *) usage >&2; exit 2 ;; esac

CONFIG="configs/cross_model_v1/${MODEL}.yaml"
RUN="${CROSS_MODEL_RUN_DIR:-outputs/${MODEL}_review2}"
if [[ ! -f "$CONFIG" ]]; then echo "Missing config: $CONFIG" >&2; exit 2; fi

if [[ -x .venv/bin/python ]]; then
  PYTHON=(.venv/bin/python -u)
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python -u)
else
  echo "No .venv/bin/python or uv found. Install the repository environment first." >&2
  exit 2
fi

cli() {
  printf '\n==>'
  printf ' %q' "${PYTHON[@]}" -m scripts.cross_model "$@"
  printf '\n'
  PYTHONUNBUFFERED=1 "${PYTHON[@]}" -m scripts.cross_model "$@"
}

score_stage() {
  local stage="$1" dataset="$2" output="$3"
  local resume=()
  if [[ -f "${output}.run.json" ]]; then resume=(--resume); fi
  cli score --config "$CONFIG" --candidates "$RUN/candidates.json" \
    --dataset "$dataset" --output "$output" "${resume[@]}"
}

show_status() {
  echo "Run directory: $RUN"
  echo "Matching processes:"
  pgrep -af 'scripts\.cross_model' 2>/dev/null || echo "No matching process is visible."
  echo "Artifacts:"
  for f in candidates.json development.jsonl development_scores.jsonl gate.jsonl \
           gate_scores.jsonl gate_report.json preflight.json confirmatory.jsonl \
           confirmatory_scores.jsonl confirmatory_analysis.json mechanism.jsonl; do
    if [[ -f "$RUN/$f" ]]; then
      ls -lh "$RUN/$f"
      case "$f" in *.jsonl) wc -l < "$RUN/$f" | awk -v f="$f" '{print "  rows=" $1 " file=" f}' ;; esac
    fi
  done
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
  fi
}

if [[ "$ACTION" == status ]]; then show_status; exit 0; fi

case "$ACTION" in
  prepare)
    mkdir -p "$RUN"
    cli validate --config "$CONFIG" --output "$RUN/candidates.json"

    cli generate --stage development --config "$CONFIG" --candidates "$RUN/candidates.json" \
      --output "$RUN/development.jsonl"
    score_stage development "$RUN/development.jsonl" "$RUN/development_scores.jsonl"
    cli analyze --stage development --config "$CONFIG" --candidates "$RUN/candidates.json" \
      --dataset "$RUN/development.jsonl" --scores "$RUN/development_scores.jsonl" \
      --output "$RUN/development_report.json"

    cli generate --stage frozen_gate --config "$CONFIG" --candidates "$RUN/candidates.json" \
      --development-report "$RUN/development_report.json" --output "$RUN/gate.jsonl"
    score_stage frozen_gate "$RUN/gate.jsonl" "$RUN/gate_scores.jsonl"
    cli analyze --stage frozen_gate --config "$CONFIG" --candidates "$RUN/candidates.json" \
      --dataset "$RUN/gate.jsonl" --scores "$RUN/gate_scores.jsonl" \
      --development-report "$RUN/development_report.json" --output "$RUN/gate_report.json"
    cli preflight --config "$CONFIG" --candidates "$RUN/candidates.json" \
      --gate "$RUN/gate_report.json" --output "$RUN/preflight.json"
    echo "Preparation complete. Review $RUN/gate_report.json and $RUN/preflight.json before confirmatory."
    ;;
  confirmatory)
    [[ -f "$RUN/preflight.json" ]] || { echo "Missing $RUN/preflight.json; run prepare first." >&2; exit 2; }
    if [[ ! -f "$RUN/confirmatory.jsonl" ]]; then
      cli generate --stage confirmatory --config "$CONFIG" --candidates "$RUN/candidates.json" \
        --gate "$RUN/gate_report.json" --preflight "$RUN/preflight.json" \
        --output "$RUN/confirmatory.jsonl"
    fi
    score_stage confirmatory "$RUN/confirmatory.jsonl" "$RUN/confirmatory_scores.jsonl"
    cli analyze --stage confirmatory --config "$CONFIG" --candidates "$RUN/candidates.json" \
      --dataset "$RUN/confirmatory.jsonl" --scores "$RUN/confirmatory_scores.jsonl" \
      --output "$RUN/confirmatory_analysis.json"
    ;;
  mechanism)
    [[ -f "$RUN/confirmatory_analysis.json" ]] || { echo "Missing completed confirmatory analysis; run confirmatory first." >&2; exit 2; }
    if [[ ! -f "$RUN/mechanism_plan.json" ]]; then
      cli mechanism-plan --config "$CONFIG" --candidates "$RUN/candidates.json" \
        --dataset "$RUN/confirmatory.jsonl" --output "$RUN/mechanism_plan.json"
    fi
    if [[ ! -f "$RUN/hook_smoke.json" ]]; then
      cli smoke --config "$CONFIG" --candidates "$RUN/candidates.json" \
        --output "$RUN/hook_smoke.json"
    fi
    local_args=()
    if [[ -f "$RUN/mechanism.jsonl.run.json" ]]; then local_args=(--resume); fi
    cli mechanism --config "$CONFIG" --candidates "$RUN/candidates.json" \
      --dataset "$RUN/confirmatory.jsonl" --scores "$RUN/confirmatory_scores.jsonl" \
      --output "$RUN/mechanism.jsonl" "${local_args[@]}"
    ;;
  heads)
    [[ -f "$RUN/confirmatory_analysis.json" ]] || { echo "Missing completed confirmatory analysis; run confirmatory first." >&2; exit 2; }
    local_args=()
    if [[ -f "$RUN/heads.jsonl.run.json" ]]; then local_args=(--resume); fi
    cli mechanism --heads --config "$CONFIG" --candidates "$RUN/candidates.json" \
      --dataset "$RUN/confirmatory.jsonl" --scores "$RUN/confirmatory_scores.jsonl" \
      --output "$RUN/heads.jsonl" "${local_args[@]}"
    ;;
  sensitivity)
    [[ -f "$RUN/confirmatory_analysis.json" ]] || { echo "Missing completed confirmatory analysis; run confirmatory first." >&2; exit 2; }
    cli sensitivity --config "$CONFIG" --candidates "$RUN/candidates.json" \
      --dataset "$RUN/confirmatory.jsonl" --scores "$RUN/confirmatory_scores.jsonl" \
      --output "$RUN/precision_sensitivity.json"
    ;;
  resume)
    [[ -f "$RUN/candidates.json" ]] || { echo "No revision-2 run found in $RUN." >&2; exit 2; }
    if [[ -f "$RUN/development.jsonl" && ! -f "$RUN/development_report.json" ]]; then
      score_stage development "$RUN/development.jsonl" "$RUN/development_scores.jsonl"
      cli analyze --stage development --config "$CONFIG" --candidates "$RUN/candidates.json" \
        --dataset "$RUN/development.jsonl" --scores "$RUN/development_scores.jsonl" \
        --output "$RUN/development_report.json"
    fi
    if [[ -f "$RUN/gate.jsonl" && ! -f "$RUN/gate_report.json" ]]; then
      [[ -f "$RUN/development_report.json" ]] || { echo "Cannot resume gate without development report." >&2; exit 2; }
      score_stage frozen_gate "$RUN/gate.jsonl" "$RUN/gate_scores.jsonl"
      cli analyze --stage frozen_gate --config "$CONFIG" --candidates "$RUN/candidates.json" \
        --dataset "$RUN/gate.jsonl" --scores "$RUN/gate_scores.jsonl" \
        --development-report "$RUN/development_report.json" --output "$RUN/gate_report.json"
    fi
    if [[ -f "$RUN/gate_report.json" && ! -f "$RUN/preflight.json" ]]; then
      cli preflight --config "$CONFIG" --candidates "$RUN/candidates.json" \
        --gate "$RUN/gate_report.json" --output "$RUN/preflight.json"
    fi
    if [[ -f "$RUN/confirmatory.jsonl" && ! -f "$RUN/confirmatory_analysis.json" ]]; then
      score_stage confirmatory "$RUN/confirmatory.jsonl" "$RUN/confirmatory_scores.jsonl"
      cli analyze --stage confirmatory --config "$CONFIG" --candidates "$RUN/candidates.json" \
        --dataset "$RUN/confirmatory.jsonl" --scores "$RUN/confirmatory_scores.jsonl" \
        --output "$RUN/confirmatory_analysis.json"
    fi
    echo "Resume checks complete. Use status to inspect artifacts and progress."
    ;;
esac
