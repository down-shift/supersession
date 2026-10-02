#!/usr/bin/env bash
set -euo pipefail

# Resume the interrupted Phi/Mistral/Gemma preparation runs without replacing
# their datasets. Gemma's incomplete FP16 scores are archived before a BF16 retry.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON=()
if [[ -x .venv/bin/python ]]; then
  PYTHON=(.venv/bin/python -u)
elif command -v uv >/dev/null 2>&1; then
  PYTHON=(uv run python -u)
else
  echo "No .venv/bin/python or uv found." >&2
  exit 2
fi

PHI_RUN="${PHI4_MINI_RUN_DIR:-outputs/phi4_mini_review2}"
MISTRAL_RUN="${MISTRAL7B_RUN_DIR:-outputs/mistral7b_review2}"
GEMMA_RUN="${GEMMA3_4B_RUN_DIR:-outputs/gemma3_4b_review2}"

recover_existing_run() {
  local model="$1" run="$2" config="$3"

  CROSS_MODEL_RUN_DIR="$run" bash scripts/run_cross_model.sh "$model" resume || return $?

  if [[ ! -f "$run/gate.jsonl" ]]; then
    "${PYTHON[@]}" -m scripts.cross_model generate \
      --stage frozen_gate --config "$config" --candidates "$run/candidates.json" \
      --development-report "$run/development_report.json" --output "$run/gate.jsonl" || return $?
  fi

  CROSS_MODEL_RUN_DIR="$run" bash scripts/run_cross_model.sh "$model" resume || return $?
}

run_status=0

if recover_existing_run phi4_mini "$PHI_RUN" configs/cross_model_v1/phi4_mini.yaml; then
  echo "Phi preparation recovery complete."
else
  echo "Phi preparation recovery failed." >&2
  run_status=1
fi

if recover_existing_run mistral7b "$MISTRAL_RUN" configs/cross_model_v1/mistral7b.yaml; then
  echo "Mistral preparation recovery complete."
else
  echo "Mistral preparation recovery failed." >&2
  run_status=1
fi

recover_gemma() (
  local run="$GEMMA_RUN"
  local config="$run/model_config.yaml"
  local score="$run/development_scores.jsonl"
  local stamp

  [[ -f "$config" && -f "$run/candidates.json" && -f "$run/development.jsonl" ]] || {
    echo "Gemma's pinned config, candidate map, or development dataset is missing in $run." >&2
    return 2
  }

  if [[ -f "$score.provenance.json" ]]; then
    echo "Gemma development score artifact is already sealed; refusing to replace it." >&2
    return 2
  fi
  if [[ -f "$config.fp16-backup" ]]; then
    echo "Gemma FP16 config backup already exists; refusing to overwrite it." >&2
    return 2
  fi

  stamp="$(date -u +%Y%m%dT%H%M%SZ)" || return $?
  if [[ -f "$score" ]]; then
    mv -- "$score" "$score.fp16-partial-$stamp" || return $?
  fi
  if [[ -f "$score.run.json" ]]; then
    mv -- "$score.run.json" "$score.fp16-partial-$stamp.run.json" || return $?
  fi
  cp -- "$config" "$config.fp16-backup" || return $?
  "${PYTHON[@]}" - "$config" <<'PY' || return $?
import sys
from pathlib import Path
import yaml

path = Path(sys.argv[1])
config = yaml.safe_load(path.read_text())
config['model']['dtype'] = 'bfloat16'
path.write_text(yaml.safe_dump(config, sort_keys=False))
PY

  CROSS_MODEL_RUN_DIR="$run" bash scripts/run_cross_model.sh gemma3_4b resume || return $?

  if [[ ! -f "$run/gate.jsonl" ]]; then
    "${PYTHON[@]}" -m scripts.cross_model generate \
      --stage frozen_gate --config "$config" --candidates "$run/candidates.json" \
      --development-report "$run/development_report.json" --output "$run/gate.jsonl" || return $?
  fi

  CROSS_MODEL_RUN_DIR="$run" bash scripts/run_cross_model.sh gemma3_4b resume || return $?
)

if recover_gemma; then
  echo "Gemma preparation recovery complete."
else
  echo "Gemma preparation recovery failed." >&2
  run_status=1
fi

exit "$run_status"
