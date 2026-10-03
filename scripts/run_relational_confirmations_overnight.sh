#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

REV=factorial_relation_counterbalanced_geometryfix_exclusions_20261003
BASE="outputs/cross_model_relational_v2/$REV"
CFG_DIR=configs/cross_model_relational_v2_geometryfix_exclusions
LOG_DIR="$BASE/overnight_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/confirmations-$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Repository: $(pwd)"
echo "Log: $LOG_FILE"
echo "Inference is limited to Qwen and Gemma confirmation scoring."

run_model() {
  local model="$1" python_version="$2"
  (
    set -euo pipefail
    local run="$BASE/$model"
    local config="$CFG_DIR/$model.yaml"
    local gate="$run/migrated/gate_report.json"
    local expected_runtime="$run/migrated/gate.jsonl.scores.jsonl.provenance.json"
    local dataset="$run/confirmatory.jsonl"
    local scores="$run/confirmatory_scores.jsonl"
    local analysis="$run/confirmatory_analysis.json"

    echo "===== $model: prepare and score confirmation ====="
    for required in "$config" "$run/candidates.json" "$gate" "$run/migrated/preflight.json" "$expected_runtime"; do
      [[ -f "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 1; }
    done

    # The current .venv is the verified Qwen/Python 3.13.5 environment;
    # keep Gemma's Python 3.12.13 environment separate so switching cannot replace it.
    if [[ "$model" == qwen3_8b ]]; then
      export UV_PROJECT_ENVIRONMENT=.venv
    else
      export UV_PROJECT_ENVIRONMENT=.venv-gemma312
    fi
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    uv sync --frozen --extra model --python "$python_version"

    export MODEL_CONFIG="$config" EXPECTED_RUNTIME="$expected_runtime"
    uv run --frozen --extra model --python "$python_version" python - <<'PY'
import json, os, torch
from src.utils import load_config, provenance

expected = json.load(open(os.environ["EXPECTED_RUNTIME"]))["provenance"]
current = provenance(load_config(os.environ["MODEL_CONFIG"]), None)
assert (current["python"], current["packages"]) == (expected["python"], expected["packages"]), (
    f"Runtime mismatch: Python {current['python']} vs {expected['python']}; "
    f"packages current={current['packages']} expected={expected['packages']}"
)
assert torch.cuda.is_available(), "CUDA unavailable"
print(f"Runtime and CUDA match (Python {current['python']})")
PY

    export MODEL="$model" MODEL_CONFIG="$config" CANDIDATES="$run/candidates.json"
    export DATASET="$dataset" GATE="$gate" PREFLIGHT="$run/migrated/preflight.json"

    if [[ ! -f "$dataset" ]]; then
      # Generation audits the full tokenizer geometry and validates gate,
      # preflight, lineage, allocation and shared-history constraints.
      uv run --frozen --extra model --python "$python_version" python -m scripts.robustness_v2 generate \
        --stage confirmatory --config "$config" --candidates "$run/candidates.json" \
        --gate "$gate" --preflight "$run/migrated/preflight.json" \
        --output "$dataset" --local-files-only
    else
      echo "Keeping existing confirmation dataset: $dataset"
    fi

    uv run --frozen --extra model --python "$python_version" python - <<'PY'
import os
from src.utils import load_config
from src.cross_model import robustness_protocol as design
from src.cross_model.workflow import verify_confirmation

config_path = os.environ["MODEL_CONFIG"]
config = load_config(config_path)
rows, _ = verify_confirmation(os.environ["DATASET"], config, config_path,
                              os.environ["CANDIDATES"], design=design)
assert len(rows) == 18432, f"Expected 18432 confirmation members; found {len(rows)}"
print(f"Confirmation lineage and geometry validated: {len(rows)} members")
PY

    if [[ -f "$scores.provenance.json" ]]; then
      echo "Validating already completed scores: $scores"
      export SCORES="$scores"
      uv run --frozen --extra model --python "$python_version" python - <<'PY'
import os
from src.utils import load_config
from src.cross_model import robustness_protocol as design
from src.cross_model.workflow import score_info
from src.cross_model.score_checks import checked_scores

config_path = os.environ["MODEL_CONFIG"]
rows, scores, _, _ = score_info(os.environ["SCORES"], os.environ["DATASET"],
    load_config(config_path), config_path, os.environ["CANDIDATES"],
    "confirmatory", design=design)
checked_scores(rows, scores, require_surfaces=True)
assert len(scores) == 18432, f"Expected 18432 scores; found {len(scores)}"
print(f"Saved scores validate: {len(scores)} members")
PY
    else
      local resume_args=()
      if [[ -f "$scores.run.json" ]]; then
        resume_args=(--resume)
        echo "Resuming saved score progress."
      elif [[ -e "$scores" ]]; then
        echo "Score file exists without a resume marker or completion provenance; preserving it and stopping." >&2
        exit 1
      fi
      uv run --frozen --extra model --python "$python_version" python -m scripts.robustness_v2 score \
        --config "$config" --candidates "$run/candidates.json" \
        --dataset "$dataset" --output "$scores" "${resume_args[@]}"
    fi

    if [[ ! -f "$analysis" ]]; then
      uv run --frozen --extra model --python "$python_version" python -m scripts.robustness_v2 analyze \
        --stage confirmatory --config "$config" --candidates "$run/candidates.json" \
        --dataset "$dataset" --scores "$scores" --output "$analysis"
    else
      export ANALYSIS="$analysis"
      uv run --frozen --extra model --python "$python_version" python - <<'PY'
import os
from src.cross_model.protocol import read_sealed
from src.data.io import sha256_file

report = read_sealed(os.environ["ANALYSIS"])
assert report["stage"] == "confirmatory"
assert report["dataset_sha256"] == sha256_file(os.environ["DATASET"])
assert report["scores_sha256"] == sha256_file(os.environ["SCORES"])
print("Existing analysis matches current dataset and scores")
PY
    fi
    echo "===== $model: confirmation complete ====="
  )
}

preflight_model() {
  local model="$1" python_version="$2"
  (
    set -euo pipefail
    local run="$BASE/$model"
    local config="$CFG_DIR/$model.yaml"
    local gate="$run/migrated/gate_report.json"
    local expected_runtime="$run/migrated/gate.jsonl.scores.jsonl.provenance.json"
    local dataset="$run/confirmatory.jsonl"

    echo "===== $model: preflight before any confirmation scoring ====="
    for required in "$config" "$run/candidates.json" "$gate" "$run/migrated/preflight.json" "$expected_runtime"; do
      [[ -f "$required" ]] || { echo "Missing required artifact: $required" >&2; exit 1; }
    done
    if [[ "$model" == qwen3_8b ]]; then
      export UV_PROJECT_ENVIRONMENT=.venv
    else
      export UV_PROJECT_ENVIRONMENT=.venv-gemma312
    fi
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    uv sync --frozen --extra model --python "$python_version"
    export MODEL_CONFIG="$config" EXPECTED_RUNTIME="$expected_runtime"
    uv run --frozen --extra model --python "$python_version" python - <<'PY'
import json, os, torch
from src.utils import load_config, provenance

expected = json.load(open(os.environ["EXPECTED_RUNTIME"]))["provenance"]
current = provenance(load_config(os.environ["MODEL_CONFIG"]), None)
assert (current["python"], current["packages"]) == (expected["python"], expected["packages"]), (
    f"Runtime mismatch: Python {current['python']} vs {expected['python']}; "
    f"packages current={current['packages']} expected={expected['packages']}"
)
assert torch.cuda.is_available(), "CUDA unavailable"
print(f"Runtime and CUDA match (Python {current['python']})")
PY

    export MODEL_CONFIG="$config" CANDIDATES="$run/candidates.json"
    export DATASET="$dataset" GATE="$gate" PREFLIGHT="$run/migrated/preflight.json"
    if [[ ! -f "$dataset" ]]; then
      uv run --frozen --extra model --python "$python_version" python -m scripts.robustness_v2 generate \
        --stage confirmatory --config "$config" --candidates "$run/candidates.json" \
        --gate "$gate" --preflight "$run/migrated/preflight.json" \
        --output "$dataset" --local-files-only
    fi
    uv run --frozen --extra model --python "$python_version" python - <<'PY'
import os
from src.utils import load_config
from src.cross_model import robustness_protocol as design
from src.cross_model.workflow import verify_confirmation

config_path = os.environ["MODEL_CONFIG"]
rows, _ = verify_confirmation(os.environ["DATASET"], load_config(config_path), config_path,
                              os.environ["CANDIDATES"], design=design)
assert len(rows) == 18432, f"Expected 18432 confirmation members; found {len(rows)}"
print(f"Confirmation lineage and geometry validated: {len(rows)} members")
PY

    # Load the pinned checkpoint once before starting the long run. No forward
    # pass is run here; this checks local weights, model loading and placement.
    uv run --frozen --extra model --python "$python_version" python - <<'PY'
import os
from src.utils import load_config
from src.cross_model.runtime import load_pinned_model

model, tokenizer = load_pinned_model(load_config(os.environ["MODEL_CONFIG"]))
print(f"Checkpoint load passed: {model.__class__.__name__}")
del model, tokenizer
PY
    echo "===== $model: preflight passed ====="
  )
}

# Preflight both models before starting either long confirmation run.
set +e
preflight_model qwen3_8b 3.13.5
QWEN_PREFLIGHT_STATUS=$?
preflight_model gemma3_4b 3.12.13
GEMMA_PREFLIGHT_STATUS=$?
set -e
echo "Qwen preflight exit status: $QWEN_PREFLIGHT_STATUS"
echo "Gemma preflight exit status: $GEMMA_PREFLIGHT_STATUS"
if (( QWEN_PREFLIGHT_STATUS != 0 || GEMMA_PREFLIGHT_STATUS != 0 )); then
  echo "Preflight failed; no confirmation scoring was started. See $LOG_FILE" >&2
  exit 1
fi

# One scoring failure must not prevent the other eligible model from running.
set +e
run_model qwen3_8b 3.13.5
QWEN_STATUS=$?
run_model gemma3_4b 3.12.13
GEMMA_STATUS=$?
set -e

echo "Qwen exit status: $QWEN_STATUS"
echo "Gemma exit status: $GEMMA_STATUS"
echo "Phi remains excluded because its frozen gate failed."
echo "Run log: $LOG_FILE"
if (( QWEN_STATUS != 0 || GEMMA_STATUS != 0 )); then
  exit 1
fi
