#!/usr/bin/env bash
# Create a new runtime-fix freeze, revalidate and migrate saved Qwen/Gemma
# competence stages, then launch the existing confirmation workflow.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

REV="${RELATIONAL_REVISION:-factorial_relation_counterbalanced_runtimefix_20261004}"
BASE="outputs/cross_model_relational_v2/$REV"
OLD="${RELATIONAL_SOURCE_REVISION:-outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003}"
CFG="${RELATIONAL_CONFIG_DIR:-configs/cross_model_relational_v2_geometryfix_exclusions}"

for tool in uv python3 bash tee; do
  command -v "$tool" >/dev/null || { echo "Required Ubuntu command is missing: $tool" >&2; exit 1; }
done
export PYTHONUNBUFFERED=1

if [[ -e "$BASE" ]]; then
  echo "Refusing to reuse existing output path: $BASE" >&2
  echo "Choose a new RELATIONAL_REVISION after reviewing the existing artifacts." >&2
  exit 1
fi

for model in qwen3_8b gemma3_4b; do
  for artifact in candidates.json development.jsonl development_scores.jsonl \
                  development.jsonl.provenance.json development_scores.jsonl.provenance.json \
                  gate.jsonl gate_scores.jsonl gate.jsonl.provenance.json gate_scores.jsonl.provenance.json; do
    if [[ ! -f "$OLD/$model/$artifact" ]]; then
      echo "Missing required source artifact: $OLD/$model/$artifact" >&2
      exit 1
    fi
  done
done

for model in qwen3_8b gemma3_4b phi4_mini; do
  # The freeze checks v1 lineage for all three configured models; Phi is not scored.
  for artifact in confirmatory_scores.jsonl confirmatory_scores.jsonl.provenance.json candidates.json; do
    [[ -f "outputs/${model}_review2/$artifact" ]] || {
      echo "Missing v1 lineage artifact: outputs/${model}_review2/$artifact" >&2; exit 1;
    }
  done
  [[ -f "$CFG/$model.yaml" ]] || { echo "Missing model config: $CFG/$model.yaml" >&2; exit 1; }
done

mkdir -p "$(dirname "$BASE")"
mkdir "$BASE"
exec > >(tee "$BASE/preparation.log") 2>&1
echo "New implementation revision: $REV"
echo "Source artifacts: $OLD"
echo "Model scoring will start only after both models pass preparation and runner preflight."

# This tool environment is separate from both model inference environments.
export UV_PROJECT_ENVIRONMENT=.venv-runtimefix-tools
uv sync --frozen --extra model
uv run --frozen --extra model python -m scripts.robustness_v2 audit \
  --stage development --design-only --output "$BASE/prompt_audit.json"
uv run --frozen --extra model python -m scripts.robustness_v2 freeze \
  --audit "$BASE/prompt_audit.json" --output "$BASE/protocol_freeze.json"

inference_python() {
  PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python3 - "$1" <<'PY'
import json, sys
from src.cross_model.runtime_provenance import resolve_inference_provenance

with open(sys.argv[1], encoding="utf8") as handle:
    print(resolve_inference_provenance(json.load(handle))["python"])
PY
}

prepare_model() {
  local model="$1"
  local source="$OLD/$model"
  local run="$BASE/$model"
  local config="$CFG/$model.yaml"
  local v1_provenance="outputs/${model}_review2/confirmatory_scores.jsonl.provenance.json"
  local python_version env

  python_version="$(inference_python "$source/gate_scores.jsonl.provenance.json")"
  if [[ "$model" == qwen3_8b ]]; then
    env="${RELATIONAL_QWEN_ENV:-.venv-qwen-runtimefix}"
  else
    env="${RELATIONAL_GEMMA_ENV:-.venv-gemma-runtimefix}"
  fi
  export UV_PROJECT_ENVIRONMENT="$env"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

  echo "===== $model: preparing with original inference Python $python_version ====="
  uv sync --frozen --extra model --python "$python_version"

  # Check the scoring environment before any tokenizer migration work.
  export EXPECTED_RUNTIME="$source/gate_scores.jsonl.provenance.json" MODEL_CONFIG="$config"
  uv run --frozen --extra model --python "$python_version" python - <<'PY'
import json, os
from src.utils import load_config, provenance
from src.cross_model.runtime_provenance import resolve_inference_provenance, runtime_mismatches

with open(os.environ["EXPECTED_RUNTIME"], encoding="utf8") as handle:
    expected = resolve_inference_provenance(json.load(handle))
mismatches = runtime_mismatches(expected, provenance(load_config(os.environ["MODEL_CONFIG"]), None))
if mismatches:
    raise SystemExit("Runtime mismatch with original inference environment: " + "; ".join(mismatches))
print("Original inference Python and package versions match")
PY

  uv run --frozen --extra model --python "$python_version" \
    python -m scripts.robustness_v2 validate \
    --config "$config" --freeze "$BASE/protocol_freeze.json" \
    --v1-provenance "$v1_provenance" \
    --output "$run/candidates.json" --local-files-only

  uv run --frozen --extra model --python "$python_version" \
    python -m scripts.robustness_v2 migrate --stage development \
    --config "$config" --candidates "$run/candidates.json" \
    --original-candidates "$source/candidates.json" \
    --dataset "$source/development.jsonl" \
    --scores "$source/development_scores.jsonl" \
    --output "$run/migrated/development.jsonl" --local-files-only

  uv run --frozen --extra model --python "$python_version" \
    python -m scripts.robustness_v2 analyze --stage development \
    --config "$config" --candidates "$run/candidates.json" \
    --dataset "$run/migrated/development.jsonl" \
    --scores "$run/migrated/development.jsonl.scores.jsonl" \
    --output "$run/migrated/development_report.json"

  uv run --frozen --extra model --python "$python_version" \
    python -m scripts.robustness_v2 migrate --stage frozen_gate \
    --config "$config" --candidates "$run/candidates.json" \
    --original-candidates "$source/candidates.json" \
    --dataset "$source/gate.jsonl" --scores "$source/gate_scores.jsonl" \
    --prior-dataset "$run/migrated/development.jsonl" \
    --development-report "$run/migrated/development_report.json" \
    --output "$run/migrated/gate.jsonl" --local-files-only

  uv run --frozen --extra model --python "$python_version" \
    python -m scripts.robustness_v2 analyze --stage frozen_gate \
    --config "$config" --candidates "$run/candidates.json" \
    --dataset "$run/migrated/gate.jsonl" \
    --scores "$run/migrated/gate.jsonl.scores.jsonl" \
    --development-report "$run/migrated/development_report.json" \
    --output "$run/migrated/gate_report.json"

  uv run --frozen --extra model --python "$python_version" \
    python -m scripts.robustness_v2 preflight \
    --config "$config" --candidates "$run/candidates.json" \
    --gate "$run/migrated/gate_report.json" \
    --output "$run/migrated/preflight.json"
}

prepare_model qwen3_8b
prepare_model gemma3_4b

echo "Both model gates and preflights passed. Starting confirmation runner."
RELATIONAL_REVISION="$REV" RELATIONAL_CONFIG_DIR="$CFG" \
  bash scripts/run_relational_confirmations_overnight.sh
