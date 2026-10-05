#!/usr/bin/env bash
# Idempotent environment bootstrap for the autonomous loop (macOS laptop / Linux
# experiment machine). Safe to re-run. See docs/AUTONOMY.md.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(git rev-parse --show-toplevel)"

# 1. uv (package/venv manager)
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2. project environment: model stack only where a CUDA GPU is present
if command -v nvidia-smi >/dev/null 2>&1; then
  uv sync --extra model --extra dev || { echo "WARN: model extra failed; falling back to dev"; uv sync --extra dev; }
else
  uv sync --extra dev
fi

# 3. soft prerequisites (warn, don't fail)
if command -v gh >/dev/null 2>&1; then
  gh auth status >/dev/null 2>&1 || echo "WARN: gh not authenticated — run 'gh auth login' for PRs/merges"
else
  echo "WARN: gh CLI missing — PRs/merges unavailable"
fi
command -v latexmk >/dev/null 2>&1 || echo "WARN: latexmk missing — install TeX Live/MacTeX to build paper/ (writing steps need it)"
if [ -d outputs ]; then
  echo "artifacts: outputs/ present — [ARTIFACTS] steps can run here"
else
  echo "NOTE: no outputs/ on this machine — [ARTIFACTS] steps (number audits against raw analyses) must run on the experiment machine"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd', ' -)"
else
  echo "NOTE: no GPU — [GPU] steps must run on the experiment machine"
fi

echo "bootstrap: OK"
