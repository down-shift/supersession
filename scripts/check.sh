#!/usr/bin/env bash
# The mandatory local gate (see docs/AUTONOMY.md). Must pass before any commit / merge.
#   1. the full test suite (torch-dependent tests skip themselves on CPU-only machines)
#   2. the paper and supplement build with no LaTeX errors and no undefined references/citations
# CI (.github/workflows/gate.yml) runs both on every PR.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(git rev-parse --show-toplevel)"

uv run --extra dev pytest -q   # torch-dependent tests skip themselves on CPU-only machines (E0.2)

if command -v latexmk >/dev/null 2>&1; then
  bash scripts/build_paper.sh
else
  echo "WARN: latexmk missing — paper build skipped locally (CI still builds it)"
fi

echo "gate: OK"
