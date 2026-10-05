#!/usr/bin/env bash
# Build paper/main.pdf and fail on LaTeX errors or unresolved references/citations.
set -euo pipefail
cd "$(dirname "$0")/../paper"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 || {
  echo "paper: LaTeX build failed — see paper/main.log"; grep -A3 '^!' main.log | head -20; exit 1; }
if grep -E "undefined (references|citations)|Reference .* undefined|Citation .* undefined" main.log; then
  echo "paper: unresolved references or citations"; exit 1
fi
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex >/dev/null 2>&1 || { echo "paper: supplement build failed"; exit 1; }
echo "paper: OK ($(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/ {print $2}') pages)"
