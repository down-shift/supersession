#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
OUT="${OUT:-outputs/four_query_288}"; CFG="${CFG:-configs/four_query_288.yaml}"
DISCOVERY_N="${DISCOVERY_N:-24}"; ALL_POSITION_N="${ALL_POSITION_N:-12}"; HELDOUT_N="${HELDOUT_N:-96}"
for f in "$OUT/pairs.jsonl" "$OUT/pair_behavior.jsonl" "$OUT/frozen_token_ids.json" "$OUT/prompt_selection.json" "$OUT/gate.json" "$OUT/analysis/four_query_summary.json"; do
  [[ -s "$f" ]] || { echo "Required frozen behavior artifact missing: $f" >&2; exit 2; }
done
uv run python - "$OUT" "$CFG" <<'PY'
import json,sys,yaml
from pathlib import Path
out=Path(sys.argv[1]); cfg=yaml.safe_load(open(sys.argv[2])); sel=json.load(open(out/'prompt_selection.json')); gate=json.load(open(out/'gate.json'))
if sel.get('selected_variant')!='initial_update': raise SystemExit('frozen prompt is not initial_update')
if gate.get('pass') is not True: raise SystemExit('behavior gate did not pass')
if not json.load(open(out/'analysis/four_query_summary.json')): raise SystemExit('behavior summary is empty')
if cfg['model'].get('revision')!='b968826d9c46dd6066d109eabc6255188de91218': raise SystemExit('config must pin the frozen model revision')
if sel.get('model_revision') not in (None,cfg['model']['revision']): raise SystemExit('prompt-selection model revision differs from frozen mechanism revision')
rows=[json.loads(x) for x in open(out/'pairs.jsonl') if x.strip()]
if not rows or any(r.get('prompt_variant')!='initial_update' for r in rows): raise SystemExit('pairs do not consistently use initial_update')
if rows[0].get('partition')!='confirmatory': raise SystemExit('pair dataset is not confirmatory')
tokens=json.load(open(out/'frozen_token_ids.json'))['token_ids']
if tokens!=sel.get('token_ids'): raise SystemExit('candidate token map differs from prompt selection artifact')
PY
uv sync --locked --extra model --extra dev
COMMON=(--config "$CFG" --pairs "$OUT/pairs.jsonl" --token-ids "$OUT/frozen_token_ids.json")
mkdir -p "$OUT/mechanism"
DISC="$OUT/mechanism/discovery.jsonl"; args=("${COMMON[@]}" --output "$DISC" --stage discovery --n-histories "$DISCOVERY_N" --cell-set focal); [[ -f "$DISC.run.json" ]] && args+=(--resume)
uv run python scripts/run_four_query_patching.py "${args[@]}"
uv run python scripts/analyze_four_query_patching.py --patches "$DISC" --output-dir "$OUT/mechanism/discovery" --stage discovery
LAYERS="$(uv run python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["selection_statistic"]["selected_layers"])))' "$OUT/mechanism/discovery/discovery_selection.json")"
echo "Frozen discovery layers: $LAYERS"
if [[ "${RUN_ALL_POSITIONS:-0}" == 1 ]]; then
  ALL="$OUT/mechanism/all_positions.jsonl"; args=("${COMMON[@]}" --output "$ALL" --stage discovery --n-histories "$ALL_POSITION_N" --all-positions --cell-set focal); [[ -f "$ALL.run.json" ]] && args+=(--resume)
  uv run python scripts/run_four_query_patching.py "${args[@]}"
  uv run python scripts/analyze_four_query_patching.py --patches "$ALL" --output-dir "$OUT/mechanism/all_positions" --stage discovery
fi
HELD="$OUT/mechanism/heldout.jsonl"; args=("${COMMON[@]}" --output "$HELD" --stage heldout --n-histories "$HELDOUT_N" --layers "$LAYERS" --cell-set focal); [[ -f "$HELD.run.json" ]] && args+=(--resume)
uv run python scripts/run_four_query_patching.py "${args[@]}"
uv run python scripts/analyze_four_query_patching.py --patches "$HELD" --output-dir "$OUT/mechanism/heldout" --stage heldout
