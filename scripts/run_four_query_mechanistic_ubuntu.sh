#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
OUT="${OUT:-outputs/four_query_288}"; CFG="${CFG:-configs/four_query_288.yaml}"
DISCOVERY_N="${DISCOVERY_N:-24}"; ALL_POSITION_N="${ALL_POSITION_N:-12}"; HELDOUT_N="${HELDOUT_N:-96}"
RUN_HELDOUT="${RUN_HELDOUT:-0}"; PREFIX_AUDIT_N="${PREFIX_AUDIT_N:-2}"; POSITION_BATCH_SIZE="${POSITION_BATCH_SIZE:-16}"
RUN_TRAJECTORY_ANALYSIS="${RUN_TRAJECTORY_ANALYSIS:-0}"; RUN_ALL_POSITIONS="${RUN_ALL_POSITIONS:-0}"
ALL_POSITION_VERSION="${ALL_POSITION_VERSION:-v2}"
if [[ "$RUN_ALL_POSITIONS" == 1 ]]; then
  [[ "$ALL_POSITION_N" =~ ^[1-9][0-9]*$ && "$ALL_POSITION_N" -le 24 ]] || { echo "ALL_POSITION_N must be between 1 and 24 (the frozen stage-1 discovery set)" >&2; exit 2; }
  [[ "$RUN_HELDOUT" != 1 ]] || { echo "RUN_ALL_POSITIONS cannot be combined with RUN_HELDOUT; held-out confirmation is frozen" >&2; exit 2; }
fi
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
if [[ "$RUN_TRAJECTORY_ANALYSIS" == 1 ]]; then
  SRC="$OUT/mechanism/discovery.jsonl"; [[ -s "$SRC" ]] || SRC="$OUT/mechanism/discovery/patch_Rx_by_layer_site.csv"
  [[ -s "$SRC" ]] || { echo "No existing targeted discovery data found for trajectory analysis" >&2; exit 2; }
  uv run python scripts/analyze_mechanistic_trajectory.py --patches "$SRC" --output-dir "$OUT/mechanism/discovery"
  echo "Trajectory artifacts: $OUT/mechanism/discovery/trajectory_by_layer_site.csv $OUT/mechanism/discovery/trajectory_summary.json $OUT/mechanism/discovery/Rx_by_layer_site.png"
  exit 0
fi
DISC="$OUT/mechanism/discovery.jsonl"
if [[ "$RUN_ALL_POSITIONS" == 1 ]]; then
  [[ -s "$OUT/mechanism/discovery/patch_Rx_by_layer_site.csv" && -s "$OUT/mechanism/discovery/discovery_selection.json" ]] || { echo "All-position sweep requires the existing frozen targeted discovery analysis and selection; refusing to rerun discovery." >&2; exit 2; }
  echo "Using frozen targeted discovery analysis; skipping targeted patch rerun."
elif [[ -s "$OUT/mechanism/discovery/patch_Rx_by_layer_site.csv" && -s "$OUT/mechanism/discovery/discovery_selection.json" ]]; then
  echo "Using frozen targeted discovery analysis; skipping targeted patch rerun."
else
  args=("${COMMON[@]}" --output "$DISC" --stage discovery --n-histories "$DISCOVERY_N" --cell-set focal --audit-prefix-invariance "$PREFIX_AUDIT_N" --position-batch-size "$POSITION_BATCH_SIZE"); [[ -f "$DISC.run.json" ]] && args+=(--resume)
  uv run python scripts/run_four_query_patching.py "${args[@]}"
  uv run python scripts/analyze_four_query_patching.py --patches "$DISC" --output-dir "$OUT/mechanism/discovery" --stage discovery
fi
LAYERS="$(uv run python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["selection_statistic"]["selected_layers"])))' "$OUT/mechanism/discovery/discovery_selection.json")"
echo "Frozen discovery layers: $LAYERS"
if [[ "${RUN_ALL_POSITIONS:-0}" == 1 ]]; then
  ALLDIR="$OUT/mechanism/all_positions_${ALL_POSITION_VERSION}"; mkdir -p "$ALLDIR"; ALL="$ALLDIR/patches.jsonl"; args=("${COMMON[@]}" --output "$ALL" --stage discovery --n-histories "$ALL_POSITION_N" --all-positions --cell-set focal --position-batch-size "$POSITION_BATCH_SIZE"); [[ -f "$ALL.run.json" ]] && args+=(--resume)
  uv run python scripts/run_four_query_patching.py "${args[@]}"
  uv run python scripts/analyze_four_query_patching.py --patches "$ALL" --output-dir "$ALLDIR/analysis" --stage discovery
  echo "All-position artifacts: $ALL $ALLDIR/analysis/all_positions_Rx.csv $ALLDIR/analysis/all_positions_Rx_by_history.csv $ALLDIR/analysis/all_positions_Rx_semantic.csv $ALLDIR/analysis/all_positions_Rx.png $ALLDIR/analysis/all_positions_Rx_semantic.png $ALLDIR/analysis/all_positions_token_legend.csv"
fi
if [[ "$RUN_HELDOUT" != 1 ]]; then
echo "Stopped after discovery. Review $OUT/mechanism/discovery; set RUN_HELDOUT=1 to run held-out confirmation with the frozen selected layers."
  exit 0
fi
HELD="$OUT/mechanism/heldout.jsonl"; args=("${COMMON[@]}" --output "$HELD" --stage heldout --n-histories "$HELDOUT_N" --layers "$LAYERS" --cell-set focal --position-batch-size "$POSITION_BATCH_SIZE"); [[ -f "$HELD.run.json" ]] && args+=(--resume)
uv run python scripts/run_four_query_patching.py "${args[@]}"
uv run python scripts/analyze_four_query_patching.py --patches "$HELD" --output-dir "$OUT/mechanism/heldout" --stage heldout
uv run python scripts/analyze_heldout_correct_answer_decomposition.py --diagnostics "$OUT/mechanism/heldout/heldout_correct_answer_diagnostics_by_history.csv" --output-dir "$OUT/mechanism/heldout"
uv run python scripts/freeze_mechanistic_partitions.py --pairs "$OUT/pairs.jsonl" --discovery "$OUT/mechanism/discovery/patch_Rx_by_layer_site.csv" --heldout "$OUT/mechanism/heldout/heldout_R_by_history.csv" --output "$OUT/mechanism/mechanistic_partitions.json"
