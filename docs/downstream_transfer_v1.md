# downstream_transfer_v1

This isolated experiment tests whether changing an obsolete entity binding transfers causal preference to that value's derived code, while the current binding, query, codebook, and correct current code stay fixed. A paired irrelevant-entity query removes effects caused merely by introducing the replacement value/code. The primary claim is about downstream causal influence, not behavioral failure or forgetting.

## Frozen design

- Histories use Nora/Owen badge updates and always ask current_x and current_z.
- Values and opaque codes are frozen in configs/downstream_transfer_values.json and configs/downstream_transfer_codes.json. Each generated history independently shuffles a one-to-one value→code mapping with the stage PRNG; both pair members share it. Code labels are never chosen from model outputs.
- Every history has two obsolete and two live bindings. Each is edited against both current queries in matched baseline/edit members.
- E(b,q) = [log P(C_r|edit,q)-log P(C_s|edit,q)] - [log P(C_r|base,q)-log P(C_s|base,q)]. Probabilities are complete-sequence continuation log probabilities.
- R_stale_derived = 0.5 * (E(old_x,current_x)-E(old_x,current_z)+E(old_z,current_z)-E(old_z,current_x)).
- R_live_derived uses the same contrast for current_x/current_z edits and is a positive control.
- Primary analysis includes every history. Current-answer stability is reported for all stale pairs; correct-only filtering is not used.

## Stages and sizes

- Development: 24 fresh histories, 48 current-code competence records. No causal output.
- Frozen gate: 24 fresh histories, 48 records. Pass at ≥97% pooled current-code accuracy (47/48); query and x/z cells are diagnostic only.
- Confirmatory: 96 fresh histories, 1,536 matched scoring records (4 edits × 2 queries × 2 directions per history).
- Confirmatory summaries use history as the unit, with mean, median, 10% trimmed mean, positive fraction, deterministic history bootstrap 95% CI, and paired sign-flip permutation p-value.

All stages use fixed seeds in src/data/downstream_transfer.py; generator also accepts prior datasets and rejects reused concrete binding histories. Generated datasets and score files are create-only. Resume requires the original checkpoint fingerprint. The sealed gate binds dataset, scoring/template, config, value/code vocabularies, tokenizer/model revisions, and scoring definition.

## v4 completed result

- Frozen gate: passed with complete-sequence candidate ranking at 48/48 (100%). Greedy unrestricted generation parsed as an exact code label at 0/48; outputs were Markdown-bold labels, and stripping the markers gives the correct code in 48/48. This formatting diagnostic did not determine gate eligibility.
- Confirmation: 96 histories, 1,536 matched records, 16 candidates per record. Token audit found all 16 codes were stable two-token continuations across 64 prompts.
- Primary `R_stale_derived`: mean 5.544 nats, median 5.182, 10% trimmed mean 5.441, positive in 96/96 histories, history-bootstrap 95% CI [5.159, 5.941], 10,000-draw sign-flip p = 0.00010 (Monte Carlo resolution floor).
- Positive control `R_live_derived`: mean 12.836 nats, median 12.945, 10% trimmed mean 12.835, positive in 96/96 histories, 95% CI [12.418, 13.257], p = 0.00010 (resolution floor).
- Mean stale E was 9.267 nats on the matching-entity query and 3.723 nats on the other-entity query. The primary contrast subtracts the latter generic replacement/context effect.
- Across 384 stale baseline/edit pairs, current-code candidate accuracy was 360/384 (93.75%) at baseline and 364/384 (94.79%) after editing. Nine pairs changed from incorrect to correct and five from correct to incorrect. The history-mean paired change in correct-code log probability was +0.108 nats (95% CI [-0.050, 0.264]); the paired margin change was +0.103 ([-0.118, 0.325]). These intervals do not establish zero interference.
- Accuracy was lower for `current_z` than `current_x` in confirmation (baseline 172/192 vs. 188/192); report this diagnostic asymmetry. Confirmatory accuracy is candidate-sequence ranking, not unrestricted text generation.

The supported result is that, for Qwen3-8B under this protocol, changing a superseded binding causally shifted preference toward its derived code more for the associated entity's current query than for the other entity's query, while the correct current code remained unchanged. This does not show stale edits caused current-answer errors, and the confirmatory run did not measure unrestricted generation. Detailed outputs are under `outputs/downstream_transfer_v1/analysis_v4/`.

## Launch

Use the repository uv environment (`uv sync --extra model --extra dev` on the experiment machine). All commands below run from the repository root.

These commands document the frozen clean-run protocol. The v4 run is complete and its outputs are create-only; do not rerun these paths. See the v4 result above and artifacts under `outputs/downstream_transfer_v1/`.

First perform the tokenizer-only audit. It loads the frozen tokenizer, never the model:

```sh
uv run python scripts/audit_downstream_transfer_tokens.py --output outputs/downstream_transfer_v1/token_audit.json
```

Then generate and score development:

```sh
uv run python scripts/generate_downstream_transfer.py --stage development --output outputs/downstream_transfer_v1/development.jsonl
uv run python scripts/run_downstream_transfer.py --dataset outputs/downstream_transfer_v1/development.jsonl --token-audit outputs/downstream_transfer_v1/token_audit.json --output outputs/downstream_transfer_v1/development_scores.jsonl
```

Generate and score the frozen gate only after development:

```sh
uv run python scripts/generate_downstream_transfer.py --stage frozen_gate --prior-dataset outputs/downstream_transfer_v1/development.jsonl --output outputs/downstream_transfer_v1/gate.jsonl
uv run python scripts/run_downstream_transfer.py --dataset outputs/downstream_transfer_v1/gate.jsonl --token-audit outputs/downstream_transfer_v1/token_audit.json --output outputs/downstream_transfer_v1/gate_scores.jsonl
```

After the sealed gate passes, generate/score confirmation and analyze:

```sh
uv run python scripts/generate_downstream_transfer.py --stage confirmatory --prior-dataset outputs/downstream_transfer_v1/development.jsonl --token-audit outputs/downstream_transfer_v1/token_audit.json --gate outputs/downstream_transfer_v1/gate_scores.jsonl.gate.json --output outputs/downstream_transfer_v1/confirmatory.jsonl
uv run python scripts/run_downstream_transfer.py --dataset outputs/downstream_transfer_v1/confirmatory.jsonl --token-audit outputs/downstream_transfer_v1/token_audit.json --gate outputs/downstream_transfer_v1/gate_scores.jsonl.gate.json --output outputs/downstream_transfer_v1/confirmatory_scores.jsonl
uv run python scripts/analyze_downstream_transfer.py --dataset outputs/downstream_transfer_v1/confirmatory.jsonl --scores outputs/downstream_transfer_v1/confirmatory_scores.jsonl --gate outputs/downstream_transfer_v1/gate_scores.jsonl.gate.json --output-dir outputs/downstream_transfer_v1/analysis
```

Add each applicable supplied prior dataset as a separate --prior-dataset argument at generation. Use --resume only with the same dataset, code vocabulary, config, and output checkpoint.

The tokenizer-only review audit passed for the pinned Qwen3 tokenizer: all 16 code labels are stable two-token continuations across 64 representative prefixes, covering both queries, both entity orientations, and stale/live edits. Run the audit command to create the sealed launch artifact; scoring checks its exact tokenizer, chat template, candidate sequences, and frozen hashes.

The frozen gate requires the correct code to have strictly greater complete-sequence log probability than every other code on at least 47 of 48 current-query records. This is pooled current derived-code accuracy; query and x/z orientation cells are diagnostic only. Deterministic unrestricted greedy generation is recorded as an additional competence diagnostic and does not veto the gate. Its parser accepts only a whole continuation that is exactly one code label. Confirmatory causal estimands use complete-sequence answer-prefix log probabilities, with no termination event and no length normalization.

All four commands emit timestamped INFO logs and tqdm progress on stderr. Generation reports histories and exclusions; the tokenizer audit reports prefixes and code lengths; scoring reports checkpoint coverage, candidate forwards, and unrestricted generations; analysis reports validation and history statistics. Development and gate logs contain competence and counts only. Completed confirmatory score artifacts receive a sealed `.complete.json` manifest binding their score, dataset, gate, config, code, and model/tokenizer revisions; analysis requires and verifies this manifest. Complete checkpoints resume without loading a model, and an existing gate is verified rather than overwritten.

The v4 gate and confirmation used the pinned Qwen3-8B model/tokenizer revisions. Scientific configs, code/value vocabularies, stage seeds, counts, the primary estimand, and the 97% gate criterion remain unchanged.

## Additional model panel

Gemma 3 4B and Phi-4-mini are supported through frozen configs in `configs/downstream_transfer_v1/`. Their tokenizer and model revisions are pinned. Both use the new `gemma_phi_panel_v1` history seed profile (development 20261040, gate 20261041, confirmation 20261042), so their concrete histories and randomized codebooks match for paired model comparisons. These histories are distinct from Qwen v4 histories. Pass the existing Qwen development, gate, and confirmatory datasets as `--prior-dataset` exclusions when generating the panel datasets. Model-specific tokenizer audits and output directories keep their provenance isolated.

Run the following sequence once for each `MODEL` (`gemma3_4b` or `phi4_mini`), setting `MODEL` and `OUT` accordingly. No inference is needed to create or audit datasets; scoring commands below are the inference stages.

```bash
MODEL=gemma3_4b # change to phi4_mini for the second model
OUT=outputs/downstream_transfer_v1/$MODEL
CFG=configs/downstream_transfer_v1/$MODEL.yaml
COMMON=(--config "$CFG")
PRIOR=(--prior-dataset outputs/downstream_transfer_v1/development.jsonl --prior-dataset outputs/downstream_transfer_v1/gate_v2.jsonl --prior-dataset outputs/downstream_transfer_v1/confirmatory_v4.jsonl)

uv run python scripts/audit_downstream_transfer_tokens.py "${COMMON[@]}" --output "$OUT/token_audit.json"
uv run python scripts/generate_downstream_transfer.py "${COMMON[@]}" --stage development "${PRIOR[@]}" --output "$OUT/development.jsonl"
uv run python scripts/run_downstream_transfer.py "${COMMON[@]}" --dataset "$OUT/development.jsonl" --token-audit "$OUT/token_audit.json" --output "$OUT/development_scores.jsonl"
uv run python scripts/generate_downstream_transfer.py "${COMMON[@]}" --stage frozen_gate "${PRIOR[@]}" --prior-dataset "$OUT/development.jsonl" --output "$OUT/gate.jsonl"
uv run python scripts/run_downstream_transfer.py "${COMMON[@]}" --dataset "$OUT/gate.jsonl" --token-audit "$OUT/token_audit.json" --output "$OUT/gate_scores.jsonl"
```

Inspect the sealed gate and proceed only if it passes. Then:

```sh
uv run python scripts/generate_downstream_transfer.py "${COMMON[@]}" --stage confirmatory "${PRIOR[@]}" --prior-dataset "$OUT/development.jsonl" --prior-dataset "$OUT/gate.jsonl" --token-audit "$OUT/token_audit.json" --gate "$OUT/gate_scores.jsonl.gate.json" --output "$OUT/confirmatory.jsonl"
uv run python scripts/run_downstream_transfer.py "${COMMON[@]}" --dataset "$OUT/confirmatory.jsonl" --token-audit "$OUT/token_audit.json" --gate "$OUT/gate_scores.jsonl.gate.json" --output "$OUT/confirmatory_scores.jsonl"
uv run python scripts/analyze_downstream_transfer.py "${COMMON[@]}" --dataset "$OUT/confirmatory.jsonl" --scores "$OUT/confirmatory_scores.jsonl" --gate "$OUT/gate_scores.jsonl.gate.json" --output-dir "$OUT/analysis"
```

The shared panel profile changes only the added models' dataset seeds. The Qwen config and its generated artifacts remain as recorded. Tokenizer audit results must be produced separately for Gemma and Phi; no assumption is made that their code labels have the same tokenization as Qwen.
