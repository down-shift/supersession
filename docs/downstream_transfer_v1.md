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

## Launch

Use the repository uv environment (`uv sync --extra model --extra dev` on the experiment machine). All commands below run from the repository root.

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

The frozen gate now requires both (a) the correct code to have strictly greater complete-sequence log probability than every other code and (b) deterministic unrestricted greedy generation to produce exactly the correct code label, each at or above 97% over the 48 current-query gate records. The generation check parses the whole short continuation as one exact code label; explanations, fragments, and other outputs fail. Confirmatory causal estimands continue to use complete-sequence log probabilities. Those scores are answer-prefix probabilities, with no termination event and no length normalization, matching the causal scorer.

All four commands emit timestamped INFO logs and tqdm progress on stderr. Generation reports histories and exclusions; the tokenizer audit reports prefixes and code lengths; scoring reports checkpoint coverage, candidate forwards, and unrestricted generations; analysis reports validation and history statistics. Development and gate logs contain competence and counts only. Completed confirmatory score artifacts receive a sealed `.complete.json` manifest binding their score, dataset, gate, config, code, and model/tokenizer revisions; analysis requires and verifies this manifest. Complete checkpoints resume without loading a model, and an existing gate is verified rather than overwritten.

No experiment-model inference has been run. Scientific configs, code/value vocabularies, stage seeds, counts, the primary estimand, and the 97% gate criterion are unchanged.
