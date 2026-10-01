# Single deep chain

Depth counts updates: versions `v0,...,vk` include the initial binding and
`k` updates. Only the focal variable receives updates. Focal x/z and literal
orientation are independently balanced; every history includes both queries.
Each focal version and the stable distractor receive an independent matched
value edit. Actual chat-rendered baseline/edit pairs must differ in exactly
one input token; every answer candidate must remain one continuation token.

Use the existing validated version-chain vocabulary map. Its candidate IDs,
configuration, tokenizer revision and chat template are checked again for the
new prompts. Existing experiment files are preserved.

## Pilot

```bash
uv run python scripts/generate_single_deep_chain.py --stage pilot --config configs/four_query_288.yaml --token-ids outputs/supersession/version_chain_token_ids.json --depths 2,3,4 --n 8 --seed 20261102 --output outputs/supersession/single_deep_chain_pilot.jsonl
uv run python scripts/run_single_deep_chain.py --config configs/four_query_288.yaml --dataset outputs/supersession/single_deep_chain_pilot.jsonl --token-ids outputs/supersession/version_chain_token_ids.json --output outputs/supersession/single_deep_chain_pilot_scores.jsonl
uv run python scripts/analyze_single_deep_chain.py --dataset outputs/supersession/single_deep_chain_pilot.jsonl --scores outputs/supersession/single_deep_chain_pilot_scores.jsonl --output-dir outputs/supersession/single_deep_chain_pilot_analysis
```

The pilot has 24 histories and 480 members. Eligibility requires full-vocabulary
and candidate accuracy at least 0.98 in every baseline and edited cell,
including focal variable, orientation, query role and edited version. With
eight histories per depth, this requires perfect accuracy in each pilot cell.
No threshold override is offered. R does not determine eligibility.

## Full, after inspecting pilot competence

```bash
uv run python scripts/generate_single_deep_chain.py --stage full --config configs/four_query_288.yaml --token-ids outputs/supersession/version_chain_token_ids.json --depths 2,3,4 --n 96 --seed 20261103 --pilot-audit outputs/supersession/single_deep_chain_pilot_analysis/analysis.json --output outputs/supersession/single_deep_chain_full.jsonl
uv run python scripts/run_single_deep_chain.py --config configs/four_query_288.yaml --dataset outputs/supersession/single_deep_chain_full.jsonl --token-ids outputs/supersession/version_chain_token_ids.json --output outputs/supersession/single_deep_chain_full_scores.jsonl
uv run python scripts/analyze_single_deep_chain.py --dataset outputs/supersession/single_deep_chain_full.jsonl --scores outputs/supersession/single_deep_chain_full_scores.jsonl --output-dir outputs/supersession/single_deep_chain_full_analysis
```

Generation recomputes pilot competence from hashed inputs and validates model,
tokenizer, configuration and template provenance. Only passing depths advance;
none passing blocks generation. Full histories use a new seed and exclude
pilot histories and their semantic swaps. Additional prior single-chain datasets
outside the output directory can be supplied with `--exclude-dataset`.
Scoring supports `--resume` and preserves raw candidate logits and ranks.

## Outputs and interpretation

Datasets have `.provenance.json`; scores have `.provenance.json` and `.run.json`.
Each analysis directory contains `analysis.json`, `R_by_age.png`, and CSVs:
`edit_effects`, `R_by_history`, `R_summary`, `paired_contrasts`,
`stable_control_by_history`, `stable_control_summary`, `competence_cells`,
`competence_by_depth`.

`E` is the edited-minus-baseline change in replacement-minus-source logits.
`R_i = E(focal query) - E(distractor query)`. Stable-control relevance reverses
query roles so its own-variable relevance is positive. x/z estimates and a
balanced pooled estimate are reported; pooling uses independent histories,
not two evolving chains within one history. Bootstrap resampling uses histories;
adjacent-version contrasts are paired within history. All audited trials remain.

The existing depth-one full analysis is read only as a sanity check from
`outputs/supersession/version_chain_full_analysis` (override with
`--depth-one-sanity-analysis`). Missing files are recorded explicitly. Its
current-versus-obsolete gap never qualifies new depths.

Age remains confounded with serial position; v0 also occupies the Initial
block. The stable variable is an active-binding control, not an irrelevant
mention. Pointwise intervals and nonsignificant adjacent differences do not
establish obsolete-version equivalence. No mechanistic interventions are run.
