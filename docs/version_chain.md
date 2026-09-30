# Audited version-chain relevance

This workflow is independent of status and focal-validity experiments. Depth k
means k updates per variable: v0 is initial and vk is current. The default depths
are 1, 2, 4, 8. Histories have distinct values for every binding in both chains;
each edit uses a replacement absent from that history. The same replacement is
used across x/z queries for a given version. Literal naming and variable block
order are crossed across histories at each depth. Initial semantic order is x,z.

The old frozen 12-value map cannot supply 18 distinct bindings plus a replacement
at depth 8. Prepare a new chain map from the original proposal vocabulary plus
the replication proposal vocabulary using the exact configured tokenizer. This
is a documented vocabulary expansion, not lexical replication. Preparation
loads only the tokenizer. Actual baseline/edit prompts, assignment spans, stable
candidate continuation IDs, and exact one-token differences are audited before
scoring. No old token map or saved output is overwritten.

Run the following on the model host. Commands stop on errors. The pilot uses
eight independent histories per depth (32 total, 1,216 pair members). Full scoring
has 96 histories per eligible depth, on a new seed and concrete histories.

```bash
# 1. Pilot generation (including the required tokenizer-only vocabulary preparation).
uv run python scripts/prepare_version_chain_tokens.py --config configs/four_query_288.yaml --output outputs/supersession/version_chain_token_ids.json &&
uv run python scripts/generate_version_chain.py --stage pilot --config configs/four_query_288.yaml --token-ids outputs/supersession/version_chain_token_ids.json --depths 1,2,4,8 --n 8 --seed 20261030 --output outputs/supersession/version_chain_pilot.jsonl

# 2. Pilot scoring. Add --resume only to continue this exact interrupted run.
uv run python scripts/run_version_chain.py --config configs/four_query_288.yaml --dataset outputs/supersession/version_chain_pilot.jsonl --token-ids outputs/supersession/version_chain_token_ids.json --output outputs/supersession/version_chain_pilot_scores.jsonl

# 3. Pilot analysis and competence-based depth eligibility.
uv run python scripts/analyze_version_chain.py --dataset outputs/supersession/version_chain_pilot.jsonl --scores outputs/supersession/version_chain_pilot_scores.jsonl --output-dir outputs/supersession/version_chain_pilot_analysis

# 4. Full generation: blocked if no depths passed; automatically capped at first failure.
uv run python scripts/generate_version_chain.py --stage full --config configs/four_query_288.yaml --token-ids outputs/supersession/version_chain_token_ids.json --depths 1,2,4,8 --n 96 --seed 20261031 --pilot-audit outputs/supersession/version_chain_pilot_analysis/analysis.json --output outputs/supersession/version_chain_full.jsonl

# 5. Full scoring: run deliberately after reviewing pilot competence.
uv run python scripts/run_version_chain.py --config configs/four_query_288.yaml --dataset outputs/supersession/version_chain_full.jsonl --token-ids outputs/supersession/version_chain_token_ids.json --output outputs/supersession/version_chain_full_scores.jsonl

# 6. Full analysis.
uv run python scripts/analyze_version_chain.py --dataset outputs/supersession/version_chain_full.jsonl --scores outputs/supersession/version_chain_full_scores.jsonl --output-dir outputs/supersession/version_chain_full_analysis
```

Pilot eligibility uses only latest-answer full-vocabulary and candidate accuracy,
requiring at least 0.98 in every depth/query/orientation/block-order cell, both
baseline and edited. Baseline scores are deduplicated by history/query; repeated
copies across interventions are not treated as independent competence trials.
Edited trials contribute to the corresponding member cells. Full-vocabulary ranks
and current-minus-previous/current-minus-oldest margins are reported separately.
Eligibility is the ascending prefix of tested depths; failure at one depth caps
all higher depths. Full generation recomputes the pilot competence and verifies
dataset, scores, provenance, config, vocabulary, template, and exact revisions.
Pilot causal effects are exploratory and never used to choose depths.

The intervention effect E is (replacement minus source logit in edited prompt)
minus (replacement minus source logit in baseline prompt). R_i subtracts E for
the other-variable query from E for the same-variable query. Analysis keeps x,z
separate and averages their effects within history before bootstrapping symmetric
means. Every descriptive contrast is formed within history before a 95% bootstrap
CI is computed. No competence-based trial filtering is applied to causal analysis.

Outputs include raw candidate logits in scored JSONL and exact rendered prompts;
dataset and score provenance sidecars; checkpoint manifests; `edit_effects.csv`,
`R_by_history.csv`, `R_summary.csv`, `paired_contrasts.csv`,
`competence_cells.csv`, `competence_by_depth.csv`, `R_by_age.png`, and `analysis.json`.

Interpretation remains descriptive: a large current-vs-previous difference with
similar obsolete means supports the discrete-status pattern; ordered obsolete
contrasts support residual recency; both together support the hybrid pattern.
Nonsignificance does not establish an inactive class or equivalence. Pointwise CIs
have no multiple-comparison correction. Age covaries with serial position; v0 also
uses the Initial block rather than Updates. Histories differ across depths, so
cross-depth comparisons are not paired. There is no clean unassigned-mention
baseline here, so oldest-vs-irrelevant is not estimated. A small pilot cannot
guarantee full-run competence. No mechanistic partitions or results are read.
