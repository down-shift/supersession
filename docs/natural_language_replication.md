# Natural-language supersession replication

This protocol reuses the paired `controls_counterbalanced` generator, token audit, resumable scorer, and history bootstrap. It uses the frozen symbolic candidate token map (`configs/supersession_values.json` proposals intersected with the validated token map). Before confirmatory interpretation, the frozen gate requires, in every template × condition cell, full-vocabulary accuracy ≥0.99, candidate accuracy ≥0.99, mean candidate rank ≤1.01, and positive mean current-minus-stale logit margin wherever a stale value exists. These criteria are fixed in `scripts/analyze_natural_competence.py` and cannot be changed from the command line.

The three controlled prompt templates are `nora_v1`, `record_v1`, and `tag_v1` in `src/data/supersession_behavior.py`. Names, attributes, entity order, query entity, irrelevant mention order, and template assignment vary by history. The existing control schema also retains the uncounterbalanced `irrelevant` diagnostic cell for audit compatibility; the confirmatory contrast is explicitly `R_superseded - R_irrelevant_counterbalanced`.

Development is competence only. Do not run the causal analysis on development outputs or inspect identity-transfer effects when selecting a template. After choosing and recording a single template family, freeze it; the frozen gate and confirmatory histories use fresh seeds. Confirmatory generation refuses to proceed without a passing frozen gate artifact.

## Commands

Examples use the repository's validated symbolic token map. Replace the paths if that artifact is elsewhere.

1. Development generation (three templates, 24 histories):

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage development --templates nora_v1,record_v1,tag_v1 --n 24 --seed 20261020 --values configs/supersession_values.json --token-ids outputs/four_query_288/frozen_token_ids.json --output outputs/supersession/nl_dev.jsonl
```

2. Development scoring:

```bash
uv run python scripts/run_supersession_behavior.py --config configs/four_query_288.yaml --dataset outputs/supersession/nl_dev.jsonl --token-ids outputs/four_query_288/frozen_token_ids.json --output outputs/supersession/nl_dev_scores.jsonl
```

3. Development analysis (competence fields only):

```bash
uv run python scripts/analyze_natural_competence.py --stage development --dataset outputs/supersession/nl_dev.jsonl --behavior outputs/supersession/nl_dev_scores.jsonl --token-ids outputs/four_query_288/frozen_token_ids.json --config configs/four_query_288.yaml --output outputs/supersession/nl_dev_competence.json
```

Choose one template using only this file, record the choice, then use that same template in the next two stages.

4. Fresh frozen competence gate generation and scoring (24 histories):

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage frozen_gate --templates nora_v1 --n 24 --seed 20261021 --values configs/supersession_values.json --token-ids outputs/four_query_288/frozen_token_ids.json --output outputs/supersession/nl_frozen_gate.jsonl
uv run python scripts/run_supersession_behavior.py --config configs/four_query_288.yaml --dataset outputs/supersession/nl_frozen_gate.jsonl --token-ids outputs/four_query_288/frozen_token_ids.json --output outputs/supersession/nl_frozen_gate_scores.jsonl
uv run python scripts/analyze_natural_competence.py --stage frozen_gate --selected-template nora_v1 --dataset outputs/supersession/nl_frozen_gate.jsonl --behavior outputs/supersession/nl_frozen_gate_scores.jsonl --token-ids outputs/four_query_288/frozen_token_ids.json --config configs/four_query_288.yaml --prior-dataset outputs/supersession/nl_dev.jsonl --output outputs/supersession/nl_frozen_gate.json
```

5. Confirmatory generation (96 fresh histories; command fails unless gate passes):

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage confirmatory --templates nora_v1 --frozen-gate-required outputs/supersession/nl_frozen_gate.json --prior-dataset outputs/supersession/nl_dev.jsonl --config configs/four_query_288.yaml --n 96 --seed 20261022 --values configs/supersession_values.json --token-ids outputs/four_query_288/frozen_token_ids.json --output outputs/supersession/nl_confirmatory.jsonl
```

6. Confirmatory scoring:

```bash
uv run python scripts/run_supersession_behavior.py --config configs/four_query_288.yaml --dataset outputs/supersession/nl_confirmatory.jsonl --token-ids outputs/four_query_288/frozen_token_ids.json --output outputs/supersession/nl_confirmatory_scores.jsonl
```

7. Confirmatory analysis:

```bash
uv run python scripts/analyze_supersession_behavior.py --kind controls_counterbalanced --dataset outputs/supersession/nl_confirmatory.jsonl --behavior outputs/supersession/nl_confirmatory_scores.jsonl --output-dir outputs/supersession/nl_confirmatory_analysis
```

The gate counts each unique rendered prompt once within every semantic cell represented by its duplicate rows and checks each template × condition × query × orientation × edited variable × edit status × pair direction × slot-order cell. The ordinary irrelevant cell remains diagnostic and cannot block the gate. The frozen artifact records data and scoring hashes, model/tokenizer revisions, config/token-map/renderer hashes, seed, selected template, and concrete history signatures. Confirmatory generation verifies those artifacts, recomputes the gate from the hash-bound raw scores, and rejects overlap with both frozen-gate and supplied development datasets. Development renders each semantic history under every candidate template, removing the wording/name/attribute confound.

The confirmatory `history_relevance.csv` contains symmetric `R_live`, `R_superseded`, and `R_irrelevant_counterbalanced` values and their history-paired differences. `summary.json` gives history-bootstrap CIs, including the confirmatory superseded-minus-counterbalanced-irrelevant contrast and the descriptive live-minus-superseded contrast. Other artifacts: generation `.provenance.json`; scoring `.run.json`, `.provenance.json`, and resumable score JSONL; analysis `matched_edit_effects.csv`, `history_relevance.csv`, `competence.csv`, and `summary.json`. Preserve failed and partial artifacts and resume scoring with `--resume`.

Scientific caveats: the template family is deliberately simple and remains an English controlled paraphrase rather than broad language generalization. Values and tokenization depend on the existing validated tokenizer-specific map. The default template embeds query wording that differs slightly between templates, so template choice must be frozen solely from competence. The strict gate is a task-competence screen, not evidence that all causal contrasts are estimable.

Gate contract `natural_competence_v2` requires all 64 semantic cells per template, including both edited-variable baseline cells. Gate seeds come from dataset records and must match scoring provenance `dataset_seed`; `scoring_config_seed` is recorded separately. Freshness signatures include initial values, replacements, entities, attribute, and orientation. Older gate artifacts require competence-only reanalysis of saved scores into a fresh output file before use; preserve the original artifacts. No new inference is needed.
