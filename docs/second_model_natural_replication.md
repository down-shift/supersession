# Independent second-model natural-language replication

## Frozen model and design

Use `mistralai/Mistral-7B-Instruct-v0.3`, revision `c170c708c41dac9275d15a8fff4eca08d52bab71`, from the Mistral family, distinct from Qwen3. Both model and tokenizer are pinned to this immutable commit in `configs/mistral7b_natural_replication.yaml`.

Keep `nora_v1`, 96 confirmatory histories, the three focal conditions (live, superseded, counterbalanced irrelevant), the original paired identity-transfer estimator, query-specific symmetric relevance, and the frozen 0.99 competence gate. Validate candidate strings independently under this model's exact tokenizer and chat template. The shared generator needs at least five candidates. Prefer the first model's shared vocabulary when at least five values qualify; otherwise use up to 12 validated values from the same proposal pool. This choice uses tokenization only.

The tokenizer-only validation produced seven candidates: `navy`, `rust`, `wheat`, `bronze`, `lemon`, `olive`, `silver`. Their token IDs are respectively 29201, 15680, 28903, 20835, 24366, 25171, 10514. Only `navy` and `rust` overlap the first model's frozen 12-value vocabulary, so lexical differences and candidate-pool size remain a limitation of the cross-model comparison. The exact paired edit audit passed for all 80 validation pairs. Model competence has not been evaluated.

Do all selection using tokenizer validity and competence only. Do not inspect any `identity_transfer`, `R_*`, or matched-effect results until the confirmatory analysis.

## Commands

Run from the repository root on the CUDA host with the model/tokenizer available. These commands use fresh second-model output paths.

### Validate this tokenizer's candidate vocabulary

```bash
uv run python scripts/validate_natural_supersession_vocab.py --config configs/mistral7b_natural_replication.yaml --values configs/supersession_values.json --match-token-ids outputs/four_query_288/frozen_token_ids.json --output outputs/supersession/mistral_nl_token_ids.json
```

This performs tokenizer-only validation. Stop if it produces fewer than five values or if `exact_one_token_pair_audit.status` is not `passed`. An existing token map is preserved; use a fresh path for revalidation. The supplied map was validated before subsequent additions to the validator's metadata fields; rerunning on the CUDA host records the current code hashes.

### Development competence set (24 histories)

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage development --templates nora_v1 --n 24 --seed 20261023 --values configs/supersession_values.json --token-ids outputs/supersession/mistral_nl_token_ids.json --output outputs/supersession/mistral_nl_dev.jsonl
uv run python scripts/run_supersession_behavior.py --config configs/mistral7b_natural_replication.yaml --dataset outputs/supersession/mistral_nl_dev.jsonl --token-ids outputs/supersession/mistral_nl_token_ids.json --output outputs/supersession/mistral_nl_dev_scores.jsonl
uv run python scripts/analyze_natural_competence.py --stage development --dataset outputs/supersession/mistral_nl_dev.jsonl --behavior outputs/supersession/mistral_nl_dev_scores.jsonl --token-ids outputs/supersession/mistral_nl_token_ids.json --config configs/mistral7b_natural_replication.yaml --output outputs/supersession/mistral_nl_dev_competence.json
```

Development output is descriptive competence only. Proceed with the fixed `nora_v1` prompt; do not select on causal effects.

### Fresh frozen competence gate (24 histories)

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage frozen_gate --templates nora_v1 --n 24 --seed 20261024 --values configs/supersession_values.json --token-ids outputs/supersession/mistral_nl_token_ids.json --config configs/mistral7b_natural_replication.yaml --output outputs/supersession/mistral_nl_frozen_gate.jsonl
uv run python scripts/run_supersession_behavior.py --config configs/mistral7b_natural_replication.yaml --dataset outputs/supersession/mistral_nl_frozen_gate.jsonl --token-ids outputs/supersession/mistral_nl_token_ids.json --output outputs/supersession/mistral_nl_frozen_gate_scores.jsonl
uv run python scripts/analyze_natural_competence.py --stage frozen_gate --selected-template nora_v1 --dataset outputs/supersession/mistral_nl_frozen_gate.jsonl --behavior outputs/supersession/mistral_nl_frozen_gate_scores.jsonl --token-ids outputs/supersession/mistral_nl_token_ids.json --config configs/mistral7b_natural_replication.yaml --prior-dataset outputs/supersession/mistral_nl_dev.jsonl --output outputs/supersession/mistral_nl_frozen_gate.json
```

Continue only if the gate artifact says `"pass": true`. It requires all 64 cells per template and counts each unique prompt once within each represented preregistered condition × query × orientation × edited-variable × edit-status × pair-direction × slot-order cell. It requires full-vocabulary accuracy ≥0.99, candidate accuracy ≥0.99, mean candidate rank ≤1.01, and positive mean current-minus-stale margin where applicable.

### Confirmatory (96 histories; do not run until the gate passes)

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage confirmatory --templates nora_v1 --frozen-gate-required outputs/supersession/mistral_nl_frozen_gate.json --prior-dataset outputs/supersession/mistral_nl_dev.jsonl --config configs/mistral7b_natural_replication.yaml --n 96 --seed 20261025 --values configs/supersession_values.json --token-ids outputs/supersession/mistral_nl_token_ids.json --output outputs/supersession/mistral_nl_confirmatory.jsonl
uv run python scripts/run_supersession_behavior.py --config configs/mistral7b_natural_replication.yaml --dataset outputs/supersession/mistral_nl_confirmatory.jsonl --token-ids outputs/supersession/mistral_nl_token_ids.json --output outputs/supersession/mistral_nl_confirmatory_scores.jsonl
uv run python scripts/analyze_supersession_behavior.py --kind controls_counterbalanced --dataset outputs/supersession/mistral_nl_confirmatory.jsonl --behavior outputs/supersession/mistral_nl_confirmatory_scores.jsonl --output-dir outputs/supersession/mistral_nl_confirmatory_analysis
```

Generation validates the frozen-gate seal and recomputes competence from the hash-bound gate data, verifies tokenizer/model/config/renderer provenance and the selected template, and rejects history overlap with development and gate data. Scoring repeats the exact token continuation and paired one-token edit audits under the second model's tokenizer/chat template.

## Expected artifacts

- `mistral_nl_token_ids.json`: selected/rejected vocabulary, tokenizer and chat-template revisions/hash, candidate validation provenance, exact edit audit.
- For development, frozen gate, and confirmatory stages: JSONL data plus `.provenance.json`; score JSONL plus `.run.json` and `.provenance.json`.
- Competence reports: `mistral_nl_dev_competence.json` and `mistral_nl_frozen_gate.json`.
- Confirmatory analysis directory: `matched_edit_effects.csv`, `history_relevance.csv`, `competence.csv`, and `summary.json`.

Confirmatory estimands are `R_superseded - R_irrelevant_counterbalanced` (primary) and `R_live - R_superseded` (secondary), with 2,000 history-bootstrap draws and the shared seed/analysis code. The ordinary uncounterbalanced irrelevant cell remains diagnostic and does not determine the primary conclusion.

Gate contract `natural_competence_v2` records actual dataset seeds (development 20261023, gate 20261024, confirmatory 20261025), separately from the scoring config seed. Old gate artifacts must be regenerated by competence-only analysis of saved scores into a fresh output file; preserve previous artifacts.
