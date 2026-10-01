# Phi-4-mini natural-language replication

Frozen protocol: `nora_v1`; 24 development histories; a fresh 24-history competence gate; 96 confirmatory histories only if that gate passes. Dataset construction, conditions, scoring definitions, thresholds, paired edits, seeds, and analysis remain those in `docs/natural_language_replication.md`. The model and tokenizer are pinned to `microsoft/Phi-4-mini-instruct` revision `cfbefacb99257ffa30c83adab238a50856ac3083`. BF16 is requested without quantization. The pinned model uses the Phi3 remote implementation; `trust_remote_code` is enabled for this exact immutable revision. The loader applies narrow compatibility shims for Transformers' moved loss-typing alias and changed tied-weight mapping API, recording both in scoring provenance.

## Validate before scoring

Run tokenizer-only validation first. This constructs the model-specific candidate vocabulary under the actual tokenizer and `nora_v1` chat template, then runs the exact one-token paired-edit audit. Validation fails closed if fewer than five distinct candidates survive or if any paired-edit token-alignment check fails. Do not score unless the token map is written successfully and its audit status is `passed`.

```bash
uv run python scripts/validate_natural_supersession_vocab.py --config configs/phi4_mini_natural_replication.yaml --values configs/supersession_values.json --match-token-ids outputs/four_query_288/frozen_token_ids.json --output outputs/supersession/phi4_mini_nl_token_ids.json
```

## Development (24 histories; competence only)

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage development --templates nora_v1 --n 24 --seed 20261023 --values configs/supersession_values.json --token-ids outputs/supersession/phi4_mini_nl_token_ids.json --config configs/phi4_mini_natural_replication.yaml --output outputs/supersession/phi4_mini_nl_dev.jsonl
uv run python scripts/run_supersession_behavior.py --config configs/phi4_mini_natural_replication.yaml --dataset outputs/supersession/phi4_mini_nl_dev.jsonl --token-ids outputs/supersession/phi4_mini_nl_token_ids.json --competence-only --output outputs/supersession/phi4_mini_nl_dev_scores.jsonl
uv run python scripts/analyze_natural_competence.py --stage development --dataset outputs/supersession/phi4_mini_nl_dev.jsonl --behavior outputs/supersession/phi4_mini_nl_dev_scores.jsonl --token-ids outputs/supersession/phi4_mini_nl_token_ids.json --config configs/phi4_mini_natural_replication.yaml --output outputs/supersession/phi4_mini_nl_dev_competence.json
```

The competence artifact reports full-vocabulary accuracy by condition, candidate-restricted accuracy as a diagnostic, parse rate (greedy decoded token, whitespace-trimmed, exactly matches a validated candidate), failed gate cells, target-token rank summaries, and the most common wrong decoded tokens. The frozen `natural_competence_v2` contract remains unchanged, including its candidate-accuracy threshold. Missing/incomplete semantic cells fail analysis rather than being silently omitted.

## Fresh frozen gate (24 histories; competence only)

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage frozen_gate --templates nora_v1 --n 24 --seed 20261024 --values configs/supersession_values.json --token-ids outputs/supersession/phi4_mini_nl_token_ids.json --config configs/phi4_mini_natural_replication.yaml --output outputs/supersession/phi4_mini_nl_frozen_gate.jsonl
uv run python scripts/run_supersession_behavior.py --config configs/phi4_mini_natural_replication.yaml --dataset outputs/supersession/phi4_mini_nl_frozen_gate.jsonl --token-ids outputs/supersession/phi4_mini_nl_token_ids.json --competence-only --output outputs/supersession/phi4_mini_nl_frozen_gate_scores.jsonl
uv run python scripts/analyze_natural_competence.py --stage frozen_gate --selected-template nora_v1 --dataset outputs/supersession/phi4_mini_nl_frozen_gate.jsonl --behavior outputs/supersession/phi4_mini_nl_frozen_gate_scores.jsonl --token-ids outputs/supersession/phi4_mini_nl_token_ids.json --config configs/phi4_mini_natural_replication.yaml --prior-dataset outputs/supersession/phi4_mini_nl_dev.jsonl --output outputs/supersession/phi4_mini_nl_frozen_gate.json
```

Do not revise the frozen competence contract after seeing results. The analyzer records failed cells and uses the repository's existing thresholds. All artifacts record the exact model/tokenizer revisions, data seeds, and hash-bound provenance. No confirmatory histories or inference may be generated unless the frozen gate artifact has a true pass field and passes the generator's integrity/provenance checks.
