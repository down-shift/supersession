# Llama 3.1 natural-language replication

This is the second-model replication of the frozen `nora_v1` protocol. It uses 24 development histories, then a fresh 24-history frozen competence gate. Generate 96 confirmatory histories only after that gate passes. Keep the existing three conditions (`live`, `superseded`, and `irrelevant_counterbalanced`), estimands (E) and (R), contrasts, competence thresholds, and 2,000-history-bootstrap analysis unchanged. Do not inspect causal outcomes during development or gating.

The model and tokenizer are pinned to `meta-llama/Llama-3.1-8B-Instruct` commit `0e9e39f249a16976918f6564b8830bc894c89659`. The config requests the shared loader's `int8` mode, CUDA placement, and the repository's existing chat-template rendering and scoring path. Llama 3.1 repositories are gated; the host must have accepted the license and be authenticated with Hugging Face.

## Tokenizer-only validation

Run from the repository root on the authorized model host. This loads tokenizer/config metadata only; it does not load weights or run inference. It checks answer continuation tokenization in the exact rendered prompts and runs the existing exact one-token paired-edit audit. Stop unless at least five distinct candidates pass and the paired-edit audit reports `passed`.

```bash
uv run python scripts/validate_natural_supersession_vocab.py --config configs/llama31_8b_natural_replication.yaml --values configs/supersession_values.json --match-token-ids outputs/four_query_288/frozen_token_ids.json --output outputs/supersession/llama31_nl_token_ids.json
```

The selected vocabulary must be taken from this validator's output. Do not substitute a vocabulary based on model answers. Preserve that artifact; it binds the chosen values to the tokenizer revision, chat-template hash, prompt renderer, validation seed, and paired-edit audit.

## Development competence only (24 histories)

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage development --templates nora_v1 --n 24 --seed 20261023 --values configs/supersession_values.json --token-ids outputs/supersession/llama31_nl_token_ids.json --output outputs/supersession/llama31_nl_dev.jsonl
uv run python scripts/run_supersession_behavior.py --config configs/llama31_8b_natural_replication.yaml --dataset outputs/supersession/llama31_nl_dev.jsonl --token-ids outputs/supersession/llama31_nl_token_ids.json --competence-only --output outputs/supersession/llama31_nl_dev_scores.jsonl
uv run python scripts/analyze_natural_competence.py --stage development --dataset outputs/supersession/llama31_nl_dev.jsonl --behavior outputs/supersession/llama31_nl_dev_scores.jsonl --token-ids outputs/supersession/llama31_nl_token_ids.json --config configs/llama31_8b_natural_replication.yaml --output outputs/supersession/llama31_nl_dev_competence.json
```

Development is descriptive competence only. Keep `nora_v1` fixed; do not select on causal outcomes.

## Fresh frozen competence gate (24 histories)

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage frozen_gate --templates nora_v1 --n 24 --seed 20261024 --values configs/supersession_values.json --token-ids outputs/supersession/llama31_nl_token_ids.json --config configs/llama31_8b_natural_replication.yaml --output outputs/supersession/llama31_nl_frozen_gate.jsonl
uv run python scripts/run_supersession_behavior.py --config configs/llama31_8b_natural_replication.yaml --dataset outputs/supersession/llama31_nl_frozen_gate.jsonl --token-ids outputs/supersession/llama31_nl_token_ids.json --competence-only --output outputs/supersession/llama31_nl_frozen_gate_scores.jsonl
uv run python scripts/analyze_natural_competence.py --stage frozen_gate --selected-template nora_v1 --dataset outputs/supersession/llama31_nl_frozen_gate.jsonl --behavior outputs/supersession/llama31_nl_frozen_gate_scores.jsonl --token-ids outputs/supersession/llama31_nl_token_ids.json --config configs/llama31_8b_natural_replication.yaml --prior-dataset outputs/supersession/llama31_nl_dev.jsonl --output outputs/supersession/llama31_nl_frozen_gate.json
```

Proceed only if the gate artifact reports `"pass": true`. The frozen contract is `natural_competence_v2`: all 64 required cells, full-vocabulary accuracy ≥0.99, candidate accuracy ≥0.99, mean candidate rank ≤1.01, and positive current-minus-stale margin wherever defined. Do not adjust these thresholds after results. The gate analysis does not calculate (E), (R), contrasts, or bootstrap effects. Generation and scoring provenance sidecars bind each stage to its data, config, token map, renderer, and code; gate and prior history signatures are checked before confirmatory data can be generated.

## Confirmatory (96 histories; run only after a passing gate)

These commands are intentionally documented for the authorized host; do not run them unless the frozen gate passed. Confirmatory generation verifies the gate seal and provenance and rejects concrete-history overlap with both earlier stages. Confirmatory scoring uses the normal causal scoring path.

```bash
uv run python scripts/generate_supersession_experiments.py --kind controls_counterbalanced --natural-language --stage confirmatory --templates nora_v1 --frozen-gate-required outputs/supersession/llama31_nl_frozen_gate.json --prior-dataset outputs/supersession/llama31_nl_dev.jsonl --config configs/llama31_8b_natural_replication.yaml --n 96 --seed 20261025 --values configs/supersession_values.json --token-ids outputs/supersession/llama31_nl_token_ids.json --output outputs/supersession/llama31_nl_confirmatory.jsonl
uv run python scripts/run_supersession_behavior.py --config configs/llama31_8b_natural_replication.yaml --dataset outputs/supersession/llama31_nl_confirmatory.jsonl --token-ids outputs/supersession/llama31_nl_token_ids.json --output outputs/supersession/llama31_nl_confirmatory_scores.jsonl
uv run python scripts/analyze_supersession_behavior.py --kind controls_counterbalanced --dataset outputs/supersession/llama31_nl_confirmatory.jsonl --behavior outputs/supersession/llama31_nl_confirmatory_scores.jsonl --output-dir outputs/supersession/llama31_nl_confirmatory_analysis
```

Use the existing primary and secondary contrasts and the existing 2,000-draw history bootstrap in `analyze_supersession_behavior.py`. No confirmatory inference is performed by the implementation task.
