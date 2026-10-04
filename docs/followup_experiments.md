# Time-limited relational follow-ups

These are separate from relational v2. Do not edit, relabel, or pool these records with v2 results. Any follow-up datasets generated before the freshness and factorial-validation fixes are archival design artifacts; do not score them or use them for development or confirmation.

## Existing infrastructure and protocol decisions

The generator reuses the v2 candidate vocabulary and the same current assignments, entity pair, query structure, paired historical-value edit, and history-level estimand logic. V2 candidate likelihood scoring is the preferred scoring method for E/R and current-versus-historical margins. Existing history bootstrap convention is 2,000 draws with seed 73021.

V2 always marks historical assignments with `Previously`; this follow-up crosses that marker with assigned superseded history versus unrelated entity mention. It does not identify a pure marker effect independent of construction, because assignment and mention differ in syntax and meaning. Report both within-construction marker effects and their difference-in-differences with history bootstrap intervals. A CI crossing zero is inconclusive, not evidence of equivalence. Report the source/replacement E components contributing to R.

V2 scores candidate continuations and imposes a ≥99% competence gate. This follow-up has no competence gate. Its runner records deterministic complete answers with exact parsing, all-history transition counts, and candidate continuation scores. The parser contract is frozen here: strip whitespace and edge punctuation, then exact case-insensitive candidate match; every other string is `other`.

## Experiment 1: marker × construction

Templates are validated in code using the supplied examples. Generated rows cross construction (`superseded`, `entity_mention`), marker (0/1), historical order (0/1), current order (0/1), edited entity (x/z), query entity (x/z), and baseline/edit direction. Histories, current assignments, value candidates, questions, and edit values are held fixed across the four marker/construction cells. Both aligned `(historical_order == current_order)` and reversed order are retained.

For each history and cell, compute E from the paired candidate scores using the existing v2 identity-transfer definition. Compute R from the two query-specific E means for each edited entity, then symmetrize x/z. Report the two marker effects (with minus without marker) within each construction, the marker × construction interaction, E's source and replacement log-mass components, and aligned/reversed strata. Bootstrap histories, not rows. Do not claim active suppression by conflict.

## Experiment 2: harder histories

Difficulty levels are prespecified as 2, 4, and 6 distractor entities. Development chooses the hardest level whose complete-answer accuracy is between 0.65 and 0.90 inclusive; if none qualifies, choose the level nearest 0.775 (ties go to the smaller level). Freeze that level, prompt, candidate values, parser, metrics, and analysis before generating test rows. Development and test use disjoint seeded history namespaces. Generate a balanced, fixed allocation before model outcomes; evaluate every test history.

Each history includes superseded, entity-mention, early-unassigned, and late-unassigned conditions, both query entities, both history/current order factors, each edited entity, and paired baseline/edit prompts. The edited history changes only the historical value; the answer and current assignment remain fixed. Report complete-answer accuracy, stale-answer frequency, all paired correct/stale/other transitions, candidate-score current-minus-historical margins, and v2 R secondarily. Estimate edit effects on stale-answer frequency and the difference of that effect between superseded and each matched control. More errors under harder prompts alone do not establish a historical-information cause.

Use deterministic greedy decoding and the parser above. No ≥99% competence gate applies. No test-only error selection or difficulty tuning is allowed. Record model/revision, runtime, seed, dataset/code hashes, exclusions, and every failed development rule. Qwen and Gemma are the intended model pair; if only Qwen can run, mark Gemma pending.

## Commands and execution status

Run tests and design/storage pilots with the repository environment:

```bash
uv run pytest -q tests/test_followups.py
uv sync --frozen --extra model --extra dev
uv run --frozen python -m scripts.followups marker --stage pilot --histories 2 \
  --output outputs/followups/pilot_marker.jsonl \
  --report outputs/followups/pilot_marker_report.json
uv run --frozen --extra model python -m scripts.run_followups score \
  --experiment marker --mode candidate --config configs/cross_model_relational_v2/qwen3_8b.yaml \
  --dataset outputs/followups/pilot_marker.jsonl \
  --output outputs/followups/qwen_marker_pilot.jsonl
uv run --frozen python -m scripts.followups harder --stage pilot --histories 4 \
  --distractors 2 --seed 20261099 --output outputs/followups/pilot_harder.jsonl \
  --report outputs/followups/pilot_harder_report.json
uv run --frozen --extra model python -m scripts.run_followups score \
  --experiment harder --mode both --config configs/cross_model_relational_v2/qwen3_8b.yaml \
  --dataset outputs/followups/pilot_harder.jsonl \
  --output outputs/followups/qwen_harder_pilot.jsonl
uv run --frozen python -m scripts.run_followups analyze --experiment harder --stage pilot \
  --dataset outputs/followups/pilot_harder.jsonl --scores outputs/followups/qwen_harder_pilot.jsonl \
  --output outputs/followups/qwen_harder_pilot_analysis.json
```

The marker pilot has 2 histories × 128 factorial members and 192 unique prompts. The harder pilot has 4 histories × 128 members and 384 unique prompts. At 12 histories per development level, three levels require 3,456 unique prompts; the 24-history test requires 2,304. Confirmatory marker scoring at 24 histories also requires 2,304. Exact runtime depends on target hardware and must be projected from the pilot's measured seconds per unique prompt. These pilot commands measure runtime and validate the end-to-end scoring/analysis path. Do not proceed if either pilot fails. The target GPU is not available in the current environment, so no model inference or development selection has run here.

After the Qwen pilot succeeds, generate all three matched development levels. Use the pilot dataset as an explicit exclusion for every development level:

```bash
uv sync --frozen --extra model --extra dev
uv run --frozen --extra model python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'

for N in 2 4 6; do
  uv run --frozen python -m scripts.followups harder --stage development --histories 12 --distractors "$N" \
    --prior-dataset outputs/followups/pilot_harder.jsonl \
    --output "outputs/followups/development_n${N}.jsonl" \
    --report "outputs/followups/development_n${N}_report.json"
  uv run --frozen --extra model python -m scripts.run_followups score \
    --experiment harder --mode both --config configs/cross_model_relational_v2/qwen3_8b.yaml \
    --dataset "outputs/followups/development_n${N}.jsonl" \
    --output "outputs/followups/qwen_development_n${N}_scores.jsonl"
  uv run --frozen python -m scripts.run_followups analyze --experiment harder --stage development \
    --dataset "outputs/followups/development_n${N}.jsonl" \
    --scores "outputs/followups/qwen_development_n${N}_scores.jsonl" \
    --output "outputs/followups/qwen_development_n${N}_analysis.json"
done
uv run --frozen python -m scripts.run_followups select-hard-difficulty \
  --development-reports outputs/followups/qwen_development_n2_analysis.json \
    outputs/followups/qwen_development_n4_analysis.json \
    outputs/followups/qwen_development_n6_analysis.json \
  --output outputs/followups/harder_protocol_freeze.json
```

Preserve any `.failures.jsonl`; stop on a failed Qwen pilot. The 12-history-per-level development loop is for after the runtime pilot passes and its measured throughput makes the cost feasible. If Gemma cannot run in the available time, use Qwen development for selection and label Gemma pending, not as a replication. Generate the test dataset only after reviewing the sealed freeze. Set `N` to the recorded `selected_n_distractors` in that file:

```bash
N=4  # replace with selected_n_distractors from the freeze
uv run --frozen python -m scripts.followups harder --stage test --histories 24 \
  --distractors "$N" --freeze outputs/followups/harder_protocol_freeze.json \
  --prior-dataset outputs/followups/pilot_harder.jsonl \
  --prior-dataset outputs/followups/development_n2.jsonl \
  --prior-dataset outputs/followups/development_n4.jsonl \
  --prior-dataset outputs/followups/development_n6.jsonl \
  --output outputs/followups/test_n${N}.jsonl \
  --report outputs/followups/test_n${N}_report.json
uv run --frozen --extra model python -m scripts.run_followups score \
  --experiment harder --mode both --config configs/cross_model_relational_v2/qwen3_8b.yaml \
  --dataset "outputs/followups/test_n${N}.jsonl" \
  --output "outputs/followups/qwen_test_n${N}_scores.jsonl" \
  --freeze outputs/followups/harder_protocol_freeze.json
uv run --frozen python -m scripts.run_followups analyze --experiment harder --stage test \
  --dataset "outputs/followups/test_n${N}.jsonl" \
  --scores "outputs/followups/qwen_test_n${N}_scores.jsonl" \
  --output "outputs/followups/qwen_test_n${N}_analysis.json"
```

Use the matching Gemma config and fresh `gemma_test...` output names for its test run after Qwen, if feasible. If Gemma cannot run in the available time, mark it pending and report Qwen as a single-model evaluation, not a replication. To run Experiment 1 after the runtime pilot passes, generate confirmatory histories with explicit pilot exclusions, then score and analyze:

```bash
uv run --frozen python -m scripts.followups marker --stage confirmatory --histories 24 \
  --prior-dataset outputs/followups/pilot_marker.jsonl \
  --output outputs/followups/marker_confirmatory.jsonl \
  --report outputs/followups/marker_confirmatory_report.json
uv run --frozen --extra model python -m scripts.run_followups score \
  --experiment marker --mode candidate --config configs/cross_model_relational_v2/qwen3_8b.yaml \
  --dataset outputs/followups/marker_confirmatory.jsonl \
  --output outputs/followups/qwen_marker_confirmatory_scores.jsonl
uv run --frozen python -m scripts.run_followups analyze --experiment marker --stage confirmatory \
  --dataset outputs/followups/marker_confirmatory.jsonl \
  --scores outputs/followups/qwen_marker_confirmatory_scores.jsonl \
  --output outputs/followups/qwen_marker_confirmatory_analysis.json
```

The score command's `--resume` requires the exact original dataset, code, config, seed, freeze and decoding options. Preserve all failure sidecars and report failed development gates; do not generate test data before a sealed freeze exists. No results are added to the paper draft until actually scored outcomes exist. The design limitation to report is that the two Experiment 1 constructions remain syntactically and semantically different.
