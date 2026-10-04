# Time-limited relational follow-ups

These are separate from relational v2. Do not edit, relabel, or pool these records with v2 results. The two pilot JSONL files and reports under `outputs/followups/` are ignored output artifacts; preserve them as generated.

## Existing infrastructure and protocol decisions

The generator reuses the v2 candidate vocabulary and the same current assignments, entity pair, query structure, paired historical-value edit, and history-level estimand logic. V2 candidate likelihood scoring is the preferred scoring method for E/R and current-versus-historical margins. Existing history bootstrap convention is 2,000 draws with seed 73021.

V2 always marks historical assignments with `Previously`; this follow-up crosses that marker with assigned superseded history versus unrelated entity mention. It does not identify a pure marker effect independent of construction, because assignment and mention differ in syntax and meaning. Report both within-construction marker effects and their difference-in-differences with history bootstrap intervals. A CI crossing zero is inconclusive, not evidence of equivalence. Report the source/replacement E components contributing to R.

V2 scores candidate continuations and imposes a ≥99% competence gate. The harder follow-up instead needs deterministic complete-answer generation, exact parsing, all-history transition counts, and has no competence gate. The existing runner does not implement that generation/parser outcome path, so no test-set inference is authorized by this preparation. The parser contract is frozen here: strip whitespace and edge punctuation, then exact case-insensitive candidate match; every other string is `other`.

## Experiment 1: marker × construction

Templates are validated in code using the supplied examples. Generated rows cross construction (`superseded`, `entity_mention`), marker (0/1), historical order (0/1), current order (0/1), edited entity (x/z), query entity (x/z), and baseline/edit direction. Histories, current assignments, value candidates, questions, and edit values are held fixed across the four marker/construction cells. Both aligned `(historical_order == current_order)` and reversed order are retained.

For each history and cell, compute E from the paired candidate scores using the existing v2 identity-transfer definition. Compute R from the two query-specific E means for each edited entity, then symmetrize x/z. Report the two marker effects (with minus without marker) within each construction, the marker × construction interaction, E's source and replacement log-mass components, and aligned/reversed strata. Bootstrap histories, not rows. Do not claim active suppression by conflict.

## Experiment 2: harder histories

Difficulty levels are prespecified as 2, 4, and 6 distractor entities. Development chooses the hardest level whose complete-answer accuracy is between 0.65 and 0.90 inclusive; if none qualifies, choose the level nearest 0.775 (ties go to the smaller level). Freeze that level, prompt, candidate values, parser, metrics, and analysis before generating test rows. Development and test use disjoint seeded history namespaces. Generate a balanced, fixed allocation before model outcomes; evaluate every test history.

Each history includes superseded, entity-mention, early-unassigned, and late-unassigned conditions, both query entities, and paired baseline/edit prompts. The edited history changes only the historical value; the answer and current assignment remain fixed. Report complete-answer accuracy, stale-answer frequency, all paired correct/stale/other transitions, candidate-score current-minus-historical margins, and v2 R secondarily. Estimate edit effects on stale-answer frequency and the difference of that effect between superseded and each matched control. More errors under harder prompts alone do not establish a historical-information cause.

Use deterministic greedy decoding and the parser above. No ≥99% competence gate applies. No test-only error selection or difficulty tuning is allowed. Record model/revision, runtime, seed, dataset/code hashes, exclusions, and every failed development rule. Qwen and Gemma are the intended model pair; if only Qwen can run, mark Gemma pending.

## Commands and current pilot status

Run tests and design/storage pilots with the repository environment:

```bash
uv run pytest -q tests/test_followups.py
uv run python -m scripts.followups marker --stage pilot --histories 2 \
  --output outputs/followups/pilot_marker.jsonl \
  --report outputs/followups/pilot_marker_report.json
uv run python -m scripts.followups harder --stage development --histories 4 --distractors 2 \
  --output outputs/followups/pilot_harder.jsonl \
  --report outputs/followups/pilot_harder_report.json
```

The recorded marker design/storage pilot has 2 histories, 256 members, 192 unique prompts, and a 294,528-byte JSONL dataset. Its full design uses 96 unique prompts per history; a 24-history evaluation would have about 2,304 unique prompts. The harder pilot at 2 distractors has 4 histories, 64 members, 64 unique prompts, and a 65,476-byte JSONL dataset. Under the planned 12 development histories per level and 24 test histories, the harder experiment would score 576 development and 384 test prompts for one selected level. Across both experiments and both models, this is about 6,528 unique prompt evaluations before retries; longer prompts make v2 throughput an optimistic runtime proxy. The saved v2 inference artifacts do not provide a trustworthy elapsed-time measure for these longer prompts.

Template and prefix-only audits pass. These are design/storage pilots only; they have no model outcomes and do not establish runtime. Local `.venv` has PyTorch 2.14.0 but `torch.cuda.is_available()` is false; the system Python has no PyTorch, and `uv run` could not read the uv sdists cache. Therefore no inference smoke/pilot ran, no difficulty was selected, and no confirmatory test run is ready. The large run remains gated on a successful target-runtime inference pilot and implementation of complete-answer scoring, deterministic generation, the frozen analysis, and provenance checks.

No results are added to the paper draft until actual scored outcomes exist. The design limitation to report is that the two Experiment 1 constructions remain syntactically and semantically different.
