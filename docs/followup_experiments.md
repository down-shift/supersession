# Time-limited relational follow-ups

These are separate from relational v2. Do not edit, relabel, or pool these records with v2 results. Any follow-up datasets generated before the freshness and factorial-validation fixes are archival design artifacts; do not score them or use them for development or confirmation.

## Existing infrastructure and protocol decisions

### Reproducible uv environments

Use the repository `uv.lock` and Python 3.13.5 for every follow-up command. Keep one environment per model, and reuse that model's environment for development, test, and confirmation scoring. This prevents dependency resolution or environment changes between stages. Example setup and test commands:

```bash
UV_PROJECT_ENVIRONMENT=.venv-qwen-followups uv sync --frozen --python 3.13.5 --extra model --extra dev
UV_PROJECT_ENVIRONMENT=.venv-gemma-followups uv sync --frozen --python 3.13.5 --extra model --extra dev
UV_PROJECT_ENVIRONMENT=.venv-qwen-followups uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.run_followups --help
UV_PROJECT_ENVIRONMENT=.venv-gemma-followups uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.run_followups --help
```

Apply the matching `UV_PROJECT_ENVIRONMENT` to every subsequent `uv run` command. Do not switch Python versions, install packages outside the lock, or reuse a score output across model environments. The score provenance records the resolved runtime. The host used to verify this note has no CUDA; its local test command used `UV_CACHE_DIR=/private/tmp/uv-cache-supersession uv run --no-sync --offline --frozen --python 3.13.5 --extra dev pytest -q tests/test_followups.py` because the sandbox blocks uv cache access. The `--no-sync` option was only needed for that already-synced test environment; experiment hosts should perform the frozen sync above.

### Amendment recorded before harder-task development

The post-pilot stopping rule is: before selecting a level or generating test data, if Qwen's complete-answer accuracy is above 90% at all three development levels (2, 4, and 6 distractors), stop Experiment 2. Report that the tested distractor manipulation did not create the intended difficulty. Do not run the nearest-to-0.775 fallback level, add levels, or redesign prompts. This amendment was recorded after the pilots and before development outcomes were available. If Qwen has a qualifying level, apply the original selection rule below unchanged. If Gemma exceeds 90% at the selected level, report it as near ceiling and not an informative behavioral replication.

### Verified execution status (2026-10-04)

The completed `rerun1` pilot artifacts are the authoritative pilots. Their score files, dataset hashes, score sidecars, and analysis provenance agree; both model runs report `complete` and use the pinned model revisions, greedy decoding, and the same 4-history/384-unique-prompt dataset (`pilot_harder_rerun1.jsonl`, 2 distractors, seed 20261006). Qwen (`Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`) scored 510/512 (99.6094%) with 2 stale answers, both in reversed-order entity-mention controls; there were no superseded errors or other answers. Gemma (`google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`) scored 512/512 (100%) with no stale answers. Measured inference times were 314.13 seconds (0.818 seconds/unique prompt) for Qwen and 870.81 seconds (2.268 seconds/unique prompt) for Gemma on an NVIDIA GeForce RTX 5080. These pilots validate the inference and scoring path; the near-ceiling results do not establish an informative behavioral test.

The marker pilot is exploratory only: 2 histories, 192 unique prompts, with model scores and analyses present for both models. Do not treat its interaction estimate as confirmatory. Older unscored preparation datasets and reports (`pilot_harder*.jsonl` and `pilot_marker*.jsonl` other than the `*_rerun1` files) are archival design artifacts and are excluded from all analyses. Preserve all pilot files; subsequent generation must explicitly exclude the authoritative rerun1 pilot datasets.

`tests/test_followups.py` passed under uv (9 passed). This workspace has PyTorch 2.14.0 but no CUDA device; its frozen Qwen and Gemma configurations require full-GPU execution. Therefore development, harder-task test, and 24-history marker confirmation have not run here. Do not substitute different precision, device, or model settings. On the pilot GPU, development is estimated at about 47 minutes for Qwen and 2 hours 11 minutes for Gemma; marker confirmation at about 31 minutes for Qwen and 87 minutes for Gemma. No protocol freeze exists because development has not run, and no test dataset should be generated until Qwen development supports selection. Review-only design validation confirmed 12 matched target histories at all three levels, 1,536 members and 1,152 unique prompts per level, and zero rerun1-pilot overlap. The 24-history marker confirmation design has 3,072 members and 2,272 unique prompts after deduplication, with zero marker-pilot overlap; no confirmation artifact or inference outcome was produced by that check.

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

## Running the follow-ups

Run these commands from the repository checkout on the CUDA host after copying the updated code and `uv.lock`. The script creates separate frozen uv environments per model and keeps the pilot artifacts untouched. It generates all three development levels before scoring and applies the stopping/selection rules automatically.

```bash
bash scripts/followups_gpu.sh develop qwen
```

If Qwen has a qualifying level, run Gemma development when runtime permits, then run Qwen confirmation:

```bash
bash scripts/followups_gpu.sh develop gemma
bash scripts/followups_gpu.sh confirm qwen
```

Run Gemma confirmation when runtime permits:

```bash
bash scripts/followups_gpu.sh confirm gemma
```

If Qwen is above 90% at all three levels, development prints the required stop decision and creates no test freeze. Run `confirm qwen` for the independent marker confirmation; the script skips the harder test. If Qwen has no level in [0.65,0.90], the freeze records the prespecified nearest-level fallback for reporting, but the harder test is skipped. If Gemma exceeds 90% at the selected level, report it as near ceiling. The script preserves failed score checkpoints for exact-runtime resume and refuses incomplete dataset/report pairs.

Use fresh output names if an earlier partial run left an incomplete artifact pair. The script records outputs under `outputs/followups/`, including each dataset report, model score provenance, and analysis. Update the execution status here after the runs with accuracies, the stopping/selection decision, freeze hash if one exists, confirmation results, and runtime. Preserve the v2 results and paper.
