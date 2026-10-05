# Time-limited relational follow-ups

These are separate from relational v2. Do not edit, relabel, or pool these records with v2 results. Any follow-up datasets generated before the freshness and factorial-validation fixes are archival design artifacts; do not score them or use them for development or confirmation.

## Existing infrastructure and protocol decisions

### Reproducible uv environments

Use the repository `uv.lock` and Python 3.13.5 for every follow-up command. Keep one environment per model, and reuse that model's environment for development, test, and confirmation scoring. This prevents dependency resolution or environment changes between stages. Example setup and test commands:

```bash
UV_PROJECT_ENVIRONMENT=.venv-qwen-followups uv sync --frozen --python 3.13.5 --extra model --extra dev
UV_PROJECT_ENVIRONMENT=.venv-gemma-followups uv sync --frozen --python 3.13.5 --extra model --extra dev
UV_PROJECT_ENVIRONMENT=.venv-phi4-mini-followups uv sync --frozen --python 3.13.5 --extra model --extra dev
UV_PROJECT_ENVIRONMENT=.venv-qwen-followups uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.run_followups --help
UV_PROJECT_ENVIRONMENT=.venv-gemma-followups uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.run_followups --help
UV_PROJECT_ENVIRONMENT=.venv-phi4-mini-followups uv run --frozen --python 3.13.5 --extra model --extra dev python -m scripts.run_followups --help
```

Apply the matching `UV_PROJECT_ENVIRONMENT` to every subsequent `uv run` command. Do not switch Python versions, install packages outside the lock, or reuse a score output across model environments. The score provenance records the resolved runtime. The host used to verify this note has no CUDA; its local test command used `UV_CACHE_DIR=/private/tmp/uv-cache-supersession uv run --no-sync --offline --frozen --python 3.13.5 --extra dev pytest -q tests/test_followups.py` because the sandbox blocks uv cache access. The `--no-sync` option was only needed for that already-synced test environment; experiment hosts should perform the frozen sync above.

### Amendment recorded before harder-task development

The post-pilot stopping rule is: before selecting a level or generating test data, if Qwen's complete-answer accuracy is above 90% at all three development levels (2, 4, and 6 distractors), stop Experiment 2. Report that the tested distractor manipulation did not create the intended difficulty. Do not run the nearest-to-0.775 fallback level, add levels, or redesign prompts. This amendment was recorded after the pilots and before development outcomes were available. If Qwen has a qualifying level, apply the original selection rule below unchanged. If Gemma exceeds 90% at the selected level, report it as near ceiling and not an informative behavioral replication.

### Verified execution status (2026-10-04)

The completed `rerun1` pilot artifacts are the authoritative pilots. Their score files, dataset hashes, score sidecars, and analysis provenance agree; both model runs report `complete` and use the pinned model revisions, greedy decoding, and the same 4-history/384-unique-prompt dataset (`pilot_harder_rerun1.jsonl`, 2 distractors, seed 20261006). Qwen (`Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`) scored 510/512 (99.6094%) with 2 stale answers, both in reversed-order entity-mention controls; there were no superseded errors or other answers. Gemma (`google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`) scored 512/512 (100%) with no stale answers. Measured inference times were 314.13 seconds (0.818 seconds/unique prompt) for Qwen and 870.81 seconds (2.268 seconds/unique prompt) for Gemma on an NVIDIA GeForce RTX 5080. These pilots validate the inference and scoring path; the near-ceiling results do not establish an informative behavioral test.

The marker pilot is exploratory only: 2 histories, 192 unique prompts, with model scores and analyses present for both models. Do not treat its interaction estimate as confirmatory. Older unscored preparation datasets and reports (`pilot_harder*.jsonl` and `pilot_marker*.jsonl` other than the `*_rerun1` files) are archival design artifacts and are excluded from all analyses. Preserve all pilot files; subsequent generation must explicitly exclude the authoritative rerun1 pilot datasets.

`tests/test_followups.py` passed under uv (9 passed). The local analysis host has PyTorch 2.14.0 but no CUDA device; model execution was performed on the remote NVIDIA GeForce RTX 5080. Complete n=2, n=4, and n=6 development datasets, scores, run records, provenance, and analyses are present for Qwen and Gemma, plus an additional exploratory Phi-4-mini scoring run. Each level has 12 histories, 1,536 scored members, and 1,152 unique prompts; n=4 and n=6 datasets pass validation and have zero rerun1-pilot overlap. All three models use the same target histories at every level. Qwen accuracy exceeds 90% at n=2, n=4, and n=6, so the prespecified Experiment 2 stopping rule applies: report that the tested distractor manipulation did not create the intended difficulty. No selection freeze or harder-task test was generated, and the nearest-level fallback must not be used. All score run records and sealed provenance are complete; n=4 resumed 31 Qwen and 88 Gemma unique prompts, while all other model/level runs resumed zero. Within each model, the n=2/n=4/n=6 runtime fingerprints, scoring seeds, model revisions, and code hashes match; recomputing the three Phi analyses from the saved datasets and score files exactly reproduces their saved summaries. Phi's strict exact-answer accuracy is only 8/1,536, 38/1,536, and 66/1,536 across the three levels, but most outputs begin with the expected candidate and continue with extra text; this post hoc formatting diagnostic makes its exact-match rates unsuitable for interpreting distractor difficulty. Phi was not part of the prespecified selection rule and no Phi test was generated. The independent 24-history marker confirmation is also complete for Qwen and Gemma: each has 3,072 scored members and 2,272 unique prompts, complete provenance, and an exactly recomputed analysis. Details and artifact hashes are in [Marker × construction confirmation](marker_construction_confirmation.md).

### Verified result ledger (2026-10-04)

All artifact paths below are relative to `outputs/followups/`. Older unscored preparation artifacts remain excluded; the tables below summarize the `*_rerun1` pilots and fresh development data at n=2, n=4, and n=6. The completed marker confirmation is reported separately in [Marker × construction confirmation](marker_construction_confirmation.md).

Accuracy denominators below count scored dataset members, including duplicated prompts across paired cells. The history bootstrap treats histories as the sampling unit. The harder pilot scored 384 unique prompts per model, the marker pilot 192, and each development level 1,152.

All n=2/n=4/n=6 development models used dataset seed 20261004 and scoring seed 20261006, with the same pinned model/tokenizer revisions and identical runtime fingerprints across levels within each model. Generation and analysis provenance explicitly record exclusion of `pilot_harder_rerun1.jsonl` (four full-history and four target-history signatures); all development datasets have zero overlap with those pilot signatures.

| Stage/model | Histories | Accuracy | Stale answers | Other answers | Superseded accuracy | Inference time |
|---|---:|---:|---:|---:|---:|---:|
| Rerun1 harder pilot, Qwen | 4 | 510/512 (99.6094%) | 2 | 0 | 100% | 314.13 s |
| Rerun1 harder pilot, Gemma | 4 | 512/512 (100%) | 0 | 0 | 100% | 870.81 s |
| Rerun1 marker pilot, Qwen | 2 | 256/256 (100%) | 0 | 0 | 100% | 154.30 s |
| Rerun1 marker pilot, Gemma | 2 | 255/256 (99.6094%) | 0 | 1 | 127/128 (99.2188%) | 430.10 s |
| Development n=2, Qwen | 12 | 99.6745% (1,531/1,536) | 4 | 1 | 100% | 2,116.35 s |
| Development n=2, Gemma | 12 | 99.8698% (1,534/1,536) | 0 | 2 | 100% | 3,541.05 s |
| Development n=4, Qwen | 12 | 99.7396% (1,532/1,536) | 3 | 1 | 100% | 936.32 s |
| Development n=4, Gemma | 12 | 99.7396% (1,532/1,536) | 0 | 4 | 100% | 2,398.50 s |
| Development n=6, Qwen | 12 | 99.8047% (1,533/1,536) | 3 | 0 | 100% | 979.23 s |
| Development n=6, Gemma | 12 | 99.7396% (1,532/1,536) | 0 | 4 | 100% | 2,561.02 s |
| Development n=2, Phi-4-mini | 12 | 0.5208% (8/1,536) | 0 | 1,528 | 1.0417% (4/384) | — |
| Development n=4, Phi-4-mini | 12 | 2.4740% (38/1,536) | 0 | 1,498 | 3.3854% (13/384) | — |
| Development n=6, Phi-4-mini | 12 | 4.2969% (66/1,536) | 0 | 1,470 | 7.0313% (27/384) | — |

Phi's strict exact-answer counts understate how often its initial output begins with the expected label: 1,516/1,536 at n=2, 1,530/1,536 at n=4, and 1,527/1,536 at n=6. These are post hoc string-prefix counts, not correctness under the frozen parser; the responses continue with extra text and remain classified as `other`. Thus Phi's low exact-match accuracy is dominated by response-format failure, and its apparent increase across levels is not interpretable as a distractor difficulty trend.

In the harder pilot, both Qwen stale answers occurred in reversed-order entity-mention controls; there were no superseded errors. In n=2 development, all 4 Qwen stale answers and its 1 other answer occurred in reversed-order entity-mention controls. The 96 reversed entity-mention baseline→edit pairs comprised 91 correct→correct, 2 stale→correct, 2 correct→stale, and 1 correct→other transitions. All 96 aligned entity-mention pairs were correct→correct. Gemma's two other answers occurred in reversed early-unassigned controls (2 other→correct transitions); it had no stale answers. These error counts do not demonstrate an effect of historical information. For Qwen and Gemma at both n=4 and n=6, all answer errors occurred in the reversed-order entity-mention condition; superseded, early-unassigned, and late-unassigned answers were all correct. Qwen had 3 stale and 1 other answer at n=4, then 3 stale answers at n=6. Gemma had 4 other answers and no stale answers at each level. Qwen n=4's three stale and one other response were correct→stale/other transitions; at n=6, one stale response changed to correct and one remained stale across its pair. Gemma n=4 had two other→other pairs; at n=6 it had two correct→other and two other→correct transitions. With only 12 histories, these few errors are descriptive and do not establish a distractor or history effect. Phi's exact-answer errors are summarized separately above because nearly all of its outputs continued beyond an initial candidate label.

For n=2, Qwen's history-bootstrap stale edit-change contrasts (2,000 draws, seed 73021) were superseded minus early-unassigned 0.000 [0.000, 0.000], superseded minus entity-mention 0.000 [−0.0208, 0.0260], and superseded minus late-unassigned 0.000 [0.000, 0.000]. Gemma's contrasts were all zero because there were no stale answers. Qwen's secondary v2 `R_all` means (95% history bootstrap intervals) were superseded 1.900 [1.634, 2.166], entity-mention 10.014 [8.489, 11.614], early-unassigned 0.196 [−0.201, 0.607], and late-unassigned −0.095 [−0.854, 0.703]. Gemma's were 1.285 [0.942, 1.657], 2.557 [2.232, 2.936], −0.027 [−0.401, 0.366], and 0.108 [−0.470, 0.687], respectively. These score metrics are secondary and do not establish a behavioral effect. At n=4/n=6, Qwen's `R_all` means (95% history-bootstrap intervals) were superseded 1.937 [1.663, 2.189]/2.104 [1.773, 2.438], entity-mention 8.083 [6.767, 9.436]/7.628 [6.315, 9.056], early-unassigned 0.210 [−0.143, 0.569]/0.085 [−0.288, 0.450], and late-unassigned −0.318 [−0.951, 0.325]/0.086 [−0.837, 0.976]. Gemma's n=4/n=6 estimates were superseded 1.172 [0.933, 1.426]/1.298 [1.002, 1.653], entity-mention 3.074 [2.547, 3.615]/3.663 [2.908, 4.442], early-unassigned 0.166 [−0.245, 0.556]/−0.112 [−0.479, 0.211], and late-unassigned 0.252 [−0.312, 0.862]/0.100 [−0.397, 0.612]. The near-zero unassigned averages again conceal positive aligned and negative reversed estimates. These 12-history development estimates are descriptive; the high complete-answer accuracy, not these secondary score contrasts, determines the amended stop decision.

The candidate current-minus-historical margin edit-change means and 95% history-bootstrap intervals were, for Qwen: superseded +0.725 [−0.168, 1.592], entity-mention −0.343 [−1.816, 1.069], early-unassigned +0.093 [−0.838, 0.893], and late-unassigned +0.361 [−0.325, 0.964]. For Gemma they were −0.644 [−1.417, 0.046], −0.484 [−1.197, 0.242], −0.325 [−0.787, 0.092], and −0.412 [−0.923, 0.181], respectively. Every interval includes zero.

The marker pilot has only 2 histories per model and is exploratory. Qwen answered all 256 scored members correctly (192 unique prompts). Gemma's single other answer occurred in the superseded, marker-absent, aligned stratum; there were no stale answers for either model. The marker effect is marker-present minus marker-absent; interaction is the difference between the superseded and entity-mention marker effects. Values below are means with 95% history-bootstrap intervals (2,000 draws, seed 73021):

| Model | Order stratum | Superseded marker effect | Entity-mention marker effect | Interaction |
|---|---|---:|---:|---:|
| Qwen | All | −0.580 [−0.618, −0.541] | −2.169 [−2.972, −1.367] | 1.589 [0.826, 2.353] |
| Qwen | Aligned | −0.606 [−0.855, −0.356] | −1.434 [−1.723, −1.146] | 0.829 [0.790, 0.867] |
| Qwen | Reversed | −0.554 [−0.726, −0.381] | −2.904 [−4.221, −1.587] | 2.350 [0.861, 3.839] |
| Gemma | All | −0.874 [−1.268, −0.481] | −1.031 [−1.429, −0.632] | 0.157 [−0.635, 0.949] |
| Gemma | Aligned | −0.365 [−0.455, −0.276] | −1.082 [−1.337, −0.828] | 0.717 [0.552, 0.882] |
| Gemma | Reversed | −1.383 [−2.259, −0.506] | −0.979 [−1.522, −0.437] | −0.403 [−1.823, 1.016] |

The pilot's component-level E summaries are retained in the analysis JSON as per-edit/per-query records. For a compact audit summary, the table gives means over those records by construction, marker state (0 absent, 1 present), and aligned/reversed history-current order; each cell is E / replacement component / source component:

| Model | Construction | Marker | Aligned | Reversed |
|---|---|---:|---:|---:|
| Qwen | Superseded | 0 | 4.533 / 0.830 / −3.702 | 5.454 / 3.077 / −2.377 |
| Qwen | Superseded | 1 | 3.296 / 2.044 / −1.252 | 3.197 / 1.060 / −2.136 |
| Qwen | Entity-mention | 0 | 8.260 / 3.949 / −4.311 | 10.707 / 5.654 / −5.053 |
| Qwen | Entity-mention | 1 | 5.779 / 2.599 / −3.181 | 7.372 / 4.268 / −3.104 |
| Gemma | Superseded | 0 | 6.086 / 1.389 / −4.697 | 6.904 / 1.579 / −5.325 |
| Gemma | Superseded | 1 | 2.596 / 0.627 / −1.968 | 2.866 / 1.131 / −1.735 |
| Gemma | Entity-mention | 0 | 5.553 / 1.594 / −3.959 | 4.704 / 1.864 / −2.840 |
| Gemma | Entity-mention | 1 | 4.656 / 2.138 / −2.518 | 4.476 / 2.783 / −1.693 |

The construction contrast does not isolate conflict from syntax. The two-history pilot estimates remain exploratory and are excluded from the now-completed 24-history marker × construction confirmation. Confirmation found negative within-construction marker effects in both models. Qwen’s overall interaction was +1.337 [0.976, 1.712]; Gemma’s was −0.125 [−0.390, 0.136] and remains inconclusive. Confirmation used candidate scoring only, so there are no confirmation accuracy or stale-answer outcomes. Recorded runtimes were 1,551.08 seconds for Qwen and 657.75 seconds for Gemma. See the separate result document for E/R components, order strata, and verification.

The full pilot result files are the scored pilot JSONLs themselves (`qwen_harder_pilot_rerun1.jsonl`, `gemma_harder_pilot_rerun1.jsonl`, and their marker-pilot equivalents), with matching `.provenance.json` and `.run.json` sidecars and analysis JSONs. Development datasets/reports are `development_n{2,4,6}.jsonl` and matching report JSONs; per-model score, provenance, run, and analysis files use `qwen_development_n*_*`, `gemma_development_n*_*`, and `phi4_mini_development_n*_*`. Verified dataset/score SHA-256 values are:

| Stage/model | Dataset SHA-256 | Model score SHA-256 |
|---|---|---|
| Harder rerun1, Qwen | `b282a4a1362b11acbd6be523d3898b5ccb77a5e26d83b94c5a13b499502b528d` | `845eb0e4b454c4c0a8a8cc85de9c71c877c96dd831e6a06e5645d9871d685c11` |
| Harder rerun1, Gemma | `b282a4a1362b11acbd6be523d3898b5ccb77a5e26d83b94c5a13b499502b528d` | `3475b1dfa835d80005389871d24a82a45aa7214e646e89bbdd11e14eff58c81e` |
| Marker rerun1, Qwen | `f42cc115be1715b46d70587fd5a8e311eed33c8868e34d516e1454774aa4fec4` | `37936c1f65d30156fed4de87371f464aa5129d95bfea6ffc93a9036c3539d405` |
| Marker rerun1, Gemma | `f42cc115be1715b46d70587fd5a8e311eed33c8868e34d516e1454774aa4fec4` | `f5fa07544df77f2f794abef91de608b9b5eeb64d0893589c27dc132b7c15bd3e` |
| Development n=2, Qwen | `26e01f7f5c8992d49178d7da4fa9136a06247c2d964d2ac558a4a071a7efc25f` | `0f224cb78efcb2013bba10b52dc6390e56d544759eb26ca3de07fb47317296f4` |
| Development n=2, Gemma | `26e01f7f5c8992d49178d7da4fa9136a06247c2d964d2ac558a4a071a7efc25f` | `6abbac7d70defc7c684a6f3cfd89f689680ad450e165bae07fac5bf8721e2ff5` |
| Development n=4, Qwen | `3ad88e333b3188eb4481785c0441ef239b2e6cc8f4c707bb2b3883bd1520b2a0` | `1efc4aca4ca913f1f781a591bf19c125289f37895fdd5db553a96cb71c4e358b` |
| Development n=4, Gemma | `3ad88e333b3188eb4481785c0441ef239b2e6cc8f4c707bb2b3883bd1520b2a0` | `972b2194c14f78b7e0d915831d2b1cd91507b48ccc0a1b3221b241f4c964dde9` |
| Development n=6, Qwen | `9c1085ab5b88810e7ce5ec9ffe7e65bdcd283c6aa06c79a80e0d1f45fc8c2053` | `43299b6a476a9406ffcc52c7f3aed764c90017efc73ffc4a915ceda07f3c97e3` |
| Development n=6, Gemma | `9c1085ab5b88810e7ce5ec9ffe7e65bdcd283c6aa06c79a80e0d1f45fc8c2053` | `2f665f13584e126272165d8f61af32b1f767ef690fa8af0c75652e4ebbe48b72` |
| Development n=2, Phi-4-mini | `26e01f7f5c8992d49178d7da4fa9136a06247c2d964d2ac558a4a071a7efc25f` | `ef3182ee3c32f3e91a46755e8aac12f089ce057741caa09698fc1145aa7739e9` |
| Development n=4, Phi-4-mini | `3ad88e333b3188eb4481785c0441ef239b2e6cc8f4c707bb2b3883bd1520b2a0` | `79689399ad50ee2db8864e7a07a6e84a1084a02719807768acf7dea30de2b3c1` |
| Development n=6, Phi-4-mini | `9c1085ab5b88810e7ce5ec9ffe7e65bdcd283c6aa06c79a80e0d1f45fc8c2053` | `cd1e5e5f6d808757b623c2743d9cbd5159737189030634a4b18b28fd3ca32709` |

Phi analysis SHA-256 values: n=2 `5726df7a1ed263f0e45eedb9f384faf4910089a2e17b0f21637ce3f4d2d62c46`; n=4 `6ac7fa07818ae60e6ce5b503027ff84f46f05ff9271f883f84501eb9ecf11051`; n=6 `543ec076e03dcdd1a8ed7c33618bd06d8bf52f074a905202d7c0ae1c8fe6f6f4`.

Harder-pilot inference times were 0.818 s/prompt for Qwen and 2.268 s/prompt for Gemma; marker-pilot times were 0.804 and 2.240 s/prompt. Development times per unique prompt were n=2: 1.837/3.074 s; n=4: 0.835/2.254 s; n=6: 0.850/2.223 s for Qwen/Gemma, respectively. At n=4, Qwen resumed 31 and Gemma 88 unique prompts from matching partial checkpoints; these were included in each complete final score file. Timings describe recorded RTX 5080 runs and are not portable runtime guarantees.

The generator reuses the v2 candidate vocabulary and the same current assignments, entity pair, query structure, paired historical-value edit, and history-level estimand logic. V2 candidate likelihood scoring is the preferred scoring method for E/R and current-versus-historical margins. Existing history bootstrap convention is 2,000 draws with seed 73021.

V2 always marks historical assignments with `Previously`; this follow-up crosses that marker with assigned superseded history versus unrelated entity mention. Within each construction, marker presence is varied with the remaining cell content held fixed. The construction contrast combines differences in syntax and assignment semantics and does not isolate conflict from syntax. Report both within-construction marker effects and their difference-in-differences with history bootstrap intervals. A CI crossing zero is inconclusive, not evidence of equivalence. Report the source/replacement E components contributing to R.

V2 scores candidate continuations and imposes a ≥99% competence gate. This follow-up has no competence gate. Its runner records deterministic complete answers with exact parsing, all-history transition counts, and candidate continuation scores. The parser contract is frozen here: strip whitespace and edge punctuation, then exact case-insensitive candidate match; every other string is `other`.

## Experiment 1: marker × construction

Templates are validated in code using the supplied examples. Generated rows cross construction (`superseded`, `entity_mention`), marker (0/1), historical order (0/1), current order (0/1), edited entity (x/z), query entity (x/z), and baseline/edit direction. Histories, current assignments, value candidates, questions, and edit values are held fixed across the four marker/construction cells. Both aligned `(historical_order == current_order)` and reversed order are retained.

For each history and cell, compute E from the paired candidate scores using the existing v2 identity-transfer definition. Compute R from the two query-specific E means for each edited entity, then symmetrize x/z. Report the two marker effects (with minus without marker) within each construction, the marker × construction interaction, E's source and replacement log-mass components, and aligned/reversed strata. Bootstrap histories, not rows. Do not claim active suppression by conflict.

## Experiment 2: harder histories

Difficulty levels are prespecified as 2, 4, and 6 distractor entities, with 12 fresh, matched target histories per development level and explicit rerun1 pilot exclusions. Complete all three Qwen levels and apply the stopping amendment above first. If the stop rule does not apply, Qwen development chooses the hardest level whose complete-answer accuracy is between 0.65 and 0.90 inclusive; if none qualifies, record the level nearest 0.775 (ties go to the smaller level) for reporting only and skip the harder-task test. For a qualifying level, freeze that level, prompt, candidate values, parser, metrics, and analysis before generating the separate 24-history test, explicitly excluding the pilot and all development histories. Development and test use disjoint seeded history namespaces. Generate a balanced, fixed allocation before model outcomes; evaluate every test history.

Each history includes superseded, entity-mention, early-unassigned, and late-unassigned conditions, both query entities, both history/current order factors, each edited entity, and paired baseline/edit prompts. The edited history changes only the historical value; the answer and current assignment remain fixed. Report complete-answer accuracy, stale-answer frequency, all paired correct/stale/other transitions, candidate-score current-minus-historical margins, and v2 R secondarily. Estimate edit effects on stale-answer frequency and the difference of that effect between superseded and each matched control. More errors under harder prompts alone do not establish a historical-information cause.

Use deterministic greedy decoding and the parser above. No ≥99% competence gate applies. No test-only error selection or difficulty tuning is allowed. Record model/revision, runtime, seed, dataset/code hashes, exclusions, and every failed development rule. Qwen remains the difficulty-selection model; Gemma is the planned paired model. Phi-4 Mini is an additional exploratory model and cannot select a level or authorize a harder-task test.

## Running the follow-ups

Run these commands from the repository checkout on the CUDA host after copying the updated code and `uv.lock`. The scripts create separate frozen uv environments per model and keep the pilot artifacts untouched.

The marker × construction confirmation is the highest experimental priority. To run only that study, use:

```bash
bash scripts/run_marker_confirmation.sh
```

If the marker confirmation dataset is absent, this creates 24 fresh histories with an explicit marker-pilot exclusion; it then scores Qwen, Gemma, and Phi-4 Mini sequentially. It reuses complete matching score artifacts, resumes matching partial checkpoints, and does not run the harder-task test. Use `bash scripts/followups_gpu.sh confirm phi4_mini` to score only Phi-4 Mini against the existing marker confirmation dataset.

The original Qwen development launch was:

```bash
bash scripts/followups_gpu.sh develop qwen
```

Qwen development is complete at all three levels and meets the amended stop rule. Do not rerun Qwen to select a level or create a harder-task test. For supplementary Phi-4 Mini development accuracy at the same 2/4/6 distractor levels, after the shared development datasets and reports are present, run:

```bash
bash scripts/followups_gpu.sh develop phi4_mini
```

This produces `phi4_mini_development_n{2,4,6}_scores.jsonl`, matching provenance/run sidecars, and analyses. It does not run the Qwen difficulty selector or generate a test dataset. The Phi-4 Mini config is pinned in `configs/cross_model_relational_v2/phi4_mini.yaml`; inference uses the existing Phi-4 compatibility loader and its separate frozen uv environment.

Gemma development is also complete. The old paired development and harder-task confirmation commands were:

```bash
bash scripts/followups_gpu.sh develop gemma
bash scripts/followups_gpu.sh confirm qwen
```

`confirm qwen` and `confirm gemma` independently run the marker study and only run a harder-task test when a qualifying Qwen freeze exists. No such freeze exists under the completed stop rule. For Phi-4 Mini marker scoring, use the command above; Phi-4 Mini cannot enter the harder-task selection or test path.

The Gemma marker command is:

```bash
bash scripts/followups_gpu.sh confirm gemma
```

If Qwen is above 90% at all three levels, development prints the required stop decision and creates no test freeze. If Qwen has no level in [0.65,0.90], the freeze records the prespecified nearest-level fallback for reporting, but the harder test is skipped. If Gemma exceeds 90% at the selected level, report it as near ceiling. The harder-task script preserves failed score checkpoints for exact-runtime resume and refuses incomplete dataset/report pairs.

Use fresh output names if an earlier partial run left an incomplete artifact pair. The script records outputs under `outputs/followups/`, including each dataset report, model score provenance, and analysis. Update the execution status here after the runs with accuracies, the stopping/selection decision, freeze hash if one exists, confirmation results, and runtime. Preserve the v2 results and paper.
