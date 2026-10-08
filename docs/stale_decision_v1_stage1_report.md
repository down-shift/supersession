# Stage 1 delivery and verification

Stage 1 is complete. No model weights were downloaded or loaded, and no GPU pilot or confirmation inference was launched. CPU tests use fake/tiny models. The actual cached tokenizer was audited without downloading weights. Existing experiment sources, frozen results, provenance and manuscript sources were preserved. The pre-existing edit to scripts/fig_decomposition.py was left untouched.

The architecture audit, hypotheses, estimands, construction details, schema, full commands, selection restrictions, gate rules, bootstrap/prediction specifications, cost assumptions and scientific risks are in [stale_decision_v1.md](stale_decision_v1.md). Configuration is configs/stale_decision_v1/protocol.json. The new CLI is `python -m scripts.stale_decision_v1` and the independent modules live in src/{data,experiments,analysis}/stale_decision_v1.py. No new dependencies were added.

Authoritative local artifacts are in outputs/stale_decision_v1/stage1_final/:

- development.jsonl and its manifest: 192 concrete histories, 2,112 records; exactly balanced state/current-action combinations within each task/vocabulary, with independent entity orientation and historical/current orders.
- validation.json: structured data/pair checks passed.
- token_audit.json: actual pinned cached Qwen tokenizer; 6,336 prompt variants, 2,880 paired query-span checks, no candidate collisions or outside-span token changes; all paired edit token-length deltas zero. This audit was carried forward from an identical final dataset, with source hash and current code/config bindings recorded.
- freeze.json: binds the protocol text and implementation code plus immutable model/tokenizer configuration. It does not authorize inference.
- plan.json: development scoring <=78,144 forwards, two deterministic generations <=67,584 forwards, combined <=145,728; JSON planning estimate 25,344,000 bytes.

Earlier files directly under outputs/stale_decision_v1/ and stage1/ are preserved development drafts. Their dataset manifests may refer to earlier implementation hashes. Use stage1_final/ for subsequent work; do not overwrite those files. Only development data were written as experiment artifacts. Split-disjointness tests generate gate/confirmation structures in memory, without inference or experimental outcomes.

Actual downstream token counts (min / mean / max):

| Construction | Token counts |
| --- | --- |
| Superseded | 163 / 193.17 / 234 |
| Updated other | 165 / 194.17 / 234 |
| Entity mention | 165 / 194.17 / 234 |
| Unassigned | 155 / 184.17 / 224 |
| Current only | 137 / 155.50 / 174 |
| Live edit | 163 / 193.17 / 234 |

Equal edit lengths coexist with construction differences; token matching does not establish semantic isolation. Unassigned sentences remain shorter. Final label balance uses seeded block permutations and rotations rather than effect-based vocabulary selection.

## Checks

`.venv/bin/python -m pytest tests/test_stale_decision_v1.py -q --tb=short`: **20 passed in 37.29 s** on the final implementation. Tests cover deterministic balance, both tasks/all construction cells, incorrect ground truth, swapped historical labels, invalid policies/current states/spans/orders/templates, collisions and prefix instability, cross-split leakage including orientation invariance, missing/nonfinite scores, model separation, strict versus relaxed parsing, cap termination, signed synthetic contrasts, reproducible bootstrap, feasibility gates, grouped prediction, complete-label-plus-EOS likelihood, interrupted writes, resume manifests and exclusive artifact creation. Synthetic effects are test fixtures, not empirical findings.

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q`: **357 passed, 6 failed in 211.08 s**. The six failures are outside the new namespace:

- tests/test_relational_shell_workflow.py::test_preparation_hands_off_to_both_confirmations_with_original_runtime
- tests/test_relational_shell_workflow.py::test_qwen_score_failure_stops_its_analysis_but_runs_gemma
- tests/test_relational_shell_workflow.py::test_runner_reuses_bundles_and_resumes_only_matching_checkpoints
- tests/test_relational_shell_workflow.py::test_runner_validates_completed_outputs_without_rescoring
- tests/test_smoke.py::test_plan_ledger_parses_and_is_consistent
- tests/test_smoke.py::test_plan_has_a_frontier_or_is_human_gated

The four shell failures arise from sandbox-denied Bash process substitution (`/dev/fd/62: Operation not permitted`); the second assertion is a downstream consequence of the preparation script failing. The two smoke failures require absent docs/PLAN.md. No legacy scripts or missing historical ledgers were changed to make this new protocol pass.

`bash scripts/check.sh` was attempted: uv failed to open `/Users/daniel/.cache/uv/sdists-v9/.git` with `Operation not permitted (os error 1)`. Retry with `UV_CACHE_DIR=/tmp/stale_decision_v1_uv_cache UV_OFFLINE=1` failed inside uv 0.9.4's macOS system-configuration code: `Attempted to create a NULL object` / `Tokio executor failed`. Therefore the wrapper did not complete. Its test step was executed directly as above.

`bash scripts/build_paper.sh` was executed separately and failed with `LaTeX Error: File 'placeins.sty' not found.` No manuscript source was edited. Syntax parsing of all five new Python files, exact state/action counterbalance, dataset validation and final freeze/config/code/dataset hash consistency also passed.

Copies of command logs are preserved in outputs/stale_decision_v1/stage1_final/. The check failures are environmental/repository limitations, not passing checks.

## Feasibility and remaining risks

Frozen sizes are development 2,112, gate 4,224 and confirmation 16,896 records. Conservative total <=1,603,008 forwards/model, including deterministic direct and downstream generation. At an assumed 10–100 forwards/s, all stages require roughly 4.5–44.5 GPU hours/model, plus loading; this is an estimate, not a measured runtime. Reserve approximately 1 GB/model for JSON datasets/audits/scores/checkpoints; actual Stage 1 final artifacts occupy about 18 MB. An 8B FP16 parameter payload is approximately 16 GB decimal before runtime memory; the pinned reference config is unquantized. Its feasibility on the eventual GPU remains untested.

Open scientific risks: natural label priors and syntactic/length confounds; entity-name distribution shift in held-out splits; grid-associated policy rotations; small stratum cluster counts; canonical EOS event mass excluding other valid answer paths; the asymmetric four-state R_easy_local adaptation; same-split grouped prediction not establishing cross-template prediction; ceiling/floor behavior; technical/runtime uncertainty. All competence and behavioral gates are unknown until Stage 2. H3 can remain unavailable if there are no errors. For the live control, D uses fixed baseline-action orientation: correct tracking of the edited current state should produce a negative D_live, while strict generated accuracy should remain high on both members.

Stop here. Stage 2 model inference requires explicit authorization. No empirical downstream-effect, generated-vulnerability or predictive-validity outcome is claimed.
