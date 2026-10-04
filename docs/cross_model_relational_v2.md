# Relational/order v2 protocol

## Geometry validation implementation migration (2026-10-03)

The first Gemma confirmation dataset contained 96 histories / 18,432 members, but scoring stopped in `dataset_info()` before model loading. The saved geometry deduplicated identical rendered prompts while incorrectly reusing abstract `x`/`z` span labels across opposite orientations. That dataset and all earlier artifacts remain preserved under `factorial_relation_counterbalanced_20261003`.

The corrected validator stores a per-example semantic-field-to-physical-span map. The unchanged scientific design was first sealed as implementation revision `factorial_relation_counterbalanced_geometryfix_20261003`; freeze `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_20261003_r5/protocol_freeze.json` binds geometry schema 2. After importing and hashing the stopped Gemma confirmation bundle, a separate exclusion revision was frozen at `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/protocol_freeze.json`. Its ledger adds the bundle's 96 donor-independent physical histories and records the source dataset, geometry, provenance and stopped-record hashes. The original and r5 freezes and all source artifacts remain unchanged. Stage claims include the implementation hash, preventing collisions with claims from earlier code revisions.

The migration command authenticates source score/dataset/candidate hashes and their original freeze, verifies recorded inference settings, and checks the original score metadata and saved surface token events before rebinding rows. It then reaudits all rows with the model tokenizer, checks candidate events, tokenizer/chat-template hashes, row semantics, and rendered prompts against the old saved score rows, and writes new dataset, geometry, score, and migration provenance files. It records source hashes, original inference revision, corrected validation revision and reason. The original inference provenance is preserved even when the source sidecar was previously migrated. Saved scores remain characterized as scores from their original inference run. No model is loaded by this migration command.

Under the exclusion freeze, Qwen and Phi tokenizer-only validation completed locally; Gemma's validation artifact was produced on the machine with its pinned tokenizer and copied here. All three artifacts bind to the exclusion freeze. Gemma passes the surface, paired-edit, exhaustive substitution and geometry checks: 5,376 substitutions, 384 edit pairs, 768 example mappings over 528 deduplicated prompts. Its events, tokenizer hash and chat-template hash match the prior Gemma candidate map. Qwen, Phi and Gemma saved development/gate score evaluations were migrated and exactly match their earlier reports. Qwen and Gemma's recomputed gates and preflights pass; Phi's gate fails `entity_mention` accuracy and has no successful preflight. Qwen's first confirmation score startup stopped before model loading because the runner compared migration-environment provenance (Python 3.13.5) instead of the original inference environment. The later runtime fix resolved original-inference provenance. That exclusion-freeze status is historical; completed runtimefix confirmation results are recorded below.

Example migration commands for Qwen are recorded here; use the matching `gemma3_4b` config and paths for Gemma. Migrate development first, analyze it, then migrate gate against that report, analyze the gate and create preflight. Use fresh, unused paths for each command. These commands use Python 3.13.5 as the migration-tool environment; that is distinct from the saved original inference runtime in `original_inference_provenance`. Include the `model` extra:

```bash
# Completed on 2026-10-04; outputs already exist. Do not rerun at these paths.
REV=outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003
OLD=outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/qwen3_8b
NEW=$REV/qwen3_8b
CONFIG=configs/cross_model_relational_v2_geometryfix_exclusions/qwen3_8b.yaml
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --frozen --extra model --python 3.13.5

uv run --frozen --extra model --python 3.13.5 python -m scripts.robustness_v2 migrate --stage development --config "$CONFIG" \
  --candidates "$NEW/candidates.json" --original-candidates "$OLD/candidates.json" \
  --dataset "$OLD/development.jsonl" --scores "$OLD/development_scores.jsonl" \
  --output "$NEW/migrated/development.jsonl" --local-files-only
uv run --frozen --extra model --python 3.13.5 python -m scripts.robustness_v2 analyze --stage development --config "$CONFIG" \
  --candidates "$NEW/candidates.json" --dataset "$NEW/migrated/development.jsonl" \
  --scores "$NEW/migrated/development.jsonl.scores.jsonl" \
  --output "$NEW/migrated/development_report.json"
uv run --frozen --extra model --python 3.13.5 python -m scripts.robustness_v2 migrate --stage frozen_gate --config "$CONFIG" \
  --candidates "$NEW/candidates.json" --original-candidates "$OLD/candidates.json" \
  --dataset "$OLD/gate.jsonl" --scores "$OLD/gate_scores.jsonl" \
  --prior-dataset "$NEW/migrated/development.jsonl" \
  --development-report "$NEW/migrated/development_report.json" \
  --output "$NEW/migrated/gate.jsonl" --local-files-only
uv run --frozen --extra model --python 3.13.5 python -m scripts.robustness_v2 analyze --stage frozen_gate --config "$CONFIG" \
  --candidates "$NEW/candidates.json" --dataset "$NEW/migrated/gate.jsonl" \
  --scores "$NEW/migrated/gate.jsonl.scores.jsonl" \
  --development-report "$NEW/migrated/development_report.json" \
  --output "$NEW/migrated/gate_report.json"
uv run --frozen --extra model --python 3.13.5 python -m scripts.robustness_v2 preflight --config "$CONFIG" \
  --candidates "$NEW/candidates.json" --gate "$NEW/migrated/gate_report.json" \
  --output "$NEW/migrated/preflight.json"
```

The Qwen, Phi and Gemma migrations under the exclusion freeze were validated from saved development/gate scores. Their recomputed development and gate evaluations exactly match the earlier reports. Qwen and Gemma's migrated gates and preflights pass; Phi's migrated gate fails `entity_mention` competence. Gemma's migration records the original inference revision and source hashes. This paragraph describes the saved-score migration stage; it is superseded for confirmation status by the runtimefix completion and analysis record below.

**Archived status for the original freeze:** `factorial_relation_counterbalanced_20261003` was the initial corrected design revision. Development and gate scoring did occur for Qwen, Gemma and Phi. The Gemma confirmation dataset was generated under that earlier freeze; score validation stopped before model loading, and no confirmation score file exists. The failed bundle is preserved under the r5 `migration_source/gemma3_4b/` directory and its histories are excluded by the newer freeze. No remote host was accessed in this work. This is a repository-frozen protocol, not externally preregistered.

The earlier `relational_order_correction_20261003` freeze and the intermediate `factorial_coherent_controls_20261003` preparation freeze, candidate maps and previews remain unchanged. Qwen and Phi development datasets in the earlier directory cite the first freeze and are not valid development evidence for this design. The intermediate candidate maps cite a relation-confounded fractional allocation and are not valid for this final freeze. All partial outputs remain preserved. A read-only process check found no active Qwen/Phi v2 jobs on 2026-10-03.

## Frozen design

The implementation and allocation are in `src/cross_model/robustness_v2.py`; the contract and model lineage checks are in `src/cross_model/robustness_protocol.py`. The exclusion-freeze migrated development/gate artifacts are under `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/`. The completed runtimefix confirmations and analyses are under `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004/`; configs and the physical-history exclusion ledger are under `configs/cross_model_relational_v2_geometryfix_exclusions/`. Earlier freeze directories remain archived separately.

The shared candidate values are `amber coral jade pearl slate teal violet ivory`. Each history samples six distinct identities: four historical/current assigned values and two intervention donors outside that set. The six fixed entity pairs are Nora/Liam, Ava/Omar, Mila/Eli, Iris/Noah, Zoe/Theo and Maya/Leo. Semantic orientation maps x/z to each pair in both directions. Historical entity/mention order and current entity order independently retain both levels in every history; all member cells cross condition, order, edited entity, query and baseline/edit direction.

### Allocation

The allocation table is explicit in the contract and asserted against generated histories. Confirmation has 96 histories: each of the 6 entity pairs × 4 target attributes (`badge`, `color`, `code`, `label`) × 2 semantic orientations has exactly two histories. Margins are 16 histories per pair, 24 per target attribute and 48 per orientation.

Development and gate each use the same declared 24-row fractional table. Every entity pair occurs four times, each target attribute six times, each orientation twelve times, and each pair × attribute occurs once. Orientation is balanced within each pair. At 24 histories, pair × orientation is only partially crossed and pair × attribute × orientation cannot be fully crossed; each pair × attribute has one history. The frozen fractional table selects two `team` and two `project` assignments within each pair, and three of each relation within each target attribute. Relation and orientation have balanced margins but not a full relation × orientation crossing within every pair × attribute cell. The table is explicit; the relation assignment is balanced rather than derived from a single row-index parity.

Alternate relations in the `other_attribute` condition are `team` and `project`, balanced 12/12 in the 24-history stages and 48/48 in confirmation. The eight color terms are plausible badge/color values and plausible code/label identifiers; they also work as team/project names. These are named **different-relation controls**, not claimed to be semantically unrelated; `team` and `project` differ from the target attribute and from `label`/`code`.

### Coherent prompts and contrasts

Every relevant history uses a neutral current-state assignment: `Currently, {entity}’s {attribute} is {value}.` Historical queried-attribute assignments use `Previously, ...`. The control constructions remain distinct:

| Condition | Context before current-state assignments | Role |
|---|---|---|
| `superseded` | Previous queried-attribute values for both entities | Obsolete assigned values |
| `early_unassigned` | Unassigned queried-attribute values, then current states | Early unassigned control |
| `late_unassigned` | Current states, then unassigned queried-attribute values | Late unassigned control |
| `entity_mention` | Values mentioned in unrelated notes, then current states | Entity association without attribute assignment |
| `other_attribute` | Previous team/project values, then current queried-attribute states | Different-relation control |
| `live` | Sole current-state assignment, edited in place | Positive control; correct answer changes on matched edit |

The v2 prompts are a new realization, not a wording replication of v1. A tokenizer-free audit checks every rendered baseline/edit pair: only the intended source/donor value span changes, aside from its unavoidable token-length consequences. Six values are distinct in every history. The audit preserves the live positive control, active orientation mapping, and complete factorial/member validation.

The three planned primary history-level contrasts are `R_superseded − R_early_unassigned`, `R_superseded − R_entity_mention`, and `R_superseded − R_other_attribute` (the last pools the two frozen alternate relations). Also report `R_superseded − R_late_unassigned`, `R_live − R_superseded`, live minus each control, all six condition means, orientation strata and each order cell. For five two-block conditions, compute E within each of the four historical/current order cells per history, edit and query; compute symmetric x/z R from those cell means. Report aligned order by averaging (historical,current) cells (0,0)/(1,1) before R, and reversed order by averaging (0,1)/(1,0) before R. Live has one assignment block, so its current-order labels are duplicate nuisance bookkeeping, deduplicated for competence and averaged for relevance; do not claim an independent live current-order contrast. Bootstrap histories (2,000 draws, seed 73021). Confirmation is all-trial; complete-history correctness-conditioned analysis is secondary only.

The bounded surface-class continuation mass remains the primary score. Strict semantic rank-one accuracy must be at least 99% on unique prompts separately in each of the six conditions; ties fail. Superseded also requires a positive aggregate current-over-stale mass margin. A failed recomputed gate blocks confirmation. Do not relax these gates.

## Lineage and audit status

The fresh exclusion ledger contains donor-independent physical-history signatures and source hashes from available older data and both superseded v2 preparation previews. It ignores generated IDs, renderer and replacement/donor identities. Fixed seeded histories structurally reject the frozen signatures and earlier stages; no seed retries or outcome-based exclusions occur. The v1 audit compared all available Qwen, Gemma, Phi and Mistral development/gate/confirmation datasets with this same physical-history definition: it found **no cross-stage collisions**. It found only expected exact shared histories across models within the same stage (24 development, 24 gate, 96 confirmation); those are shared-model comparison histories, not cross-stage overlap. A repeated signature alone is not evidence of leakage. The unrecovered original 4.52-logit Qwen raw histories cannot be audited.

For the r5 freeze, Qwen and Phi tokenizer-only checks passed; the local Gemma validation attempt stopped because its pinned files were absent from that machine's cache. Under the exclusion freeze, all three candidate maps bind to the same freeze. All saved development/gate scores have been migrated and recomputed without model loading; their original score provenance remains recorded, and evaluations are unchanged. Qwen and Gemma pass their gates and preflights; Phi fails the frozen `entity_mention` accuracy threshold.

The exclusion freeze binds the revised contract, code hash, model configs, v1 lineage and prompt audit. The configs preserve the actual v1 model/tokenizer pins and scoring settings. All three candidate maps, migrated stage reports and the Qwen/Gemma preflights bind to this freeze. The sealed v1 donor-independent stage audit is saved at `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/v1_donor_independent_stage_audit_sealed.json`.

## Operational provenance and confirmation status (2026-10-04)

The runtimefix implementation freeze is `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004/protocol_freeze.json`, SHA-256 `f2981fa0f05a5e29e9478f8a94987e7578b2ce3667d1c941e8ce31cf8cdfef9a`. Its embedded status (“no model inference authorized or executed by this artifact”) records the freeze-time state; it does not describe later scoring. The prior stopped Gemma generation/scoring attempt remains preserved in the earlier revisions. The runtimefix preparation, audits, migrations, gates, and preflights then completed; the overnight log records both confirmation preflights passing before the runner began Qwen scoring. This is the recorded release of confirmation under the runtimefix freeze.

Both runtimefix confirmation bundles contain 96 histories / 18,432 members and have identical dataset SHA-256 `27c1e9b80e62a0cb0266d393fd6120dff2bcf637d8e9468491748e740287c08d`. Their 96 donor-independent physical-history signatures match, and all rows match on the history, semantic condition, attribute/relation, orientation, historical/current order, edited entity/field, query entity, and answer assignments. Thus Qwen and Gemma were generated from the same physical histories and assignments. Each model's candidate map binds to the runtimefix freeze; its confirmation provenance binds to that freeze and to its own migrated gate report and preflight hashes. Both migrated preflights record a passing frozen gate. The recorded inference settings are Qwen revision `b968826d9c46dd6066d109eabc6255188de91218` (int8 weights, float16 activations) and Gemma revision `093f9f388b31de276ce2de164bdc2081324b9767` (int8 weights, bfloat16 activations), both on CUDA with Python 3.12.13. The preparation and overnight logs, dataset/provenance/geometry sidecars, and per-model migration/gate/preflight artifacts preserve this lineage.

The confirmation provenance sidecars cite shared-history registry SHA-256 `392f6a158b4f024cb634f7240fd9dd2febe9755ecbe2cc4b4de7b6caf3a25d25`; the registry is present at `outputs/cross_model_relational_v2/shared_histories/factorial_relation_counterbalanced_geometryfix_exclusions_20261003_confirmatory.json` and lists histories shared across the two model bundles. The stopped-Gemma exclusion record is `resolved_confirmation_exclusion` in `configs/cross_model_relational_v2_geometryfix_exclusions/prior_history_exclusions.json`. Recomputing donor-independent physical signatures from the stopped source dataset and both runtimefix datasets gives 96/96 signatures in the exclusion record and **zero overlap** with either runtimefix confirmation dataset. The runtimefix datasets also have zero overlap with the stopped source dataset under the same signature function. Matching the shared-history registry across models is expected; it is not exclusion leakage.

Both confirmation score files and analysis reports are present. Each score file has 18,432 valid rows, 18,432 unique example IDs in dataset order, and zero row errors. Score sidecars authenticate the score hashes; analysis reports bind to the matching dataset and score hashes; dataset provenance, candidate maps, and preflights bind to freeze SHA-256 `f2981fa0f05a5e29e9478f8a94987e7578b2ce3667d1c941e8ce31cf8cdfef9a`. Both analyses use 96 histories, 2,000 history-bootstrap draws, seed 73021, and all-trial estimates. Runtime provenance records Qwen revision `b968826d9c46dd6066d109eabc6255188de91218` (int8 weights, float16 activations) and Gemma revision `093f9f388b31de276ce2de164bdc2081324b9767` (int8 weights, bfloat16 activations), both on CUDA with Python 3.12.13.

Applying the dated analysis amendment's lower-95%-bootstrap-bound-above-zero rule, **neither model supports the full three-control claim**:

| Contrast (`R_superseded − R_control`) | Qwen3-8B, nats (95% history-bootstrap CI) | Gemma 3 4B, nats (95% history-bootstrap CI) | Interpretation |
|---|---:|---:|---|
| Early unassigned | 2.011 `[1.855, 2.170]` | 1.575 `[1.440, 1.719]` | Supported in both, overall |
| Entity mention | -3.504 `[-3.806, -3.198]` | -2.589 `[-2.829, -2.361]` | Opposite direction in both; entity-mention relevance exceeds superseded relevance |
| Other attribute | 0.314 `[0.147, 0.482]` | 0.004 `[-0.104, 0.118]` | Supported for Qwen only; inconclusive for Gemma |

Order-stratified results narrow that interpretation. For `R_superseded − R_early_unassigned`, aligned-order estimates are negative for Qwen (-1.108, CI `[-1.281, -0.925]`) and Gemma (-0.895, CI `[-1.084, -0.719]`), while reversed-order estimates are positive for Qwen (5.130, CI `[4.920, 5.348]`) and Gemma (4.044, CI `[3.810, 4.292]`). Reversed-order entity-mention contrasts remain negative in both models (Qwen -4.659, CI `[-5.119, -4.207]`; Gemma -1.700, CI `[-1.946, -1.465]`). Reversed-order other-attribute contrasts do not clear zero (Qwen 0.162, CI `[-0.049, 0.367]`; Gemma 0.069, CI `[-0.080, 0.213]`). The full three-control claim therefore fails overall and under reversed order. Keep model results separate; do not pool raw nats.

Chronology: (1) the earlier Gemma confirmation attempt stopped on geometry validation before model loading and produced no scores; (2) the corrected runtimefix freeze and audits were completed; (3) both model preflights passed and the overnight runner released confirmation; (4) Qwen and Gemma scoring and analysis are complete in the available artifacts. The captured overnight log contains the preflights and Qwen startup but lacks terminal completion lines, so preserve it as an incomplete log; the complete score counts, sidecars, hashes, and analysis reports establish the saved outputs' completion. No freeze or experiment artifact was modified to record later events.

## Workflow commands (historical; confirmation complete)

The commands below document the completed setup and migration path for reproducibility. Do not rerun them against the existing sealed output paths. The runtimefix Qwen and Gemma confirmation runs are complete; preserve their saved scores and reports.

The runtimefix tokenizer-only design audit, candidate maps, migrations, gate reports and preflights bind to the runtimefix freeze. Qwen and Gemma pass the recomputed gates and preflights; Phi fails the frozen `entity_mention` competence gate and must remain stopped. The runtimefix Qwen and Gemma confirmation datasets are present and byte-identical, and scoring/analysis are complete. See the dated operational status above for validated results.

Use a separate uv environment for each model. The migration environment is recorded in the sidecar's `provenance`; the scoring environment is in `original_inference_provenance`. Runtime checks resolve the latter when present and otherwise use `provenance`. The required Python version is read from that resolved inference record, and both Python and every recorded package version must match before scoring or resuming. Always include `--extra model`; omitting it caused uv to recreate the default `.venv` without the model dependencies. `UV_PROJECT_ENVIRONMENT` gives each model its own environment. If a runtime check fails, stop and investigate; do not relax the equality check. A partial score file may be resumed only when its checkpoint runtime matches the selected environment. Preserve a mismatched partial file and continue with a fresh output path after resolving its provenance.

Do not use `scripts/run_relational_robustness.sh` for this continuation: it invokes `.venv/bin/python` and its confirmation action regenerates the dataset. Use the dedicated uv runner below. It does not run Phi.

The runtime fix changes hashed implementation files, so the earlier exclusion freeze cannot authorize execution of the updated code. On Ubuntu, after transferring the code changes, use `bash scripts/prepare_and_run_relational_runtimefix.sh`. It creates a new implementation freeze under `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004/`, validates Qwen/Gemma tokenizers locally, and migrates saved development/gate scores from the original `factorial_relation_counterbalanced_20261003` directory. The freeze's prompt audit uses `--design-only`: it reviews the unchanged seeded design without treating reused development histories as a fresh experiment. The source v1 artifacts for all three configured models are required for freeze lineage validation; this does not run Phi inference. All new files have fresh paths, and the preparation script refuses an existing destination. It saves its full log as `preparation.log` in the new revision directory.

After preparation succeeds, that script automatically invokes `scripts/run_relational_confirmations_overnight.sh` with the new revision. Both scripts derive Python from original inference provenance and use `.venv-qwen-runtimefix` and `.venv-gemma-runtimefix`, optionally overridden with `RELATIONAL_QWEN_ENV` and `RELATIONAL_GEMMA_ENV`. Both models' runtime, saved-score checkpoint runtime, CUDA, gate bindings, confirmation lineage and geometry are checked before any scoring starts. Pinned checkpoints are loaded without a forward pass during preflight. If either preflight fails, neither confirmation run starts. The runner reuses validated confirmation bundles, resumes matching checkpoints, validates completed outputs, and runs Gemma even if Qwen scoring fails. Overnight logs are under the selected revision's `overnight_logs/`. Keep the process in `tmux` or `screen`.

The command below records how scoring was invoked. It is historical; scoring and analysis are complete, so do not run it again:

```bash
RELATIONAL_REVISION=factorial_relation_counterbalanced_runtimefix_20261004 \
  bash scripts/run_relational_confirmations_overnight.sh
```

The scripts use portable Bash syntax suitable for Ubuntu. Shell control-flow regression tests substitute a fake `uv` command and load no model weights; they do not establish that remote CUDA, local checkpoint files, or model forward passes work. The completed run's preflights and saved artifact validation are summarized above.

Confirmation scoring is complete for Qwen and Gemma in the artifacts above. Do not rerun the overnight confirmation runner. Preserve the complete scores and reports; proceed to paper interpretation using the prespecified contrasts and order rules.

| Model | Required uv runtime | Status in the inspected runtimefix snapshot |
|---|---|---|
| Qwen 3 8B | `.venv-qwen-runtimefix`; Python is derived from its original inference provenance | Confirmation scoring and analysis complete; 18,432/18,432 rows validate. |
| Gemma 3 4B | `.venv-gemma-runtimefix`; Python is derived from its original inference provenance | Confirmation scoring and analysis complete; 18,432/18,432 rows validate. |
| Phi 4 mini | — | Do not run confirmation; frozen gate fails `entity_mention` accuracy. |

Preserve all score rows and errors. The expected record counts and freeze/provenance bindings have been verified; the frozen contrasts and order results are reported above. No v1 inference rerun is part of this workflow.
