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

For the final runtimefix confirmation, tokenizer-aware geometry sidecars additionally show that all 9,216 baseline/edit pairs per model replace a one-token source span with a one-token donor span. The unchanged suffix therefore retains the same token positions in these evaluated prompts. The Qwen and Gemma geometry sidecars are `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004/{qwen3_8b,gemma3_4b}/confirmatory.jsonl.geometry.json`.

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

## Extended saved-score reanalysis (2026-10-04)

The narrow original confirmation reports were preserved. `scripts/analyze_relational_confirmation_v2.py` independently rechecks their frozen history rows against saved datasets and scores, verifies seals and hash/binding metadata, runtimefix freeze, gate/preflight bindings, row/ID/pair completeness and the stopped-Gemma source exclusion. It writes the extension to `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004/extended_analysis_v6/`. Reproduce with:

```bash
PYTHONPATH=. python3 scripts/analyze_relational_confirmation_v2.py
```

The script refuses to overwrite an existing output directory. Each model folder contains complete per-history estimates, all six condition R values by aligned/reversed order, competence cells, paired correctness, and paired source/donor score changes. `report.json` records history-bootstrap estimates and intervals; `audit.json` records freeze and exclusion checks. The overall frozen contrasts reproduce the table above: Qwen early 2.011, entity mention −3.504, other attribute 0.314; Gemma early 1.575, entity mention −2.589, other attribute 0.004 nats. There is no material discrepancy from the supplied table.

The contrast export includes every historical/current order cell (`h0_c0`, `h0_c1`, `h1_c0`, `h1_c1`), plus aligned and reversed aggregates. Direct aligned-minus-reversed differences use paired history values and are secondary/exploratory. The all-trial analysis remains primary. Saved surface event scores support source/donor score-change summaries, but are bounded prefix continuation scores without termination, not unrestricted answer probabilities or practical harm measures. Null-crossing intervals are inconclusive; no equivalence margin was prespecified. The stopped source dataset hash and all 96 ledger signatures are checked, with zero overlap in either confirmation bundle.

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

### Complete results for the paper writer

All estimates below were recomputed from the sealed v2 confirmation scores. Values are means with 95% history-bootstrap intervals (2,000 draws; seed 73021), in nats. The bootstrap resamples histories. These tables are summaries; all 96 per-history rows for both models are also retained in `extended_analysis_v6/{qwen3_8b,gemma3_4b}/per_history.csv`.

#### Condition-level R

Each cell gives overall R, aligned-order R, and reversed-order R, respectively. Live has only one physical assignment block; its duplicated order labels are bookkeeping and its identical values do not estimate an order effect.

| Model | Condition | Overall R | Aligned R | Reversed R |
| --- | --- | --- | --- | --- |
| Qwen 3 8B | live | 32.598 [31.258, 33.967] | 32.598 [31.258, 33.967] | 32.598 [31.258, 33.967] |
| Qwen 3 8B | superseded | 1.974 [1.840, 2.109] | 2.144 [1.995, 2.301] | 1.804 [1.660, 1.947] |
| Qwen 3 8B | early_unassigned | -0.037 [-0.203, 0.121] | 3.252 [3.046, 3.455] | -3.327 [-3.518, -3.150] |
| Qwen 3 8B | late_unassigned | 0.001 [-0.208, 0.211] | 3.187 [2.948, 3.442] | -3.184 [-3.433, -2.945] |
| Qwen 3 8B | entity_mention | 5.478 [5.128, 5.825] | 4.493 [4.235, 4.770] | 6.463 [5.971, 6.940] |
| Qwen 3 8B | other_attribute | 1.660 [1.488, 1.835] | 1.679 [1.527, 1.835] | 1.641 [1.422, 1.863] |
| Gemma 3 4B | live | 39.355 [38.303, 40.415] | 39.355 [38.303, 40.415] | 39.355 [38.303, 40.415] |
| Gemma 3 4B | superseded | 1.688 [1.568, 1.817] | 1.703 [1.574, 1.841] | 1.672 [1.529, 1.831] |
| Gemma 3 4B | early_unassigned | 0.113 [-0.023, 0.245] | 2.598 [2.395, 2.798] | -2.372 [-2.567, -2.178] |
| Gemma 3 4B | late_unassigned | 0.125 [-0.055, 0.305] | 3.445 [3.202, 3.696] | -3.194 [-3.447, -2.945] |
| Gemma 3 4B | entity_mention | 4.277 [4.029, 4.545] | 5.181 [4.884, 5.496] | 3.372 [3.131, 3.633] |
| Gemma 3 4B | other_attribute | 1.683 [1.541, 1.832] | 1.763 [1.598, 1.932] | 1.603 [1.453, 1.771] |


#### Frozen superseded-minus-control contrasts and order dependence

For each control, aligned-minus-reversed is the direct paired contrast computed within history; it is secondary/exploratory. Order-cell rows compare the matching cell-specific R values. Negative/positive interval rules apply to the frozen primary contrasts overall; order-stratified results are descriptive.

| Control | Scope/order | Qwen 3 8B | Gemma 3 4B |
| --- | --- | --- | --- |
| early_unassigned | overall | 2.011 [1.855, 2.170] | 1.575 [1.440, 1.719] |
| early_unassigned | aligned | -1.108 [-1.281, -0.925] | -0.895 [-1.084, -0.719] |
| early_unassigned | reversed | 5.130 [4.920, 5.348] | 4.044 [3.810, 4.292] |
| early_unassigned | aligned_minus_reversed_secondary_exploratory | -6.238 [-6.481, -5.983] | -4.939 [-5.275, -4.620] |
| early_unassigned | historical_0_current_0 | -1.019 [-1.265, -0.747] | -0.840 [-1.087, -0.589] |
| early_unassigned | historical_0_current_1 | 5.106 [4.842, 5.370] | 3.998 [3.705, 4.293] |
| early_unassigned | historical_1_current_0 | 5.155 [4.883, 5.432] | 4.090 [3.814, 4.382] |
| early_unassigned | historical_1_current_1 | -1.197 [-1.445, -0.936] | -0.949 [-1.174, -0.719] |
| entity_mention | overall | -3.504 [-3.806, -3.198] | -2.589 [-2.829, -2.361] |
| entity_mention | aligned | -2.350 [-2.572, -2.125] | -3.478 [-3.769, -3.197] |
| entity_mention | reversed | -4.659 [-5.119, -4.207] | -1.700 [-1.946, -1.465] |
| entity_mention | aligned_minus_reversed_secondary_exploratory | 2.309 [1.933, 2.707] | -1.778 [-2.049, -1.519] |
| entity_mention | historical_0_current_0 | -2.458 [-2.760, -2.166] | -3.451 [-3.811, -3.100] |
| entity_mention | historical_0_current_1 | -4.715 [-5.236, -4.203] | -1.572 [-1.874, -1.272] |
| entity_mention | historical_1_current_0 | -4.602 [-5.116, -4.076] | -1.828 [-2.158, -1.495] |
| entity_mention | historical_1_current_1 | -2.241 [-2.520, -1.951] | -3.505 [-3.893, -3.157] |
| other_attribute | overall | 0.314 [0.147, 0.482] | 0.004 [-0.104, 0.118] |
| other_attribute | aligned | 0.465 [0.295, 0.649] | -0.060 [-0.215, 0.099] |
| other_attribute | reversed | 0.162 [-0.049, 0.367] | 0.069 [-0.080, 0.213] |
| other_attribute | aligned_minus_reversed_secondary_exploratory | 0.302 [0.103, 0.509] | -0.128 [-0.323, 0.081] |
| other_attribute | historical_0_current_0 | 0.436 [0.227, 0.644] | -0.113 [-0.326, 0.093] |
| other_attribute | historical_0_current_1 | 0.097 [-0.174, 0.367] | -0.069 [-0.260, 0.123] |
| other_attribute | historical_1_current_0 | 0.228 [-0.037, 0.501] | 0.207 [-0.007, 0.425] |
| other_attribute | historical_1_current_1 | 0.494 [0.274, 0.714] | -0.007 [-0.212, 0.208] |
| late_unassigned | overall | 1.973 [1.762, 2.183] | 1.562 [1.390, 1.742] |
| late_unassigned | aligned | -1.043 [-1.281, -0.810] | -1.741 [-1.968, -1.505] |
| late_unassigned | reversed | 4.988 [4.724, 5.266] | 4.866 [4.592, 5.155] |
| late_unassigned | aligned_minus_reversed_secondary_exploratory | -6.031 [-6.307, -5.754] | -6.607 [-6.951, -6.243] |
| late_unassigned | historical_0_current_0 | -0.879 [-1.212, -0.565] | -1.452 [-1.711, -1.174] |
| late_unassigned | historical_0_current_1 | 5.001 [4.670, 5.360] | 4.599 [4.237, 4.960] |
| late_unassigned | historical_1_current_0 | 4.975 [4.586, 5.370] | 5.133 [4.811, 5.466] |
| late_unassigned | historical_1_current_1 | -1.207 [-1.585, -0.845] | -2.030 [-2.357, -1.683] |


The order-stratified contrast table above shows the early-unassigned superseded-minus-control contrast changes sign with order in both models. The direct paired interaction intervals exclude zero for Qwen and Gemma on early unassigned; entity mention also has a nonzero paired order difference in both; other-attribute interaction is supported for Qwen but its Gemma interval includes zero. Do not interpret these secondary interactions as prespecified confirmatory tests. The full three-control criterion is conjunctive: all three primary superseded-minus-control lower 95% bounds must exceed zero within a model. It fails for each model because entity-mention contrasts are negative; other attribute also fails for Gemma. This is evidence against the relation-selective prediction, not proof of no superseded-relation effect. No equivalence margin was prespecified.

#### Candidate competence and correctness-conditioned secondary analysis

Candidate accuracy is the proportion of rows with strict rank one (ties count incorrect), summarized over all 3,072 members per condition. The cell range is the minimum to maximum accuracy across the 32 query × edited-slot × historical-order × current-order × baseline/edit cells for each condition. Current-minus-stale margin is only defined by the saved rows for superseded; other controls have no stale queried-attribute answer under the frozen definition.

| Model | Condition | All-member accuracy | Range across 32 cells | Current−stale margin (superseded only) |
| --- | --- | --- | --- | --- |
| Qwen 3 8B | live | 99.7396% | 97.9167%–100.0000% | — |
| Qwen 3 8B | superseded | 100.0000% | 100.0000%–100.0000% | 21.734 |
| Qwen 3 8B | early_unassigned | 100.0000% | 100.0000%–100.0000% | — |
| Qwen 3 8B | late_unassigned | 99.9023% | 98.9583%–100.0000% | — |
| Qwen 3 8B | entity_mention | 100.0000% | 100.0000%–100.0000% | — |
| Qwen 3 8B | other_attribute | 100.0000% | 100.0000%–100.0000% | — |
| Gemma 3 4B | live | 100.0000% | 100.0000%–100.0000% | — |
| Gemma 3 4B | superseded | 100.0000% | 100.0000%–100.0000% | 23.854 |
| Gemma 3 4B | early_unassigned | 99.9674% | 98.9583%–100.0000% | — |
| Gemma 3 4B | late_unassigned | 100.0000% | 100.0000%–100.0000% | — |
| Gemma 3 4B | entity_mention | 100.0000% | 100.0000%–100.0000% | — |
| Gemma 3 4B | other_attribute | 100.0000% | 100.0000%–100.0000% | — |


Complete-history correctness conditioning is secondary, defined as histories with every confirmation row strictly rank one. It selects 91/96 Qwen histories and 95/96 Gemma histories. The three conditioned primary contrasts are:

| Model | Control | Conditioned superseded−control |
| --- | --- | --- |
| Qwen 3 8B | early_unassigned | 2.015 [1.861, 2.176] |
| Qwen 3 8B | entity_mention | -3.445 [-3.750, -3.141] |
| Qwen 3 8B | other_attribute | 0.371 [0.201, 0.532] |
| Gemma 3 4B | early_unassigned | 1.576 [1.433, 1.714] |
| Gemma 3 4B | entity_mention | -2.591 [-2.827, -2.354] |
| Gemma 3 4B | other_attribute | 0.004 [-0.106, 0.119] |


These conditioned results are descriptive and do not replace all-history estimates. Paired correctness counts and all cell-level accuracy/margin summaries are in `competence_cells.csv` and `paired_correctness.csv`; the condition-level completeness counts are in each `report.json`.

#### Paired source/donor score changes

These are descriptive means over saved baseline/edit score pairs, in bounded semantic log-mass units; they are not unrestricted probabilities. `source change` is the source candidate’s edited-minus-baseline mass; `donor change` is the replacement/donor candidate’s edited-minus-baseline mass. Their difference is identity transfer.

| Model | Condition | Source change | Donor change | Identity transfer |
| --- | --- | --- | --- | --- |
| Qwen 3 8B | live | -15.623 | 15.954 | 31.576 |
| Qwen 3 8B | superseded | -1.610 | 1.497 | 3.108 |
| Qwen 3 8B | early_unassigned | -4.208 | 4.012 | 8.221 |
| Qwen 3 8B | late_unassigned | -5.901 | 5.682 | 11.583 |
| Qwen 3 8B | entity_mention | -3.832 | 3.584 | 7.417 |
| Qwen 3 8B | other_attribute | -1.348 | 1.331 | 2.679 |
| Gemma 3 4B | live | -14.265 | 14.142 | 28.407 |
| Gemma 3 4B | superseded | -1.241 | 1.230 | 2.471 |
| Gemma 3 4B | early_unassigned | -3.315 | 2.975 | 6.290 |
| Gemma 3 4B | late_unassigned | -4.511 | 3.999 | 8.510 |
| Gemma 3 4B | entity_mention | -3.209 | 2.957 | 6.166 |
| Gemma 3 4B | other_attribute | -1.444 | 1.113 | 2.557 |


The paired table separates decreased source mass from increased donor mass. Both contribute to the algebraic identity-transfer contrast. These row-pair means have no interval attached; the frozen history-bootstrap inference remains the R and contrast tables above.

The saved event masses also provide absolute bounded semantic log masses for the current answer, source, and donor candidates before and after each edit. The following are descriptive pair-level means (not probabilities; no history-bootstrap intervals are attached):

| Model | Condition | Current before | Current after | Source before | Source after | Donor before | Donor after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen 3 8B | live | -0.004 | -0.012 | -8.027 | -23.650 | -24.086 | -8.132 |
| Qwen 3 8B | superseded | 0.000 | 0.000 | -22.271 | -23.881 | -23.618 | -22.120 |
| Qwen 3 8B | early_unassigned | 0.000 | 0.000 | -16.488 | -20.697 | -20.497 | -16.485 |
| Qwen 3 8B | late_unassigned | -0.004 | -0.004 | -17.218 | -23.119 | -22.899 | -17.216 |
| Qwen 3 8B | entity_mention | -0.001 | -0.001 | -19.836 | -23.668 | -23.452 | -19.868 |
| Qwen 3 8B | other_attribute | 0.000 | 0.000 | -21.048 | -22.396 | -22.234 | -20.903 |
| Gemma 3 4B | live | 0.000 | 0.000 | -9.784 | -24.049 | -24.191 | -10.049 |
| Gemma 3 4B | superseded | 0.000 | 0.000 | -24.311 | -25.552 | -25.498 | -24.268 |
| Gemma 3 4B | early_unassigned | 0.000 | -0.004 | -20.919 | -24.235 | -23.971 | -20.996 |
| Gemma 3 4B | late_unassigned | 0.000 | 0.000 | -20.887 | -25.398 | -25.138 | -21.139 |
| Gemma 3 4B | entity_mention | 0.000 | 0.000 | -20.540 | -23.749 | -23.606 | -20.650 |
| Gemma 3 4B | other_attribute | 0.000 | -0.001 | -23.486 | -24.929 | -24.734 | -23.621 |

#### Reading the order-cell decomposition

The individual order-cell contrasts and aligned/reversed summaries are all listed above for the four frozen controls. The contrast rows are direct differences of the relevant R values within each history and then summarized over histories. For live, the duplicate order split is not a physical factor. The early-unassigned sign reversal is substantial in both models: aligned contrasts are negative, reversed contrasts positive, and the paired aligned-minus-reversed differences are negative. Do not infer interaction by comparing whether two separate confidence intervals include zero.


#### Orientation and remaining frozen contrasts

Orientation is the frozen x/z semantic mapping factor. The table gives each primary contrast separately within orientation. These are stratified summaries; no model or orientation pooling is performed.

| Model | Orientation | Control | Superseded−control |
| --- | --- | --- | --- |
| Qwen 3 8B | 0 | early_unassigned | 2.155 [1.947, 2.376] |
| Qwen 3 8B | 0 | entity_mention | -3.464 [-3.876, -3.055] |
| Qwen 3 8B | 0 | other_attribute | 0.399 [0.134, 0.628] |
| Qwen 3 8B | 1 | early_unassigned | 1.868 [1.660, 2.096] |
| Qwen 3 8B | 1 | entity_mention | -3.544 [-4.023, -3.117] |
| Qwen 3 8B | 1 | other_attribute | 0.228 [0.011, 0.450] |
| Gemma 3 4B | 0 | early_unassigned | 1.529 [1.333, 1.726] |
| Gemma 3 4B | 0 | entity_mention | -2.538 [-2.915, -2.181] |
| Gemma 3 4B | 0 | other_attribute | 0.080 [-0.078, 0.249] |
| Gemma 3 4B | 1 | early_unassigned | 1.620 [1.435, 1.808] |
| Gemma 3 4B | 1 | entity_mention | -2.640 [-2.960, -2.355] |
| Gemma 3 4B | 1 | other_attribute | -0.071 [-0.226, 0.090] |


The frozen secondary contrasts below complete the live comparisons. Estimates use the same history bootstrap; these are descriptive secondary contrasts, not substitutions for the three-control decision.

| Model | Secondary contrast | Estimate [95% history-bootstrap interval] |
| --- | --- | --- |
| Qwen 3 8B | R live minus R superseded | 30.624 [29.270, 31.985] |
| Qwen 3 8B | R live minus R early unassigned | 32.636 [31.291, 34.030] |
| Qwen 3 8B | R live minus R entity mention | 27.120 [25.572, 28.646] |
| Qwen 3 8B | R live minus R other attribute | 30.938 [29.498, 32.400] |
| Qwen 3 8B | R live minus R late unassigned | 32.597 [31.228, 34.001] |
| Qwen 3 8B | R superseded minus R late unassigned | 1.973 [1.762, 2.183] |
| Gemma 3 4B | R live minus R superseded | 37.667 [36.595, 38.713] |
| Gemma 3 4B | R live minus R early unassigned | 39.242 [38.201, 40.315] |
| Gemma 3 4B | R live minus R entity mention | 35.078 [33.867, 36.256] |
| Gemma 3 4B | R live minus R other attribute | 37.671 [36.601, 38.741] |
| Gemma 3 4B | R live minus R late unassigned | 39.229 [38.181, 40.267] |
| Gemma 3 4B | R superseded minus R late unassigned | 1.562 [1.390, 1.742] |


The frozen prompt design uses one v2 lexical frame/template throughout, so a between-frame contrast is unsupported by the saved design. The complete cell-level competence export crosses query, edited slot, historical order, current order, and baseline/edit member; the condition table above summarizes each condition and reports the full cell-accuracy range. The supplemental tables below report per-attribute R and the query/edited-entity role decomposition; both are descriptive secondary analyses, not additional confirmatory claims.

#### Supplemental heterogeneity summaries

These stratified summaries use the same 2,000-draw, seed-73021 history bootstrap. Attribute and entity-role splits are descriptive secondary decompositions; the overall frozen contrasts above remain primary.

##### Target attribute

Each attribute has 24 confirmation histories. The table gives superseded R and the three primary superseded-minus-control contrasts.

| Model | Attribute | Estimate | Histories | Mean [95% history-bootstrap interval] |
| --- | --- | --- | --- | --- |
| Qwen 3 8B | badge | R superseded | 24 | 2.494 [2.209, 2.797] |
| Qwen 3 8B | badge | R superseded minus R early unassigned | 24 | 2.518 [2.234, 2.820] |
| Qwen 3 8B | badge | R superseded minus R entity mention | 24 | -4.310 [-4.819, -3.787] |
| Qwen 3 8B | badge | R superseded minus R other attribute | 24 | 0.748 [0.512, 0.969] |
| Qwen 3 8B | color | R superseded | 24 | 2.036 [1.778, 2.292] |
| Qwen 3 8B | color | R superseded minus R early unassigned | 24 | 2.160 [1.899, 2.442] |
| Qwen 3 8B | color | R superseded minus R entity mention | 24 | -1.794 [-2.129, -1.469] |
| Qwen 3 8B | color | R superseded minus R other attribute | 24 | 0.876 [0.683, 1.068] |
| Qwen 3 8B | code | R superseded | 24 | 1.768 [1.561, 1.976] |
| Qwen 3 8B | code | R superseded minus R early unassigned | 24 | 1.666 [1.424, 1.913] |
| Qwen 3 8B | code | R superseded minus R entity mention | 24 | -4.570 [-5.098, -4.058] |
| Qwen 3 8B | code | R superseded minus R other attribute | 24 | -0.570 [-0.866, -0.280] |
| Qwen 3 8B | label | R superseded | 24 | 1.597 [1.404, 1.797] |
| Qwen 3 8B | label | R superseded minus R early unassigned | 24 | 1.701 [1.413, 1.984] |
| Qwen 3 8B | label | R superseded minus R entity mention | 24 | -3.342 [-3.668, -3.017] |
| Qwen 3 8B | label | R superseded minus R other attribute | 24 | 0.200 [-0.042, 0.459] |
| Gemma 3 4B | badge | R superseded | 24 | 1.872 [1.679, 2.065] |
| Gemma 3 4B | badge | R superseded minus R early unassigned | 24 | 1.591 [1.379, 1.815] |
| Gemma 3 4B | badge | R superseded minus R entity mention | 24 | -1.794 [-2.047, -1.556] |
| Gemma 3 4B | badge | R superseded minus R other attribute | 24 | 0.011 [-0.206, 0.232] |
| Gemma 3 4B | color | R superseded | 24 | 1.648 [1.419, 1.867] |
| Gemma 3 4B | color | R superseded minus R early unassigned | 24 | 1.723 [1.506, 1.954] |
| Gemma 3 4B | color | R superseded minus R entity mention | 24 | -2.532 [-2.852, -2.203] |
| Gemma 3 4B | color | R superseded minus R other attribute | 24 | -0.028 [-0.236, 0.208] |
| Gemma 3 4B | code | R superseded | 24 | 1.754 [1.522, 2.003] |
| Gemma 3 4B | code | R superseded minus R early unassigned | 24 | 1.543 [1.188, 1.882] |
| Gemma 3 4B | code | R superseded minus R entity mention | 24 | -3.937 [-4.418, -3.452] |
| Gemma 3 4B | code | R superseded minus R other attribute | 24 | -0.045 [-0.292, 0.217] |
| Gemma 3 4B | label | R superseded | 24 | 1.477 [1.228, 1.766] |
| Gemma 3 4B | label | R superseded minus R early unassigned | 24 | 1.441 [1.176, 1.713] |
| Gemma 3 4B | label | R superseded minus R entity mention | 24 | -2.094 [-2.358, -1.809] |
| Gemma 3 4B | label | R superseded minus R other attribute | 24 | 0.078 [-0.110, 0.264] |


##### Query/edited-entity role

For each history and condition, `queried entity` averages E when the edited variable equals the query variable (`x→x`, `z→z`); `other entity` averages E for the cross-entity cells (`x→z`, `z→x`). Their paired same-minus-other difference equals the symmetric relevance R algebra. This is an explanatory role split of the frozen estimand, not an independent confirmatory claim.

| Model | Condition | Queried entity E | Other entity E | Paired same−other |
| --- | --- | --- | --- | --- |
| Qwen 3 8B | live | 47.875 [46.757, 48.995] | 15.277 [14.651, 15.847] | 32.598 [31.258, 33.967] |
| Qwen 3 8B | superseded | 4.095 [3.916, 4.274] | 2.121 [2.000, 2.242] | 1.974 [1.840, 2.109] |
| Qwen 3 8B | early_unassigned | 8.202 [7.920, 8.499] | 8.239 [7.955, 8.530] | -0.037 [-0.203, 0.121] |
| Qwen 3 8B | late_unassigned | 11.584 [11.178, 12.012] | 11.582 [11.209, 11.958] | 0.001 [-0.208, 0.211] |
| Qwen 3 8B | entity_mention | 10.156 [9.675, 10.626] | 4.678 [4.458, 4.889] | 5.478 [5.128, 5.825] |
| Qwen 3 8B | other_attribute | 3.509 [3.296, 3.726] | 1.849 [1.746, 1.955] | 1.660 [1.488, 1.835] |
| Gemma 3 4B | live | 48.084 [47.099, 49.076] | 8.730 [8.431, 9.023] | 39.355 [38.303, 40.415] |
| Gemma 3 4B | superseded | 3.315 [3.143, 3.479] | 1.628 [1.512, 1.738] | 1.688 [1.568, 1.817] |
| Gemma 3 4B | early_unassigned | 6.347 [6.139, 6.548] | 6.234 [6.015, 6.445] | 0.113 [-0.023, 0.245] |
| Gemma 3 4B | late_unassigned | 8.573 [8.269, 8.893] | 8.448 [8.128, 8.786] | 0.125 [-0.055, 0.305] |
| Gemma 3 4B | entity_mention | 8.304 [7.903, 8.710] | 4.028 [3.801, 4.249] | 4.277 [4.029, 4.545] |
| Gemma 3 4B | other_attribute | 3.398 [3.214, 3.591] | 1.715 [1.591, 1.839] | 1.683 [1.541, 1.832] |


The protocol uses the same `nora_relational_v2` rendering throughout; there is one lexical frame, so frame contrasts are not identified. Attribute subgroup intervals are less precise than the overall estimate and should not be used to select a favorable subgroup. No Qwen/Gemma pooling was performed.
