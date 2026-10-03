# Relational/order v2 protocol

## Geometry validation implementation migration (2026-10-03)

The first Gemma confirmation dataset contained 96 histories / 18,432 members, but scoring stopped in `dataset_info()` before model loading. The saved geometry deduplicated identical rendered prompts while incorrectly reusing abstract `x`/`z` span labels across opposite orientations. That dataset and all earlier artifacts remain preserved under `factorial_relation_counterbalanced_20261003`.

The corrected validator stores a per-example semantic-field-to-physical-span map. The unchanged scientific design is sealed as implementation revision `factorial_relation_counterbalanced_geometryfix_20261003`, with a separate config directory and output namespace. Freeze `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_20261003_r5/protocol_freeze.json` binds the current implementation and geometry schema 2. Stage claims now include the implementation hash, preventing collisions with claims from earlier code revisions.

The migration command reaudits all rows with the model tokenizer, checks candidate events, tokenizer/chat-template hashes, row semantics, and rendered prompts against the old saved score rows, and writes new dataset, geometry, score, and migration provenance files. It records source hashes, original inference revision, corrected validation revision and reason. Saved scores remain characterized as scores from their original inference run. No model is loaded by this migration command.

Qwen and Phi tokenizer validations pass under the new freeze. Their candidate events, tokenizer hashes, chat-template hashes and example prompt strings match the prior candidate artifacts. Gemma tokenizer validation is not available in this local environment. The failed remote confirmation dataset has not been imported; the new exclusion ledger therefore keeps confirmation generation blocked until its original bundle is supplied and its donor-independent physical-history signatures are added. Updating that ledger requires a new freeze and fresh validation artifacts.

Example migration commands for Qwen are recorded here; use the matching `gemma3_4b` config and paths after its tokenizer validation succeeds. Migrate development first, analyze it, then migrate gate against that report, analyze the gate and create preflight. Use fresh, unused paths for each command:

```bash
REV=outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_20261003_r5
OLD=outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/qwen3_8b
NEW=$REV/qwen3_8b
CONFIG=configs/cross_model_relational_v2_geometryfix/qwen3_8b.yaml

.venv/bin/python -m scripts.robustness_v2 migrate --stage development --config "$CONFIG" \
  --candidates "$NEW/candidates.json" --original-candidates "$OLD/candidates.json" \
  --dataset "$OLD/development.jsonl" --scores "$OLD/development_scores.jsonl" \
  --output "$NEW/migrated/development.jsonl" --local-files-only
.venv/bin/python -m scripts.robustness_v2 analyze --stage development --config "$CONFIG" \
  --candidates "$NEW/candidates.json" --dataset "$NEW/migrated/development.jsonl" \
  --scores "$NEW/migrated/development.jsonl.scores.jsonl" \
  --output "$NEW/migrated/development_report.json"
.venv/bin/python -m scripts.robustness_v2 migrate --stage frozen_gate --config "$CONFIG" \
  --candidates "$NEW/candidates.json" --original-candidates "$OLD/candidates.json" \
  --dataset "$OLD/gate.jsonl" --scores "$OLD/gate_scores.jsonl" \
  --prior-dataset "$NEW/migrated/development.jsonl" \
  --development-report "$NEW/migrated/development_report.json" \
  --output "$NEW/migrated/gate.jsonl" --local-files-only
.venv/bin/python -m scripts.robustness_v2 analyze --stage frozen_gate --config "$CONFIG" \
  --candidates "$NEW/candidates.json" --dataset "$NEW/migrated/gate.jsonl" \
  --scores "$NEW/migrated/gate.jsonl.scores.jsonl" \
  --development-report "$NEW/migrated/development_report.json" \
  --output "$NEW/migrated/gate_report.json"
.venv/bin/python -m scripts.robustness_v2 preflight --config "$CONFIG" \
  --candidates "$NEW/candidates.json" --gate "$NEW/migrated/gate_report.json" \
  --output "$NEW/migrated/preflight.json"
```

The Qwen and Phi migrations were validated from saved development/gate scores. Their recomputed development and gate evaluations exactly match the original reports. Qwen's migrated gate passes; Phi's migrated gate still fails `entity_mention` competence and remains ineligible. Qwen's migrated preflight passes. Gemma saved-stage migration awaits tokenizer-only validation on a machine with the pinned tokenizer. Confirmation for every model remains held until the failed Gemma attempt history is imported, excluded in a new freeze, and that freeze's audit prerequisites are satisfied.

**Archived status for the original freeze:** `factorial_relation_counterbalanced_20261003` was the initial corrected design revision. Development and gate scoring did occur for Qwen, Gemma and Phi. Confirmation scoring did not occur; the reported Gemma confirmation attempt stopped before model loading. Its failed dataset was generated under that earlier freeze. The original Gemma final-freeze candidate/validation artifacts remain unavailable in this workspace; no remote host was accessed. The active implementation freeze and current hold are described above. This is a repository-frozen protocol, not externally preregistered.

The earlier `relational_order_correction_20261003` freeze and the intermediate `factorial_coherent_controls_20261003` preparation freeze, candidate maps and previews remain unchanged. Qwen and Phi development datasets in the earlier directory cite the first freeze and are not valid development evidence for this design. The intermediate candidate maps cite a relation-confounded fractional allocation and are not valid for this final freeze. All partial outputs remain preserved. A read-only process check found no active Qwen/Phi v2 jobs on 2026-10-03.

## Frozen design

The implementation and allocation are in `src/cross_model/robustness_v2.py`; the contract and model lineage checks are in `src/cross_model/robustness_protocol.py`. Fresh artifacts are under `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/`; configs and the physical-history exclusion ledger are under `configs/cross_model_relational_v2_factorial_final/`.

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

For this design revision, prompt audit and development/gate previews were regenerated in fresh paths. Qwen and Phi's pinned tokenizer-only checks passed their surface continuation, paired-edit, candidate-substitution and value/entity span audits. Their event counts and token-length profiles differ across values; these differences are disclosed and follow the frozen surface-class policy. The Gemma validation attempt in this workspace stopped under local-only loading because the pinned files were absent from its local cache. That historical stop is preserved and does not establish whether the reported remote files are usable. No new development dataset scoring, gate scoring or confirmation dataset generation occurred. Qwen/Phi histories and plans from the previous freeze are preserved but excluded from this revision. The previous Gemma tokenizer stop is also preserved; it does not satisfy this revision's prerequisite.

The new freeze binds the revised contract, code hash, model configs, v1 lineage and prompt audit. The configs preserve the actual v1 model/tokenizer pins and scoring settings. Development and gate histories are registered under the new contract/revision; no model-specific stage claims exist until official datasets are generated, and confirmation remains unregistered until a passing gate. The sealed v1 donor-independent stage audit is saved at `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/v1_donor_independent_stage_audit_sealed.json`.

## Safe command sequence

The final tokenizer-free audit, previews and `protocol_freeze.json` have already been created in the fresh revision directory. Qwen and Phi candidate maps are sealed to this freeze and their paired-edit and candidate-substitution audits passed. Gemma's final-freeze candidate map and validation artifact are not yet present in this workspace; this is the remaining required audit. The user reports the pinned files are available remotely, but that environment has not been accessed. Do not rerun create-only commands at existing paths.

Run Gemma's no-logit tokenizer audit only in an environment where the pinned files and this frozen checkout are available, and save the resulting artifact to a fresh final-freeze run path:

```bash
bash scripts/run_relational_robustness.sh gemma3_4b validate
```

Before scoring, verify that all three candidate maps and validation artifacts bind to `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/protocol_freeze.json`; review their lineage and token geometry. The hold remains until this passes for Gemma.

Then, on the configured inference host, run each eligible model in order. Inspect each fresh gate report and preflight before that model's confirmation action. A failed gate blocks confirmation and must remain sealed as failed:

```bash
bash scripts/run_relational_robustness.sh qwen3_8b development
bash scripts/run_relational_robustness.sh qwen3_8b gate
# Continue only if outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/qwen3_8b/gate_report.json records pass=true.
bash scripts/run_relational_robustness.sh qwen3_8b confirmatory

bash scripts/run_relational_robustness.sh gemma3_4b development
bash scripts/run_relational_robustness.sh gemma3_4b gate
# Continue only if outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/gemma3_4b/gate_report.json records pass=true.
bash scripts/run_relational_robustness.sh gemma3_4b confirmatory

bash scripts/run_relational_robustness.sh phi4_mini development
bash scripts/run_relational_robustness.sh phi4_mini gate
# Continue only if outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/phi4_mini/gate_report.json records pass=true.
bash scripts/run_relational_robustness.sh phi4_mini confirmatory
```

Each shell action invokes the protocol CLI through the same frozen workflow and performs its stage analysis; do not substitute commands from another protocol. The workflow resumes only scored JSONL checkpoints with their original fingerprints. Never point `RELATIONAL_RUN_DIR` at an earlier revision. Do not run development, gate or confirmation while the hold remains. No v1 rerun is part of this workflow.
