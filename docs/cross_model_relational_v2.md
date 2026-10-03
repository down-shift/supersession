# Relational/order v2 protocol

## Geometry validation implementation migration (2026-10-03)

The first Gemma confirmation dataset contained 96 histories / 18,432 members, but scoring stopped in `dataset_info()` before model loading. The saved geometry deduplicated identical rendered prompts while incorrectly reusing abstract `x`/`z` span labels across opposite orientations. That dataset and all earlier artifacts remain preserved under `factorial_relation_counterbalanced_20261003`.

The corrected validator stores a per-example semantic-field-to-physical-span map. The unchanged scientific design was first sealed as implementation revision `factorial_relation_counterbalanced_geometryfix_20261003`; freeze `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_20261003_r5/protocol_freeze.json` binds geometry schema 2. After importing and hashing the stopped Gemma confirmation bundle, a separate exclusion revision was frozen at `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/protocol_freeze.json`. Its ledger adds the bundle's 96 donor-independent physical histories and records the source dataset, geometry, provenance and stopped-record hashes. The original and r5 freezes and all source artifacts remain unchanged. Stage claims include the implementation hash, preventing collisions with claims from earlier code revisions.

The migration command reaudits all rows with the model tokenizer, checks candidate events, tokenizer/chat-template hashes, row semantics, and rendered prompts against the old saved score rows, and writes new dataset, geometry, score, and migration provenance files. It records source hashes, original inference revision, corrected validation revision and reason. Saved scores remain characterized as scores from their original inference run. No model is loaded by this migration command.

Under the exclusion freeze, Qwen and Phi tokenizer-only validation completed locally with 528 surface continuations, 384 edit pairs, 768 candidate substitutions and 768 span mappings per model. Their migrated saved development and gate score evaluations exactly match their original reports. Qwen's recomputed gate passes and preflight passes; Phi's gate fails `entity_mention` semantic accuracy and remains ineligible. Gemma's prior r5 tokenizer validation does not bind to this exclusion freeze. Revalidate Gemma against the exclusion freeze on the machine with its pinned tokenizer. No scoring or confirmation generation has run under the exclusion freeze.

Example migration commands for Qwen are recorded here; use the matching `gemma3_4b` config and paths after its tokenizer validation succeeds. Migrate development first, analyze it, then migrate gate against that report, analyze the gate and create preflight. Use fresh, unused paths for each command:

```bash
REV=outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003
OLD=outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/qwen3_8b
NEW=$REV/qwen3_8b
CONFIG=configs/cross_model_relational_v2_geometryfix_exclusions/qwen3_8b.yaml

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

The Qwen and Phi migrations under the exclusion freeze were validated from saved development/gate scores. Their recomputed development and gate evaluations exactly match the original reports. Qwen's migrated gate and preflight pass; Phi's migrated gate fails `entity_mention` competence. Gemma's saved-stage scores still require migration under the exclusion freeze after Gemma validation is run there. Confirmation data has not been generated under this freeze.

**Archived status for the original freeze:** `factorial_relation_counterbalanced_20261003` was the initial corrected design revision. Development and gate scoring did occur for Qwen, Gemma and Phi. The Gemma confirmation dataset was generated under that earlier freeze; score validation stopped before model loading, and no confirmation score file exists. The failed bundle is preserved under the r5 `migration_source/gemma3_4b/` directory and its histories are excluded by the newer freeze. No remote host was accessed in this work. This is a repository-frozen protocol, not externally preregistered.

The earlier `relational_order_correction_20261003` freeze and the intermediate `factorial_coherent_controls_20261003` preparation freeze, candidate maps and previews remain unchanged. Qwen and Phi development datasets in the earlier directory cite the first freeze and are not valid development evidence for this design. The intermediate candidate maps cite a relation-confounded fractional allocation and are not valid for this final freeze. All partial outputs remain preserved. A read-only process check found no active Qwen/Phi v2 jobs on 2026-10-03.

## Frozen design

The implementation and allocation are in `src/cross_model/robustness_v2.py`; the contract and model lineage checks are in `src/cross_model/robustness_protocol.py`. Current artifacts are under `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/`; its configs and physical-history exclusion ledger are under `configs/cross_model_relational_v2_geometryfix_exclusions/`. Earlier freeze directories remain archived separately.

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

For the r5 freeze, Qwen and Phi tokenizer-only checks passed; the local Gemma validation attempt stopped because its pinned files were absent from that machine's cache. The later exclusion freeze has fresh Qwen/Phi candidate artifacts. Their saved development and gate scores were migrated and recomputed under the new freeze without model loading; the original score provenance remains recorded. Their development and gate evaluations are unchanged. Qwen passes its gate and preflight; Phi fails the frozen `entity_mention` accuracy threshold. The failed Gemma confirmation bundle is now present, and its 96 physical signatures are recorded in the new exclusion ledger. Gemma still needs a tokenizer-only validation artifact bound to the exclusion freeze.

The exclusion freeze binds the revised contract, code hash, model configs, v1 lineage and prompt audit. The configs preserve the actual v1 model/tokenizer pins and scoring settings. Qwen and Phi candidate maps and migrated reports bind to this freeze; Gemma validation and saved-stage migration remain outstanding. The sealed v1 donor-independent stage audit is saved at `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/v1_donor_independent_stage_audit_sealed.json`.

## Safe command sequence

The tokenizer-free design audit and freeze are at `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/`. Qwen and Phi candidate maps are sealed to it; migrated saved-score development/gate analyses and Qwen preflight are complete. The inference hold remains until Gemma has a candidate map and validation artifact under this exact freeze and its saved development/gate scores have been migrated and checked. Do not access the remote host from this workspace or rerun create-only commands at existing paths.

On the host where the pinned Gemma tokenizer is available, run this tokenizer-only validation with the frozen checkout, then copy the resulting fresh artifacts back for review. It does not score the model:

```bash
RELATIONAL_REVISION=factorial_relation_counterbalanced_geometryfix_exclusions_20261003 \
RELATIONAL_CONFIG_DIR=configs/cross_model_relational_v2_geometryfix_exclusions \
RELATIONAL_RUN_DIR=outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/gemma3_4b \
RELATIONAL_FREEZE=outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/protocol_freeze.json \
bash scripts/run_relational_robustness.sh gemma3_4b validate
```

After import, verify all three candidate maps bind to `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/protocol_freeze.json`, then migrate Gemma's saved development/gate scores using the documented `migrate`, `analyze`, and `preflight` sequence above with its saved-score paths. Preserve its original score hashes and inference revision. Review Gemma's recomputed gate and preflight. Keep Phi stopped because its gate fails. Do not generate confirmation datasets or score confirmation until the audits and eligible-model checks are complete.

After the remaining Gemma audit and migration are verified, the eligible Qwen/Gemma confirmation workflow can be resumed using the existing scripts with explicit revision overrides. The commands below are the future inference sequence; do not execute them while the audit hold remains:

```bash
REV=factorial_relation_counterbalanced_geometryfix_exclusions_20261003
CFG=configs/cross_model_relational_v2_geometryfix_exclusions
OUT=outputs/cross_model_relational_v2/$REV
export RELATIONAL_REVISION="$REV" RELATIONAL_CONFIG_DIR="$CFG" RELATIONAL_FREEZE="$OUT/protocol_freeze.json"
export RELATIONAL_RUN_DIR="$OUT/qwen3_8b"
bash scripts/run_relational_robustness.sh qwen3_8b confirmatory
export RELATIONAL_RUN_DIR="$OUT/gemma3_4b"
bash scripts/run_relational_robustness.sh gemma3_4b confirmatory
```

Both models' development/gate saved scores must first be migrated and recomputed under this freeze; no development or gate inference is needed if correspondence is verified and the gate remains passing. Each confirmation action generates the fresh confirmation bundle, scores it, and writes analysis. Preserve all error rows. Never point `RELATIONAL_RUN_DIR` at an earlier revision. No v1 rerun is part of this workflow.
