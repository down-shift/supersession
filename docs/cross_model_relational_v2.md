# Relational/order v2 protocol

## Geometry validation implementation migration (2026-10-03)

The first Gemma confirmation dataset contained 96 histories / 18,432 members, but scoring stopped in `dataset_info()` before model loading. The saved geometry deduplicated identical rendered prompts while incorrectly reusing abstract `x`/`z` span labels across opposite orientations. That dataset and all earlier artifacts remain preserved under `factorial_relation_counterbalanced_20261003`.

The corrected validator stores a per-example semantic-field-to-physical-span map. The unchanged scientific design was first sealed as implementation revision `factorial_relation_counterbalanced_geometryfix_20261003`; freeze `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_20261003_r5/protocol_freeze.json` binds geometry schema 2. After importing and hashing the stopped Gemma confirmation bundle, a separate exclusion revision was frozen at `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/protocol_freeze.json`. Its ledger adds the bundle's 96 donor-independent physical histories and records the source dataset, geometry, provenance and stopped-record hashes. The original and r5 freezes and all source artifacts remain unchanged. Stage claims include the implementation hash, preventing collisions with claims from earlier code revisions.

The migration command reaudits all rows with the model tokenizer, checks candidate events, tokenizer/chat-template hashes, row semantics, and rendered prompts against the old saved score rows, and writes new dataset, geometry, score, and migration provenance files. It records source hashes, original inference revision, corrected validation revision and reason. Saved scores remain characterized as scores from their original inference run. No model is loaded by this migration command.

Under the exclusion freeze, Qwen and Phi tokenizer-only validation completed locally; Gemma's validation artifact was produced on the machine with its pinned tokenizer and copied here. All three artifacts bind to the exclusion freeze. Gemma passes the surface, paired-edit, exhaustive substitution and geometry checks: 5,376 substitutions, 384 edit pairs, 768 example mappings over 528 deduplicated prompts. Its events, tokenizer hash and chat-template hash match the prior Gemma candidate map. Qwen, Phi and Gemma saved development/gate score evaluations were migrated and exactly match their earlier reports. Qwen and Gemma's recomputed gates and preflights pass; Phi's gate fails `entity_mention` accuracy and has no successful preflight. Qwen's 96-history confirmation dataset was generated and validated, but the first score startup stopped before model loading because the gate provenance records Python 3.13.5 while that invocation used Python 3.12.13. A later uv check with Python 3.13.5 and the model extra reported matching runtime and available CUDA; confirmation scoring and analysis are still pending. Gemma confirmation data has not been generated.

Example migration commands for Qwen are recorded here; use the matching `gemma3_4b` config and paths for Gemma. Migrate development first, analyze it, then migrate gate against that report, analyze the gate and create preflight. Use fresh, unused paths for each command. The Qwen migration provenance records Python 3.13.5, so these commands use its own uv environment and include the `model` extra:

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

The Qwen, Phi and Gemma migrations under the exclusion freeze were validated from saved development/gate scores. Their recomputed development and gate evaluations exactly match the earlier reports. Qwen and Gemma's migrated gates and preflights pass; Phi's migrated gate fails `entity_mention` competence. Gemma's migration records the original inference revision and source hashes. Qwen has a valid confirmation dataset but no scores because the runtime provenance check stopped before model loading; the transferred bundle does not include its shared confirmation-history registry. Gemma has no confirmation dataset under this freeze.

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

For the r5 freeze, Qwen and Phi tokenizer-only checks passed; the local Gemma validation attempt stopped because its pinned files were absent from that machine's cache. Under the exclusion freeze, all three candidate maps bind to the same freeze. All saved development/gate scores have been migrated and recomputed without model loading; their original score provenance remains recorded, and evaluations are unchanged. Qwen and Gemma pass their gates and preflights; Phi fails the frozen `entity_mention` accuracy threshold.

The exclusion freeze binds the revised contract, code hash, model configs, v1 lineage and prompt audit. The configs preserve the actual v1 model/tokenizer pins and scoring settings. All three candidate maps, migrated stage reports and the Qwen/Gemma preflights bind to this freeze. The sealed v1 donor-independent stage audit is saved at `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_20261003/v1_donor_independent_stage_audit_sealed.json`.

## Safe command sequence

The tokenizer-only design audit, candidate maps, migrations, gate reports and preflights bind to the exclusion freeze. Qwen and Gemma pass the recomputed gates and preflights; Phi fails the frozen `entity_mention` competence gate and must remain stopped. Qwen's confirmation dataset already exists; its first score attempt stopped before model loading. Gemma confirmation has not been generated. No development or gate inference needs to be repeated.

Use a separate uv environment for each model. The migrated gate score provenance records Qwen at Python 3.13.5 and Gemma at Python 3.12.13. The lockfile pins package versions but allows both Python versions. Always include `--extra model`; omitting it caused uv to recreate the default `.venv` without the model dependencies. `UV_PROJECT_ENVIRONMENT` gives each Python version its own environment, avoiding replacement when switching models. Check the runtime and CUDA against the model's saved gate provenance before scoring. If the check fails, stop and investigate; do not relax the equality check.

The shell wrapper `scripts/run_relational_robustness.sh` invokes `.venv/bin/python`, and its `confirmatory` action regenerates the dataset. Use the direct uv commands below: Qwen's dataset already exists, while Gemma needs generation. These commands do not run for Phi.

For one sequential run, use `bash scripts/run_relational_confirmations_overnight.sh`. It uses the verified `.venv` for Qwen (Python 3.13.5) and a separate `.venv-gemma312` for Gemma (Python 3.12.13). Before scoring either model, it checks both runtimes, CUDA, gate bindings, confirmation lineage and geometry, and loads both pinned checkpoints without a forward pass; if either preflight fails, neither confirmation run starts. It generates Gemma's confirmation dataset only if absent, scores and analyzes both models, and continues to the second model if the first scoring run fails. It writes a timestamped log under `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/overnight_logs/`, resumes scoring only when a valid progress checkpoint exists, validates existing completed score/analysis artifacts instead of overwriting them, and leaves Phi stopped. Keep the process attached to a persistent terminal session such as `tmux` or `screen` for an overnight run.

### Qwen 3 8B

```bash
cd ~/supersession
set -euo pipefail
REV=factorial_relation_counterbalanced_geometryfix_exclusions_20261003
BASE="outputs/cross_model_relational_v2/$REV"
CFG=configs/cross_model_relational_v2_geometryfix_exclusions
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --frozen --extra model --python 3.13.5

uv run --frozen --extra model --python 3.13.5 python - <<'PY'
import json, torch
from src.utils import provenance
path = "outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/qwen3_8b/migrated/gate.jsonl.scores.jsonl.provenance.json"
expected = json.load(open(path))["provenance"]
current = provenance({}, None)
assert (current["python"], current["packages"]) == (expected["python"], expected["packages"])
assert torch.cuda.is_available()
print("Qwen runtime and CUDA match")
PY

uv run --frozen --extra model --python 3.13.5 python -m scripts.robustness_v2 score \
  --config "$CFG/qwen3_8b.yaml" --candidates "$BASE/qwen3_8b/candidates.json" \
  --dataset "$BASE/qwen3_8b/confirmatory.jsonl" \
  --output "$BASE/qwen3_8b/confirmatory_scores.jsonl"
uv run --frozen --extra model --python 3.13.5 python -m scripts.robustness_v2 analyze \
  --stage confirmatory --config "$CFG/qwen3_8b.yaml" \
  --candidates "$BASE/qwen3_8b/candidates.json" \
  --dataset "$BASE/qwen3_8b/confirmatory.jsonl" \
  --scores "$BASE/qwen3_8b/confirmatory_scores.jsonl" \
  --output "$BASE/qwen3_8b/confirmatory_analysis.json"
```

### Gemma 3 4B

```bash
cd ~/supersession
set -euo pipefail
REV=factorial_relation_counterbalanced_geometryfix_exclusions_20261003
BASE="outputs/cross_model_relational_v2/$REV"
CFG=configs/cross_model_relational_v2_geometryfix_exclusions
export UV_PROJECT_ENVIRONMENT=.venv-gemma312
uv sync --frozen --extra model --python 3.12.13

uv run --frozen --extra model --python 3.12.13 python - <<'PY'
import json, torch
from src.utils import provenance
path = "outputs/cross_model_relational_v2/factorial_relation_counterbalanced_geometryfix_exclusions_20261003/gemma3_4b/migrated/gate.jsonl.scores.jsonl.provenance.json"
expected = json.load(open(path))["provenance"]
current = provenance({}, None)
assert (current["python"], current["packages"]) == (expected["python"], expected["packages"])
assert torch.cuda.is_available()
print("Gemma runtime and CUDA match")
PY

uv run --frozen --extra model --python 3.12.13 python -m scripts.robustness_v2 generate \
  --stage confirmatory --config "$CFG/gemma3_4b.yaml" \
  --candidates "$BASE/gemma3_4b/candidates.json" \
  --gate "$BASE/gemma3_4b/migrated/gate_report.json" \
  --preflight "$BASE/gemma3_4b/migrated/preflight.json" \
  --output "$BASE/gemma3_4b/confirmatory.jsonl" --local-files-only
uv run --frozen --extra model --python 3.12.13 python -m scripts.robustness_v2 score \
  --config "$CFG/gemma3_4b.yaml" --candidates "$BASE/gemma3_4b/candidates.json" \
  --dataset "$BASE/gemma3_4b/confirmatory.jsonl" \
  --output "$BASE/gemma3_4b/confirmatory_scores.jsonl"
uv run --frozen --extra model --python 3.12.13 python -m scripts.robustness_v2 analyze \
  --stage confirmatory --config "$CFG/gemma3_4b.yaml" \
  --candidates "$BASE/gemma3_4b/candidates.json" \
  --dataset "$BASE/gemma3_4b/confirmatory.jsonl" \
  --scores "$BASE/gemma3_4b/confirmatory_scores.jsonl" \
  --output "$BASE/gemma3_4b/confirmatory_analysis.json"
```

### Phi 4 mini

Do not generate or score confirmation for Phi. Its recomputed frozen gate fails `entity_mention` accuracy. Keep its existing development and gate artifacts as the record of that result.

For Qwen, preserve the existing confirmation dataset and stopped-score log. For both eligible models, preserve all score rows and errors, verify expected record counts and freeze/provenance bindings, then report all frozen contrasts and order cells. No v1 inference rerun is part of this workflow.
