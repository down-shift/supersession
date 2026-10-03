# cross_model_relational_v2: corrected relational/order design

**Not ready for inference. No v2 model scoring has been executed.** The initial preview had substantive value-allocation, live-control and validation bugs. Its previews and the first saved-score reanalysis reports must not enter the paper. The corrected design revision is `relational_order_correction_20261003`; preparation artifacts and corrected reanalyses use fresh paths. A code/config freeze, actual tokenizer audits, fresh competence stages and passing recomputed gates are required before confirmation. This is a logged repository protocol, not an externally registered preregistration.

Completed `cross_model_v1` Qwen/Gemma/Phi confirmation results retain their original scientific status. Mistral's failed gate, Gemma/Phi downstream failures, completed mechanisms, the old Qwen 4.52-logit experiment and failed historical branches are preserved. Correcting descriptive reanalysis does not rerun or revise those experiments. The v1 relevance estimator itself is reused here.

## Fixed design

The shared vocabulary is `amber coral jade pearl slate teal violet ivory`. Model IDs, immutable tokenizer/model revisions, dtype, int8 weights, eager attention, chat template, `enable_thinking=False`, generation prefix and `Answer:` suffix follow each actual v1 confirmation run. `verify_v1_lineage()` compares settings to the saved score provenance and verifies its score hash. In particular, Gemma's actual recorded revision is `093f9f388b31de276ce2de164bdc2081324b9767`, with bfloat16 compute. Pinning alone is not used as evidence of continuity.

Seeds are central in `src/cross_model/robustness_v2.py`: tokenizer validation **20261030**, development **20261031**, gate **20261101**, confirmation **20261102**. Counts are 4 tokenizer-only audit histories, then **24/24/96** fresh behavioral histories. All three models share identical stage histories.

Each history draws **six distinct values** in a fixed order: `initial_x`, `initial_z`, `proposed_x`, `proposed_z`, replacement for initial x, replacement for initial z. Neither replacement is any of the four assigned identities. Replacement values never supply current assignments in fixed-answer conditions. Both replacements are matched across conditions, order cells and queries.

`configs/cross_model_relational_v2/prior_history_exclusions.json` freezes physical entity/attribute/historical/current signatures from available earlier datasets and the deterministic faulty previews. It records source hashes, retains failed branches, and ignores donor identities, IDs, templates and analytic orientation labels when testing overlap. Within the fixed seeded streams, generation rejects these signatures and signatures reserved for earlier stages in validation/development/gate/confirmation priority. This is structural rejection before logits, with no seed retries or outcome-based selection. Additional supplied or automatically discovered prior datasets cause a clear stop on overlap; they cannot silently change the frozen draws. Stage-specific IDs are never a disjointness test. The original 4.52-logit Qwen raw histories remain unavailable here, as documented in the v1 audit; missing historical artifacts are not certified as covered.

Each behavioral history has this complete product:

```text
6 conditions × 2 historical/mention orders × 2 current orders
             × 2 edited entities × 2 queried entities × 2 pair members
= 192 members / 96 edit pairs
```

Development and gate each contain 4,608 members; confirmation contains 18,432. The validator checks exact counts, unique IDs, every cell including the edited-entity axis, the six-value separation, derived answers, orientation, pair identities and all nonedited semantic metadata. It reconstructs every expected member from its immutable history definition.

| Condition | Initial block | Current block | Paired answer behavior |
|---|---|---|---|
| `superseded` | queried attribute assigned historical values | queried attribute updated to separate proposed values | unchanged |
| `early_unassigned` | historical identities listed as unassigned values | queried attribute assigned proposed values | unchanged |
| `late_unassigned` | unassigned value block placed **after** current block | queried attribute assigned proposed values | unchanged |
| `entity_mention` | each entity mentions its historical identity in an unrelated note, with no attribute assignment | queried attribute assigned proposed values | unchanged |
| `other_attribute` | each entity's **tag** assigned its historical identity | queried attribute assigned proposed values | unchanged |
| `live` | sole queried-attribute assignment remains live | no later restoration or reaffirmation block | changes to replacement only for the associated edited-entity query |

The first corrected development superseded intervention has historical x=amber, current x=violet, replacement x=teal. Baseline/edit are `amber → violet` and `teal → violet`; neither intervention identity is current. The live counterparts are a sole amber assignment versus a sole teal assignment, with the associated correct answer changing from amber to teal.

For the five two-block conditions, historical entity/mention order and current entity order independently render all four combinations. `variables` is the actual analytic x/z→entity mapping used by the renderer, and both semantic orientations are retained. **Live has a single assignment block:** its historical order is physically rendered; its current-order levels duplicate the same prompt solely to preserve matched cell bookkeeping. They are deduplicated for competence and averaged for relevance. An aligned/reversed live comparison is structurally uninformative and must not be presented as evidence of independent live-order robustness. Adding a restoration block would change the positive control and is prohibited.

The renderer reuses v1's exact assigned/updated/unassigned sentences, query, answer-only instruction, and chat suffix where applicable. Entity-note and other-attribute sentences are new renderings under this separate version. The study is not described as an exact wording replication of v1. No wording is selected on causal results; development is competence/implementation-only.

Behavioral `pair_direction=0/1` means **baseline/edited members** of initial-x→independent-replacement-x and initial-z→independent-replacement-z interventions. It does not mean two activation-transfer directions. Reciprocal activation patching is a separate later intervention and is not implemented here; obsolete identity swaps are not part of this design.

## Scoring, gate and analyses

Shared scoring uses the existing bounded surface-class continuation mass, preserving the fixed lowercase/title-case × zero/one ASCII space proposals, tokenizer-only exclusions, disjoint token-event deduplication, full sequence teacher forcing, and full-vocabulary probability normalization. There is **no EOS, termination, length normalization, candidate renormalization or new causal estimand**.

Tokenizer-only validation uses the shared `tokens.validate()` with explicit renderer/position callbacks. It audits every paired edit and all candidate substitutions, and saves model/tokenizer/template geometry. Dataset geometry joins every example ID to a prompt hash and records exact character bounds, zero-based token offsets, token lengths and IDs for historical/control values, current values, all represented entity occurrences and the queried entity. All tokens outside each edited value span must be identical; unequal edit lengths remain disclosed. No mechanism is executed.

The gate retains **strict semantic candidate accuracy ≥99% on unique prompts separately in each of the six focal conditions**. Ties are incorrect. Superseded prompts additionally require positive aggregate current-over-stale mass margin. Mean rank and the **384** orientation/order/query/edit/member cells are diagnostic and completeness checks, not additional small-cell thresholds. Development has `pass=null`; poor development competence cannot veto the one fresh gate or authorize prompt/vocabulary redevelopment. A failed gate is sealed as failed, and confirmation generation stops before tokenizer/model work. Gate, development and preflight reports are recomputed from underlying scores and hashes; an asserted `pass=true` cannot bypass the gate.

The v1 algebra remains:

```text
E = [S(replacement)-S(source)]edited - [S(replacement)-S(source)]baseline
R = 0.5 * [(E_edit_x_query_x - E_edit_x_query_z)
         + (E_edit_z_query_z - E_edit_z_query_x)]
```

For each history and condition, first average E over the four independent historical/current order cells for each edit/query; then compute symmetric x/z R and average the two entity-specific differences. For aligned order, average cells `(0,0)` and `(1,1)`; for reversed order, average `(0,1)` and `(1,0)` before the same R calculation. Control mention-order averaging follows this same rule. Higher contrasts are paired within history; bootstrap histories, never prompts/variants/layers. Draws=2,000, seed=73021, 95% intervals. The vectorized bootstrap is tested against the existing seeded v1 algorithm.

The three predefined primary contrasts are:

- `R_superseded - R_early_unassigned`
- `R_superseded - R_entity_mention`
- `R_superseded - R_other_attribute`

Also report superseded minus late-unassigned, live minus superseded, live minus each control, each order cell and aligned/reversed comparisons, each entity variable, both orientations, per-history distributions and each model separately. Confirmation includes **all trials**, regardless of correctness. Causal reporting is unavailable for development/gate stages.

The shared workflow now accepts an optional protocol module; original v1 calls retain their original defaults. Relational configs, contract, candidate maps, stage claims and shared-history registry are isolated. Artifact seals bind config/model pins, seeds, dataset/score hashes, git/code hashes, source v1 provenance, tokenizer audits, token geometry, freeze, runtime details, Python/packages, dtype, quantization and resolved device placement. Confirmation requires a passing recomputed gate/preflight and the same Python/package runtime as gate scoring. Scoring resumes use the existing JSONL checkpoint mechanism with exact fingerprints; interrupted/failed artifacts are retained. Every new output uses exclusive creation.

## Corrected reanalysis of existing confirmation scores

The corrected report is `reporting_correction_20261003`, written under `outputs/cross_model_v1_reanalysis_corrected_20261003/`. Earlier `cross_model_v1_reanalysis_relational*` reports are superseded and must not enter the paper. Source datasets, scores, sidecars and original confirmation analyses are untouched. Reanalysis validates the original sealed dataset/score provenance and their hashes; no weights are loaded.

- Recompute strict rank and correctness from finite saved masses, compare stored diagnostics, and reject mismatches, duplicate IDs, duplicate members, missing cells and missing/inconsistent saved surface likelihoods.
- Compute each member's current-answer margin against **its own strongest incorrect candidate**. Retain member-target versus fixed-baseline-competitor and fixed-baseline-target versus fixed-competitor contrasts under separate labels.
- Report accuracy/rank, current/stale margins, absolute current/source/replacement prefix masses, paired current-answer score and margin changes **separately by condition**. Live member-target score changes can compare different correct-answer identities; fixed-baseline-answer score changes remain separately available.
- Compute query-specific counterbalanced-unassigned R separately for `xz`/`zx`, preserving all four edit/query cells in each order. An average of raw E is never labeled R or used to establish sign reversal. Fixed-order ordinary irrelevant R is reported separately.
- Retain full per-history relevance, relevant/other-query edit terms, source and replacement score changes, per-surface likelihoods, additive replacement/negative-source relevance decompositions, and orientation/variable strata. The frozen normalized metric remains ratio of model means, eligible only if the irrelevant-corrected live bootstrap lower bound and **every** paired draw denominator exceed 1 nat.
- Keep the all-trial estimate primary. The secondary table selects only **complete histories with every superseded baseline/edited/query/edit member correct**, then retains all matched live/ordinary/counterbalanced members, including both irrelevant orders. It recomputes R and the primary contrast; it does not average correctness-filtered raw E.

JSON preserves detailed terms and probability mass, while `.json.md` provides a compact paper-review table. New relational conditions cannot be recovered from v1 because they were not collected; the report labels them unavailable. Missing fields cause a clear failure, never new inference.

## Local preparation results (2026-10-03)

- Full non-GPU suite: **282 passed in 79.06s**; shell syntax and `git diff --check` passed.
- Fresh development/gate previews contain 24 histories and 4,608 members each. The development audit contains 192 rendered baseline/edit pairs spanning every condition/order/query/edit cell in both orientations. Prompts were reviewed before creating `outputs/cross_model_relational_v2/corrected_20261003/protocol_freeze.json`.
- Cached Qwen/Phi tokenizer-only validation passed all continuation, paired-edit, substitution and entity/value-offset audits. Actual tokenizer/template hashes and candidate events match their original v1 run artifacts. Both official development datasets passed token audits and lineage checks; their 4,608 semantic members are identical. Forward-count plans were generated without logits.
- Gemma's local-files-only tokenizer audit stopped because its pinned tokenizer/config is absent from this host's cache; its stopped artifact is retained. Gemma continuity against actual v1 saved provenance was verified, but its new tokenizer geometry remains unaudited. **The inference hold remains in force.** No development/gate/confirmation scoring or confirmation dataset generation occurred.
- All three corrected saved-score reports were recomputed in fresh paths. Every shared numeric history-level v1 estimate matches the original confirmation analysis within `1e-10` across all 96 histories per model. The all-trial primary estimates are unchanged; the corrected descriptive tables supersede the faulty reporting.

### Files added or modified

| Area | Files |
|---|---|
| Protocol and data | `src/cross_model/robustness_v2.py`, new `src/cross_model/robustness_protocol.py`, three YAMLs and new `prior_history_exclusions.json` under `configs/cross_model_relational_v2/` |
| Analysis and integrity | new `src/cross_model/robustness_analysis.py`, `src/cross_model/reanalysis.py`, `src/cross_model/score_checks.py` |
| Shared infrastructure | `src/cross_model/tokens.py`, `src/cross_model/scoring.py`, `src/cross_model/workflow.py`; optional callbacks/protocol argument preserve original defaults |
| Commands | `scripts/robustness_v2.py`, new `scripts/run_relational_robustness.sh` |
| Tests | `tests/test_cross_model_robustness_v2.py`, new `tests/test_cross_model_reanalysis.py`, `tests/test_cross_model_robustness_workflow.py` |
| Documentation | `README.md`, `docs/cross_model_relational_v2.md` |

Preparation and report artifacts remain under the repository's existing ignored `outputs/` tree; completed artifacts were not overwritten.

## Exact commands

Run at repository root. These preparation/reanalysis commands load no model weights:

```bash
# All available non-GPU tests and shell syntax checks.
.venv/bin/python -m pytest -q
bash -n scripts/run_relational_robustness.sh

# Use fresh paths; existing artifacts are never overwritten.
.venv/bin/python -m scripts.robustness_v2 audit --stage development \
  --output outputs/cross_model_relational_v2/corrected_20261003/prompt_audit.json
.venv/bin/python -m scripts.robustness_v2 preview --stage development \
  --output outputs/cross_model_relational_v2/corrected_20261003/development_preview.json
.venv/bin/python -m scripts.robustness_v2 preview --stage frozen_gate \
  --output outputs/cross_model_relational_v2/corrected_20261003/gate_preview.json

# Review actual baseline/edit pairs before freezing. The freeze includes all
# three configs, original-v1 lineage, the ledger and code/audit hashes.
.venv/bin/python -m scripts.robustness_v2 freeze \
  --audit outputs/cross_model_relational_v2/corrected_20261003/prompt_audit.json \
  --output outputs/cross_model_relational_v2/corrected_20261003/protocol_freeze.json

# Stage 0: pinned tokenizer/config only. Use LOCAL_FILES_ONLY=1 for cached audits.
RELATIONAL_LOCAL_FILES_ONLY=1 bash scripts/run_relational_robustness.sh qwen3_8b validate
RELATIONAL_LOCAL_FILES_ONLY=1 bash scripts/run_relational_robustness.sh phi4_mini validate
# Gemma requires its actual pinned tokenizer to be available on the host.
bash scripts/run_relational_robustness.sh gemma3_4b validate

# Actual development dataset token/geometry audit and forward-count plan;
# these generate no logits. Gate/confirmation are blocked until prior reports pass.
bash scripts/run_relational_robustness.sh qwen3_8b generate-development
bash scripts/run_relational_robustness.sh qwen3_8b plan

# Correct all three saved-score reports without new inference.
for model in qwen3_8b gemma3_4b phi4_mini; do
  .venv/bin/python -m scripts.robustness_v2 reanalyze \
    --dataset outputs/${model}_review2/confirmatory.jsonl \
    --scores outputs/${model}_review2/confirmatory_scores.jsonl \
    --output outputs/cross_model_v1_reanalysis_corrected_20261003/${model}.json
done
```

**The following commands perform inference and have not been executed. Do not begin v2 scoring while the readiness warning is in force.** On the configured CUDA host, after reviewing the corrected design/token audits and resolving readiness, the fixed behavioral sequence for each of `qwen3_8b`, `gemma3_4b`, `phi4_mini` is:

```bash
bash scripts/run_relational_robustness.sh qwen3_8b development
bash scripts/run_relational_robustness.sh qwen3_8b gate
# Inspect the frozen gate/preflight; a failed gate blocks this command.
bash scripts/run_relational_robustness.sh qwen3_8b confirmatory
# Separate recomputation from saved v2 confirmation scores; no inference.
bash scripts/run_relational_robustness.sh qwen3_8b analyze
```

`RELATIONAL_RUN_DIR` selects a fresh model output directory; it does not bypass one-per-model/stage claims. `RELATIONAL_FREEZE` selects the immutable shared freeze path. To resume interrupted scoring, use the CLI `score --resume` with its original config/candidate/dataset/output paths. No gate thresholds, vocabulary, counts, seeds or wording have CLI overrides. There is no Mistral replacement, downstream multi-model panel, new mechanism experiment, or long-context expansion. Optional downstream order/codebook robustness remains deferred and isolated from these commands.
