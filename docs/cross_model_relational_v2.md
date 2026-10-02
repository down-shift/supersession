# cross_model_relational_v2

This is a separate, frozen design extension. It does not change `cross_model_v1`, its configs, its output files, or its reported status. It reuses the shared eight-value vocabulary, pinned Qwen/Gemma/Phi revisions, eager/chat settings, candidate surface classes, and continuation-mass scorer. Seeds are fixed in `src/cross_model/robustness_v2.py`: development 20261031, gate 20261101, confirmation 20261102; counts are 24/24/96. All three model configs use the same semantic histories.

Each history has the six conditions `superseded`, `early_unassigned`, `late_unassigned`, `entity_mention`, `other_attribute`, and `live`, both entity queries, both edited historical values, and both donor directions. Live prompts state the historical assignments and reaffirm the same current assignments, so editing the historical mention preserves the queried current answer. Historical and current entity order are independently crossed; unassigned mention order is crossed with current order. Rows carry semantic condition, order, entity/query/edit identity, donor/source values, and pair identifiers. The generator is deterministic and history IDs include protocol and stage to make split overlap auditable.

The intended relevance estimand is the existing paired identity-transfer estimator from `src/cross_model/analysis.py`. For every condition/order cell, calculate the same replacement-minus-source paired effect and query-specific relevance contrast used in v1. First average the two historical-order levels and two current-order levels within history and condition; for unassigned controls, average mention order within each current-order level, then average current order. Then calculate the three preregistered superseded-minus-control contrasts, superseded versus late-unassigned, live attenuation, and aligned versus reversed order contrasts. Bootstrap histories (2,000 draws, seed 73021). Confirmation uses every trial irrespective of correctness.

## Status and limitation

The tokenizer-free generator and prompt audit are implemented, along with tests for deterministic generation, condition/order cells, and matched edit answers. It is not yet wired into the sealed `cross_model.py` prepare/gate/confirmatory scorer workflow. Therefore these generated data are design previews and are not eligible for model scoring or scientific use. A full implementation still needs tokenizer span-offset auditing, candidate-map sealing/provenance and stage claims, competence recomputation/gating, scorer integration, and the preregistered aggregation contrasts. Do not launch inference from these preview files.

The live condition edits a current value and therefore its answer changes in the edited member; the other five conditions edit a historical/control occurrence and preserve the current answer. This follows the positive-control logic in v1. The meaning of “both historical values / donor directions” is implemented as two historical entity values, each paired with its own fixed replacement donor; if “donor directions” is intended to mean reciprocal swaps between the two obsolete values, that changes the design and should be decided before the inference workflow is completed.

## Existing confirmation reanalysis

`scripts/robustness_v2.py reanalyze` reads saved v1 dataset and score JSONL files and writes a separate sealed descriptive artifact. It reports baseline/edited semantic accuracy and rank, paired current score and current-versus-strongest-alternative margin changes, absolute bounded prefix masses for current/source/replacement values, condition and orientation strata, and counterbalanced unassigned effects split by mention order. It refuses incomplete or mismatched saved rows and never performs inference. Its correctness-conditioned table is secondary. Relevance history aggregation follows the saved edit/query cells. Per-surface source/replacement log likelihoods and the frozen v1 normalized-relative-sensitivity guard are included. The new relational contrasts remain unavailable in historical v1 artifacts because those conditions were not collected.

## Commands currently supported

```bash
# Generate and print one representative prompt per condition (plain renderer)
.venv/bin/python scripts/robustness_v2.py audit --stage development --output /tmp/unused.json

# Save a deterministic design preview (does not tokenize or score it)
.venv/bin/python scripts/robustness_v2.py generate --stage development --output outputs/cross_model_relational_v2/development_preview.json

# Reanalyze one existing saved model confirmation without new inference
.venv/bin/python scripts/robustness_v2.py reanalyze \
  --dataset outputs/qwen3_8b_review2/confirmatory.jsonl \
  --scores outputs/qwen3_8b_review2/confirmatory_scores.jsonl \
  --output outputs/cross_model_v1_reanalysis/qwen3_8b.json
```

Full stage 0, development scoring, gate, and confirmation commands will be documented when the scoring/gate integration and token/provenance audits are implemented. Optional downstream robustness is deferred; no multi-model downstream panel is added.
