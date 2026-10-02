# Supersession Research Roadmap

## 1. Core behavioral finding

- [x] **Establish behavioral supersession competence** — **PASSED**
  - Model reliably answers with the current value after overwrite.
  - Current value beats stale value by a large logit margin.

- [x] **Show obsolete bindings remain causally influential** — **PASSED**
  - Counterfactual edits to obsolete values still affect output logits.
  - Query-specific stale relevance \(R_{\text{stale}}\) is strongly positive.

- [x] **Rule out simple lexical-presence explanation** — **PASSED**
  - Counterbalanced irrelevant control reduces query-specific relevance to ~0.
  - Superseded bindings remain strongly above the irrelevant baseline.

- [x] **Diagnose fixed-order irrelevant-control artifact** — **PASSED**
  - Original irrelevant effect was largely caused by mention position.
  - Reversing mention order flips the sign.

## 2. Mechanistic localization

- [x] **Residual-stream patching** — **PASSED**
  - Stale identity is causally present in the late pre-answer representation.

- [x] **Layer × position localization** — **PASSED**
  - Early causal influence is strongest at the historical value token.
  - Late causal influence is strongest at the final query/readout position.
  - Interpretation: causal localization shifts with depth; do not claim literal information “movement.”

- [x] **Late block-component patching** — **PASSED / INTERPRET WITH CAUTION**
  - Late attention-output interventions have large downstream effects.
  - Not an additive attention-vs-MLP decomposition.

## 3. Attention-head analysis

- [x] **Exploratory late-head scan** — **PASSED**
  - Several heads carry substantial stale-binding relevance.

- [x] **Named-head functional profiling** — **PASSED**
  - L32H8, L34H1, L34H28 carry stale identity.
  - Their historical/current effects are stronger than stale effects.
  - Main interpretation: these heads look more like **generic binding-retrieval heads** than supersession-specific heads.

- [x] **Negative-control head inspection** — **PASSED**
  - Some heads show consistently negative relevance contrasts.
  - Do not interpret them as suppression heads yet.

- [ ] **Reserve confirmation of head set** — **DEFERRED**
  - Do only if the final paper still relies heavily on the generic-retrieval-head claim.

- [ ] **Q/K/V decomposition** — **DEFERRED**
  - Not justified until the version-selection mechanism is better localized.

## 4. Semantic-validity / status experiments

- [x] **Global accepted/rejected status experiment** — **SUGGESTIVE BUT CONFOUNDED**
  - Accepted proposed values became stronger.
  - Rejected proposals became weaker.
  - Retained initial values became stronger after rejection.
  - Confounded by global status manipulation and position/order structure.

- [x] **Independent YY/YN/NY/NN design** — **FAILED COMPETENCE GATE**
  - Strong failure in the NY condition, especially focal-variable queries.
  - Semantic role remained partly confounded with block position.

- [x] **Focal APPLIED/IGNORED design** — **FAILED COMPETENCE GATE**
  - Model handled APPLIED cases and IGNORED-other queries well.
  - Model failed badly on **IGNORED × focal-query** cases.

- [x] **Persistence-explicit focal prompt redevelopment** — **FAILED**
  - `persistence_rule` and `accept_reject_rule` still failed required competence cells.
  - Do not continue prompt tuning unless there is a strong new reason.

- [ ] **Focal confirmatory causal experiment** — **ABANDONED FOR NOW**
  - Do not run without a passing competence gate.

- [ ] **Validity residual patching** — **ABANDONED FOR NOW**
  - Do not run without a valid behavioral manipulation.

## 5. Version-chain / recency structure

- [x] **Audited multi-depth version-chain pilot and exploratory full run**
  - The two-updated-variable design passed competence only at depth 1; do not use it to answer the multiple-obsolete question.
  - The single-deep-chain pilot qualified depth 2, so one full run was conducted at depth 2.
  - That full run failed the fixed per-cell 98% competence gate: baseline focal accuracy was 94/96 (97.92%).

- [x] **One final fresh replication, frozen design** — **RECENCY PATTERN REPLICATED; COMPETENCE GATE FAILED**
  - Same depth-2 prompt, token map, model/tokenizer revision, and analysis; seed `20261104`; 96 fresh histories excluding pilot and exploratory-full histories.
  - Full run baseline accuracy was 100% for focal and distractor queries. Edited focal accuracy was 382/384 (99.48%), but two individual cells had 23/24 correct (95.83%), below the unchanged 98% cell threshold. No depth qualified.
  - The two edited-prompt misses were both focal-z queries under swapped literal names: one stable-control edit elicited the previous answer; one previous-version edit elicited the edited previous value.
  - Exploratory $R_i$ means: current 39.00 [37.64, 40.42], previous 17.93 [16.58, 19.21], oldest -8.40 [-9.35, -7.45]. Intervals are history-bootstrap 95% CIs.
  - Preregistered previous-minus-oldest contrast: 26.33 [24.81, 27.67], positive in 96/96 histories.
  - Current-minus-previous contrast: 21.07 [19.54, 22.76], positive in 96/96 histories.
  - Both contrasts were positive in focal-x and focal-z strata and in both literal-name orientations.
  - These estimates reproduce the exploratory-full pattern, but the final run failed competence. Label both runs exploratory; they do not establish a competence-qualified confirmatory result.

- [x] **Version-chain branch decision: STOP**
  - The one permitted fresh replication has been completed and failed the fixed competence gate.
  - Do not run a third replication, tune wording, or lower the threshold.
  - Preserve both full runs and their analyses as exploratory evidence of a replicated recency pattern.
  - Do not proceed to version-age mechanistic patching or head/QKV follow-up on this branch.

## 6. Mechanistic follow-up

The version-chain branch stopped after its single allowed replication failed the competence gate. The following version-age follow-ups are not authorized by the present evidence and remain deferred.

- [ ] **Layer × position patching by version age**
  - Ask whether older and newer obsolete bindings converge onto the same late retrieval pathway.

- [ ] **Version-age effect on current-vs-old decision margin**
  - Measure how interventions change:
    \[
    S_i = \logit(current) - \logit(version_i)
    \]

- [ ] **Compare generic retrieval heads across version age**
  - Test whether L32H8 / L34H1 / L34H28 retrieve all historical versions similarly or with a recency gradient.

- [ ] **Targeted component/head localization**
  - Only after a clear version-age or current-vs-obsolete selection signal is localized.

## 7. Generalization / paper robustness

- [x] **Controlled natural-language replication** — **CORE CONTRAST REPLICATED; COMPETENCE GATE PASSED**
  - 96 fresh histories with simple entity–attribute updates, one frozen template (`nora_v1`), and the validated 12-token candidate universe.
  - Primary history-bootstrap contrast: `R_superseded − R_irrelevant_counterbalanced = 4.52` logits, 95% CI [4.12, 4.94], positive in 96/96 histories.
  - Descriptive live-minus-superseded contrast: 18.48 logits, 95% CI [17.30, 19.57], positive in 96/96 histories.
  - Full-vocabulary accuracy ranged from 99.61% to 99.87% in the substantive conditions; mean target rank was approximately 1.
  - The ordinary uncounterbalanced irrelevant estimate was 3.44 [2.94, 3.92], while the counterbalanced estimate was 0.07 [-0.29, 0.41]. Reversing mention order reverses the apparent positional preference, confirming why the counterbalanced control is needed.
  - Scope: one model (Qwen3-8B), one selected English template family, and 12 validated values. Treat this as a successful controlled natural-language replication, not broad language generalization.
  - Analysis: `outputs/supersession/nl_confirmatory_analysis/summary.json`; preserve the dataset, scores, gate, and provenance sidecars with the analysis.

- [x] **Cross-model behavioral and causal replication** — **CONFIRMED IN GEMMA 3 4B AND PHI-4-MINI; LIMITED PANEL EVIDENCE**
  - Gemma 3 4B passed its frozen `cross_model_v1` gate and completed 96-history confirmation using bounded surface-class continuation mass: `R_superseded - R_irrelevant_counterbalanced = 1.76` nats, 95% history-bootstrap CI `[1.51, 2.03]`, positive in 88/96 histories.
  - `R_live=34.58`, `R_superseded=1.66`, `R_irrelevant_counterbalanced=-0.10`; live-minus-superseded was 32.92 nats, CI `[31.55, 34.20]`, positive in 96/96 histories.
  - Keep this estimate distinct from Qwen's 3.50-nat cross-model estimate and original 4.52-logit natural-language result; raw effect sizes are not architecture-invariant.
  - Mistral-7B-Instruct-v0.3 failed its frozen gate; this is not evidence that the causal effect is absent.
  - Analysis: `outputs/gemma3_4b_review2/confirmatory_analysis.json`; mechanism report: `outputs/gemma3_4b_review2/mechanism.jsonl.analysis.json`.
  - Phi-4-mini passed its gate and completed confirmation: primary `R_superseded - R_irrelevant_counterbalanced = 5.53` nats, 95% history-bootstrap CI `[5.27, 5.79]`, positive in 96/96 histories. `R_live=17.14`, `R_superseded=5.39`, `R_irrelevant_counterbalanced=-0.14`; live-minus-superseded was 11.75 nats, CI `[11.43, 12.06]`, positive in 96/96 histories. The effect varied by semantic variable (x 2.30; z 8.75 nats).
  - Phi used int8 weights and float16 activations. Its confirmatory result does not establish a mechanism; real pinned-model hook smoke and mechanistic analysis remain pending. Analysis: `outputs/phi4_mini_review2/confirmatory_analysis.json`.

- [ ] **Broaden model-family replication** — **OPTIONAL NEXT STEP**
  - Any additional model must follow the frozen protocol and model panel policies. Do not pick a model based on its causal result.

## 8. Paper-level interpretation

### Supported now

- [x] Obsolete contextual bindings are not causally erased after supersession.
- [x] Their influence remains query/binding-specific.
- [x] This effect survives a counterbalanced irrelevant-occurrence control.
- [x] The superseded-versus-counterbalanced-irrelevant contrast generalizes to one controlled natural-language entity–attribute template with high task competence.
- [x] Stale information becomes causally available at the late query/readout state.
- [x] Prominent late heads are more consistent with generic binding retrieval than with a stale-specific mechanism.
- [x] The bounded surface-class causal contrast replicates in two additional families (Gemma 3 4B and Phi-4-mini). Gemma's mechanistic battery finds early historical-position and late readout relevance but does not establish the preregistered broad depth-window shift; Phi mechanism remains untested.

### Not established yet

- [ ] Why the current value wins over still-retrievable obsolete values.
- [ ] Whether obsolete versions form a discrete “inactive” class or a graded recency hierarchy.
- [ ] A dedicated supersession/version-selection circuit.
- [ ] Broad generalization across model families and natural-language formulations. Current cross-model confirmatory evidence covers Qwen3-8B, Gemma 3 4B, and Phi-4-mini under one controlled English protocol.

## Immediate priority order

1. [x] Complete and stop the version-chain branch after its frozen replication failed the competence gate; retain its estimates as exploratory.
2. [x] Complete the controlled natural-language replication and confirm the superseded > counterbalanced-irrelevant result under the frozen competence gate.
3. [ ] Complete Phi-4-mini's real pinned-model hook smoke and minimal mechanistic battery; its confirmatory behavioral result is already complete.
4. [ ] Consider broader model-family or language-formulation replication after reviewing the Qwen/Gemma/Phi evidence; any extension must follow the frozen protocol.
