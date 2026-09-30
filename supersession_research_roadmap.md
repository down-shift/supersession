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

## 5. Next primary direction: version-chain / recency structure

- [ ] **Build audited version-chain dataset** — **NEXT**
  - Example: `x=v0 -> v1 -> v2 -> ... -> vk`.
  - Keep the same paired counterfactual-edit methodology and token audits.
  - Prefer depths such as 1, 2, 4, 8 if competence remains high.

- [ ] **Measure causal relevance of every version**
  - Independently edit each historical value.
  - Measure query-specific causal relevance \(R_i\) for each version.

- [ ] **Test competing hypotheses**
  - **Discrete status hypothesis:** current value is strongly privileged, old versions are similarly attenuated.
  - **Recency-gradient hypothesis:** causal relevance decreases smoothly with version age.
  - **Hybrid hypothesis:** sharp current-vs-old gap plus residual age gradient among obsolete versions.

- [ ] **Check behavioral competence across chain depth**
  - Confirm that the model still selects the latest value reliably.
  - Track current-vs-previous and current-vs-oldest logit margins.

- [ ] **Counterbalance order / variable / lexical factors**
  - Two variables.
  - Query focal vs distractor variable.
  - Multiple literal variable-name assignments if useful.
  - Preserve history-level paired analysis.

## 6. Mechanistic follow-up after version-chain behavior

Run only if the chain experiment gives a clear behavioral structure.

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

- [ ] **Controlled natural-language replication**
  - Replace symbolic assignments with simple entity-attribute updates.
  - Replicate stale-vs-counterbalanced-irrelevant causal relevance.

- [ ] **Second-model behavioral replication**
  - Optional but valuable if compute/time permit.
  - Behavioral replication is higher priority than full mechanistic replication.

## 8. Paper-level interpretation

### Supported now

- [x] Obsolete contextual bindings are not causally erased after supersession.
- [x] Their influence remains query/binding-specific.
- [x] This effect survives a counterbalanced irrelevant-occurrence control.
- [x] Stale information becomes causally available at the late query/readout state.
- [x] Prominent late heads are more consistent with generic binding retrieval than with a stale-specific mechanism.

### Not established yet

- [ ] Why the current value wins over still-retrievable obsolete values.
- [ ] Whether obsolete versions form a discrete “inactive” class or a graded recency hierarchy.
- [ ] A dedicated supersession/version-selection circuit.
- [ ] Strong generalization beyond the controlled symbolic Qwen3 setup.

## Immediate priority order

1. [ ] Implement the audited **version-chain causal relevance experiment**.
2. [ ] Run a small competence + pilot check.
3. [ ] Run the full paired behavioral version-depth experiment.
4. [ ] Decide between discrete-status vs recency-gradient vs hybrid interpretation.
5. [ ] Only then choose the next mechanistic localization experiment.
6. [ ] Add one natural-language replication before finalizing the paper.
