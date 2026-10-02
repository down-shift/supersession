# Paper claims and evidence limits

This document fixes the intended claim hierarchy for *Supersession Without Erasure*. Results are model- and protocol-specific; a competence-gate pass alone is not causal evidence.

## Primary claim

After successful in-context supersession, obsolete bindings retain attenuated but nonzero binding-specific causal relevance.

The established Qwen3-8B controlled-natural-language result supports this claim: in the 96-history `nora_v1` analysis, `R_superseded - R_irrelevant_counterbalanced = 4.52` logits (95% history-bootstrap CI `[4.12, 4.94]`), positive in 96/96 histories. The newer `cross_model_v1` Qwen run supports the same qualitative contrast under bounded surface-class continuation mass: 3.50 nats (95% history-bootstrap CI `[3.20, 3.80]`), positive in 94/96 histories. These are distinct estimands and must not be combined or presented as numerically interchangeable.

## Supporting claims

- Live bindings have substantially greater causal relevance than superseded bindings.
- Superseded bindings have greater relevance than counterbalanced irrelevant information.
- The effect occurs in controlled natural language.
- Obsolete bindings remain historically addressable.
- Existing Qwen patching localizes causal influence near the historical value token early and at the final query/readout computation late. This is a qualitative localization pattern, not literal movement of information and not evidence for a dedicated stale-information circuit. Current head results are more consistent with generic binding retrieval.
- The effect generalizes beyond the original Qwen setup only if a preregistered cross-model confirmatory causal run succeeds. Phi-4-mini has passed the `cross_model_v1` semantic competence gate but has no confirmatory causal result. Mistral-7B-Instruct-v0.3 failed that gate; its failure is not evidence that the causal effect is absent. Gemma 3 4B remains pending in the last recorded status.

## Reserved for downstream transfer

Do not claim this from the current evidence:

> Obsolete bindings can causally influence computations derived from the obsolete state even while the model correctly uses the current state.

This requires downstream-transfer evidence.

## Interpretation limits

- Report the original Qwen logit estimand and the `cross_model_v1` bounded surface-class continuation-mass estimand separately.
- A competence-gate pass establishes eligibility for causal testing, not supersession-without-erasure.
- Do not infer a dedicated circuit, suppression mechanism, or homologous heads from behavioral effects or qualitative patching localization.
- The original exact-token Mistral and Phi gates remain failed; the later `cross_model_v1` Phi pass and Mistral failure have their own protocol status and do not rewrite those records.
