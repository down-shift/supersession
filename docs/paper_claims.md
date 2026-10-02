# Paper claims and evidence limits

This document fixes the intended claim hierarchy for *Supersession Without Erasure*. Results are model- and protocol-specific; a competence-gate pass alone is not causal evidence.

## Primary claim

After successful in-context supersession, obsolete bindings retain attenuated but nonzero binding-specific causal relevance.

The established Qwen3-8B controlled-natural-language result supports this claim: in the 96-history `nora_v1` analysis, `R_superseded - R_irrelevant_counterbalanced = 4.52` logits (95% history-bootstrap CI `[4.12, 4.94]`), positive in 96/96 histories. The newer `cross_model_v1` Qwen run supports the same qualitative contrast under bounded surface-class continuation mass: 3.50 nats (95% history-bootstrap CI `[3.20, 3.80]`), positive in 94/96 histories. These are distinct estimands and must not be combined or presented as numerically interchangeable.

The original Qwen 4.52-logit result remains an earlier canonical result with a lineage limitation: its raw datasets, scores, gate reports, and provenance were not recovered in the documented repository/local search. Do not describe its full lineage as independently verified. The new gated Qwen `cross_model_v1` run is the reproducible cross-model anchor; retain the original result with this qualification unless its original artifacts and hashes are recovered from backup.

Gemma 3 4B also passed its frozen `cross_model_v1` competence gate and confirmatory run: `R_superseded - R_irrelevant_counterbalanced = 1.76` nats (95% history-bootstrap CI `[1.51, 2.03]`), positive in 88/96 histories. This is a second-family behavioral/causal replication under the bounded surface-class continuation-mass estimand. Its estimate is smaller than Qwen's 3.50 nats, but raw effect sizes are not architecture-invariant. Gemma's `R_live = 34.58` nats, `R_superseded = 1.66` nats, and `R_irrelevant_counterbalanced = -0.10` nats; `R_live - R_superseded = 32.92` nats. Report the models separately.

Phi-4-mini has now also passed its frozen gate and confirmatory run: `R_superseded - R_irrelevant_counterbalanced = 5.53` nats (95% history-bootstrap CI `[5.27, 5.79]`), positive in 96/96 histories. `R_live = 17.14`, `R_superseded = 5.39`, and `R_irrelevant_counterbalanced = -0.14` nats; `R_live - R_superseded = 11.75` nats (CI `[11.43, 12.06]`), positive in 96/96 histories. The primary contrast is positive in both literal-orientation strata, with noticeable heterogeneity by semantic variable. This is a third-family behavioral/causal replication under the bounded surface-class continuation-mass estimand; it does not establish shared mechanism.

## Supporting claims

- Live bindings have substantially greater causal relevance than superseded bindings.
- Superseded bindings have greater relevance than counterbalanced irrelevant information.
- The effect occurs in controlled natural language.
- Obsolete bindings remain historically addressable.
- Existing Qwen patching localizes causal influence near the historical value token early and at the final query/readout computation late. Gemma's 12-history mechanism battery finds positive superseded-binding relevance at early historical-value positions and at late pre-answer states. Its preregistered early-historical-minus-readout window contrast is positive (1.53 nats, 95% CI `[1.08, 2.04]`), but its late-readout-minus-historical contrast is inconclusive and trends negative (-0.16 nats, 95% CI `[-0.48, 0.19]`). This supports late readout causal relevance, but not a consistent depth-window shift in Gemma. These are qualitative localizations, not literal movement of information or evidence for a dedicated stale-information circuit. Qwen's current head results are more consistent with generic binding retrieval.
- The bounded surface-class behavioral/causal effect has now replicated in Qwen3-8B, Gemma 3 4B, and Phi-4-mini. This supports generalization to two additional model families, not broad architecture-level generalization. Phi's mechanistic result is still pending. Mistral-7B-Instruct-v0.3 failed its gate; that failure is not evidence that the causal effect is absent.

## Downstream transfer evidence

The Qwen3-8B `downstream_transfer_v1` confirmation supports the scoped claim that manipulating a superseded contextual binding causally shifts preference in a derived-code representation, with the shift stronger for the associated entity's current query than for the other entity's query, while the current binding and correct derived code stay fixed. The primary history-level contrast was 5.54 nats (95% history-bootstrap CI [5.16, 5.94]), positive in 96/96 histories; the live-binding control was 12.84 nats (CI [12.42, 13.26]), also positive in 96/96.

Current-code candidate-ranking accuracy across stale pairs was 93.75% before and 94.79% after editing. Paired correct-code log-probability and margin intervals included zero; this is not proof of zero interference. The confirmatory run did not measure unrestricted generation. The gate's unrestricted greedy-generation diagnostic yielded 0/48 exact code strings because outputs were Markdown-bold; removing the markers gives the correct code in 48/48. This diagnostic was not part of the frozen gate rule. Do not claim that obsolete information caused overt current-state errors, that the model failed to forget, or that this result generalizes beyond this model/task without further evidence.

Gemma 3 4B failed its frozen downstream-transfer competence gate at 45/48 (93.75%; requirement ≥47/48), and Phi-4-mini failed at 33/48 (68.75%). Both sealed gate artifacts record `pass: false`; neither model has downstream-transfer confirmation or causal-analysis outputs. These failed extension gates do not alter the separate `cross_model_v1` bounded continuation-mass confirmatory results for Gemma and Phi, and they provide no causal downstream-transfer evidence for either model. Keep the Qwen downstream-transfer result secondary and model/task-specific.

## Interpretation limits

- Report the original Qwen logit estimand and the `cross_model_v1` bounded surface-class continuation-mass estimand separately.
- Gemma's mechanism results use int8 weights, bfloat16 activations, and 12 histories; no Gemma full-precision sensitivity run or head profiling was performed. Treat the component/localization results as model- and setup-specific.
- Phi's confirmatory result uses int8 weights and float16 activations. Its real pinned remote-model CUDA hook smoke and mechanistic battery remain outstanding; do not infer mechanistic localization from behavioral results.
- A competence-gate pass establishes eligibility for causal testing, not supersession-without-erasure.
- The downstream-transfer result is one Qwen3-8B task/protocol result. Candidate-sequence preference is not the same measure as unrestricted answer generation.
- Do not infer a dedicated circuit, suppression mechanism, or homologous heads from behavioral effects or qualitative patching localization.
- The original exact-token Mistral and Phi gates remain failed; the later `cross_model_v1` Phi pass and Mistral failure have their own protocol status and do not rewrite those records.
