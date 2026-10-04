# Paper claims and evidence limits

This document records claim boundaries across the project's distinct protocols. The current LaTeX manuscript is *What a Stale Value Still Does: Mention Structure and Order Explain Score Sensitivity After In-Context Updates* (`paper/main.tex`); its primary story is the relational/order v2 comparison and the separate marker confirmation. Earlier v1 and mechanistic results remain protocol-specific context and are not pooled with v2.

## Primary claim

In relational/order v2, matched edits causally change bounded candidate scores, and query-specific score sensitivity varies across the tested prompt constructions and orders. The prespecified three-control relation-selective hypothesis failed because entity-mention relevance exceeds superseded relevance in both Qwen3-8B and Gemma 3 4B. Unassigned-value relevance reverses sign across order groups in both models, while the entity-mention order effect has opposite directions across models. The order-averaged superseded-minus-other-attribute contrast is small relative to the entity-mention contrast, but Qwen's attribute-specific relation contrasts vary in sign. These results show that the score measure is construction- and order-sensitive; they do not decompose independent contributions or identify an internal obsolete-binding mechanism. The separate marker confirmation supports an effect of adding the explicit `Previously` prefix on measured query-specific score sensitivity, not active suppression.

The newer Qwen `cross_model_v1` run estimates `R_superseded - R_irrelevant_counterbalanced = 3.50` nats (95% history-bootstrap CI `[3.20, 3.80]`), positive in 94/96 histories. Gemma's corresponding estimate is 1.76 nats (CI `[1.51, 2.03]`; 88/96 positive), and Phi's is 5.53 nats (CI `[5.27, 5.79]`; 96/96 positive). These model-specific results support the narrow value-occurrence claim above. The older 4.52-logit result is discussed only as provenance-limited historical context below; single-token logits and bounded continuation mass are different estimands and must not be combined or treated as numerically interchangeable.

The original Qwen 4.52-logit result is a provenance-limited historical note: its raw datasets, scores, gate reports, and provenance were not recovered in the documented repository/local search. It is not primary evidence and its full lineage is not independently verified. Retain the traceable, newer Qwen `cross_model_v1` run as the primary v1 evidence; do not combine the distinct estimands.

Gemma 3 4B also passed its frozen `cross_model_v1` competence gate and confirmatory run: `R_superseded - R_irrelevant_counterbalanced = 1.76` nats (95% history-bootstrap CI `[1.51, 2.03]`), positive in 88/96 histories. This is a second-family behavioral/causal replication under the bounded surface-class continuation-mass estimand. Its estimate is smaller than Qwen's 3.50 nats, but raw effect sizes are not architecture-invariant. Gemma's `R_live = 34.58` nats, `R_superseded = 1.66` nats, and `R_irrelevant_counterbalanced = -0.10` nats; `R_live - R_superseded = 32.92` nats. Report the models separately.

Phi-4-mini has now also passed its frozen gate and confirmatory run: `R_superseded - R_irrelevant_counterbalanced = 5.53` nats (95% history-bootstrap CI `[5.27, 5.79]`), positive in 96/96 histories. `R_live = 17.14`, `R_superseded = 5.39`, and `R_irrelevant_counterbalanced = -0.14` nats; `R_live - R_superseded = 11.75` nats (CI `[11.43, 12.06]`), positive in 96/96 histories. The primary contrast is positive in both literal-orientation strata, with noticeable heterogeneity by semantic variable. This is a third-family behavioral/causal replication under the bounded surface-class continuation-mass estimand; it does not establish shared mechanism.

## Relational/order v2 follow-up

The frozen Qwen3-8B and Gemma 3 4B confirmation analyses each include 96 histories and use 2,000 history-bootstrap draws (seed 73021). Per the dated analysis amendment, the full three-control claim requires the lower 95% interval bound to exceed zero for all three superseded-minus-control contrasts. Neither model meets that rule:

| Contrast (`R_superseded − R_control`) | Qwen3-8B, nats (95% CI) | Gemma 3 4B, nats (95% CI) |
|---|---:|---:|
| Early unassigned | 2.011 `[1.855, 2.170]` | 1.575 `[1.440, 1.719]` |
| Entity mention | -3.504 `[-3.806, -3.198]` | -2.589 `[-2.829, -2.361]` |
| Other attribute | 0.314 `[0.147, 0.482]` | 0.004 `[-0.104, 0.118]` |

Both models exceed the early-unassigned control overall, but the entity-mention control exceeds the superseded condition. Qwen also exceeds the other-attribute control overall; Gemma's contrast is inconclusive. Under reversed order, superseded-minus-early-unassigned remains positive in both models (Qwen 5.130, CI `[4.920, 5.348]`; Gemma 4.044, CI `[3.810, 4.292]`), while reversed-order contrasts against entity mention remain negative and contrasts against other attribute include zero. Under aligned order, superseded-minus-early-unassigned is negative for both models (Qwen -1.108, CI `[-1.281, -0.925]`; Gemma -0.895, CI `[-1.084, -0.719]`). The result is sensitive to order and does not establish relation-selective supersession. Report models separately; do not pool raw nats.

## Supporting claims

- Live bindings have substantially greater causal relevance than superseded bindings.
- In the tested construction, the obsolete value occurrence has greater query-specific relevance than the particular counterbalanced unassigned-value control.
- The effect occurs in controlled natural language.
- Obsolete bindings remain historically addressable.
- Existing Qwen patching localizes causal influence near the historical value token early and at the final query/readout computation late. Gemma's 12-history mechanism battery finds positive superseded-value relevance at early historical-value positions and at late pre-answer states. Its planned early-historical-minus-readout window contrast is positive (1.53 nats, 95% CI `[1.08, 2.04]`), but its late-readout-minus-historical contrast is inconclusive and trends negative (-0.16 nats, 95% CI `[-0.48, 0.19]`). This supports late readout causal relevance, but not a consistent depth-window shift in Gemma. These are qualitative localizations, not literal movement of information or evidence for a dedicated stale-information circuit. Qwen's current head results are more consistent with generic binding retrieval.
- The bounded surface-class behavioral/causal effect has now replicated in Qwen3-8B, Gemma 3 4B, and Phi-4-mini. This supports generalization to two additional model families, not broad architecture-level generalization. Phi's mechanistic result is still pending. Mistral-7B-Instruct-v0.3 is gate-ineligible under this protocol; its failed gate is not a negative causal result.

## Protocol history and provenance

The dated v1 audit records that the earlier exact-token gates exposed a casing/tokenization discrepancy for Phi and genuine candidate-ranking errors for Mistral. The surface-class continuation protocol and fresh competence gates were introduced to score complete candidate continuations under a fixed, tokenizer-audited event definition and to avoid treating a first-token fragment as a complete answer. The old gate outcomes remain preserved and are not relabeled. The repository records protocol decisions and artifact dates; it does not establish external preregistration or support a stronger claim about when causal outcomes were inspected than the dated records show. See [the v1 audit](cross_model_v1_audit.md).

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
