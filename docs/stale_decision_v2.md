# Stale-decision manuscript guide

The current manuscript is **Obsolete and Incidental Words Shift Language-Model Decision Preferences**, in [`paper_v2/main.tex`](../paper_v2/main.tex). This guide summarizes its current evidence and reporting rules from the manuscript and committed derived data. The dated experiment decisions are recorded in [Appendix A](../paper_v2/sections/A-protocol-history.tex); this guide is not a new protocol amendment.

## Design and measurements

Each synthetic record has two entities, a state update and an explicit policy. Routing maps a state through a bin to `EAST`/`WEST`; access maps clearance to `APPROVE`/`DENY`. An edit changes one historical word between a state implying the wrong action and one implying the correct action, while holding the current state, policy, query and correct action fixed. The same edit is applied to a superseded assignment (S), an updated other attribute (O), an incidental entity mention (M), and an unattached word (U). Current-only decisions and edits to the live state supply controls. Sequential, interleaved and competing cells use both natural and opaque state labels.

`D_c` is the mean paired change in wrong-minus-correct action log probability, in nats: `mean(L(old-wrong) - L(old-correct))`, where `L = log P(wrong action) - log P(correct action)`. Action events include the generated label and its end-of-turn suffix. The net switch rate is the mean paired difference in the indicator for a wrong generated action; it is not the total error rate. Invalid outputs count as not wrong for this indicator and as failures for H3. Retrieval sensitivity `R_easy` instead uses bounded answer-prefix masses. See [the design section](../paper_v2/sections/03-design.tex).

## Main evidence and populations

Qwen3-8B confirmation contains 768 histories (384 per task), 8,448 records, and held-out introduction/query wording. Development and gate have 192 and 384 histories respectively. The main design always makes the other entity's current state imply the wrong action. Thus a decision error can reflect interaction between the edited word and entity confusion.

The following routing score and switch estimates use cells selected by the **frozen gate**, as in the manuscript's main cross-model table. The tracked-history column uses a separate post hoc restriction across all 384 routing histories per model: correct queried-current retrieval under both edited members, correct initial retrieval under both superseded members, and a correct current-only decision. These populations differ.

| Model | Gate-selected routing cells | Histories in compact estimate | `D_S` (nats) | Net S switch | Tracked net S switch (retained histories) |
|---|---|---:|---:|---:|---:|
| Qwen3-8B | interleaved, sequential | 256 | 3.248 | 0.195 | 0.158 (348) |
| Qwen3-14B | competing, interleaved, sequential | 384 | 4.785 | 0.167 | 0.165 (376) |
| Gemma 4 12B | competing, interleaved, sequential | 384 | 3.949 | 0.148 | 0.143 (349) |
| Qwen3.5 9B | competing, sequential | 256 | 1.085 | 0.230 | 0.233 (326) |
| Granite 4.1 8B | interleaved, sequential | 256 | 3.441 | 0.168 | 0.145 (331) |

Access meets confirmation competence only in Gemma 4 12B (all three cells; `D_S = 4.285`, net S switch `0.083`). The primary gate-selected access estimates are Qwen3-8B interleaved (`D_S = 4.221`, net S switch `0.273`) and Gemma 4 12B competing (`D_S = 4.384`, net S switch `0.062`). Qwen3-8B fails confirmation competence in that cell. Its additional access estimates (`D_S = 3.904`, net S switch `0.167`) average the full fixed grid. Gemma 3 4B and Ministral 3 14B fail every gate cell and have no confirmation. OLMo 3 7B was stopped at the numerical preflight, before development or gate scoring.

Sources: [main cross-model table](../paper_v2/data/tables/cross_model.tex), [tracked-history table](../paper_v2/data/tables/joint_by_model.tex), and [results](../paper_v2/sections/05-results.tex).

The [results index](../paper_v2/RESULTS_INDEX.md) maps the remaining tables and figures to committed derived-data keys.

## What the controls change

The paper's central finding is that obsolete and incidental words shift decision preferences across the tested models. The follow-ups describe how generated decisions vary across tested conditions. In Qwen3-8B, new balanced routing histories make the other entity's current state imply the correct action. The superseded preference shift remains positive (`3.895` nats), while its net switch rate falls from `0.185` in the main design to `0.031`; mention switches remain `0.177`. The histories differ and identical current states may simplify the task, so this comparison is descriptive. Thinking-enabled generation on the original routing rows gives zero superseded switches in both Qwen3 models; it has no preference scores. Qwen3-8B still has seven wrong-directed mention switches (`0.018`) and two invalid answers. Thinking is a different generation process, and no statistical test was prespecified. See [the conditions table](../paper_v2/data/tables/conditions.tex).

Interference does not require supersession or assignment to the queried attribute. The superseded assignment exceeds the updated other attribute in routing scores in every confirmed model, but a fifth construction leaves that other attribute **never updated**. Updating it reduces interference by `1.204` and `1.942` nats in the two Qwen3 models. The never-updated other attribute is unresolved from S in Qwen3-8B and exceeds S in Qwen3-14B. Therefore `Delta_SO` does not isolate assignment to the queried attribute. See [the never-updated table](../paper_v2/data/tables/static_other.tex).

An incidental entity mention shifts preferences in every confirmed model. It produces more generated routing switches than S in Qwen3-8B and Granite 4.1 8B; the switch differences are unresolved in Qwen3-14B and Qwen3.5 9B. Gemma 4 12B has a larger S score effect and more S switches on gate-selected cells; its post hoc confirmation-selected switch difference is unresolved. Mention superiority is model-dependent.

H3 adds easy-query sensitivity to fixed pre-decision features in history-grouped five-fold cross-validation within confirmation. The predictive improvement is unresolved in every confirmed model. This supports the scoped negative result for the specified model and outcome; it does not establish external prediction performance or rule out other diagnostics.

## Reporting and runtime

Competence requires direct current-state accuracy at least `0.95`, current-only decision accuracy at least `0.90`, and invalid-answer rates at most `0.05`. The frozen gate controls eligibility. The main compact table and model figure follow the amendment's original gate-selected rule. Appendix B reports the post hoc confirmation-selected estimates and competence discrepancies as a robustness comparison. Detailed full-grid estimates are labeled separately. Tracked-history analyses and their common-construction intersection are post hoc; retained sets differ by construction unless the common intersection is used.

H1 (`D_S > 0`) and H2 (`Delta_SO != 0`) use 97.5% history-bootstrap intervals with 2,000 draws, adjusted over the two tasks. Secondary contrasts, switches and H3 use 95% intervals. An interval including zero is unresolved. Tasks and models are reported separately. The internal records support “frozen before confirmation,” not external preregistration.

The reported runtime is unquantized float32, SDPA, one 80 GB H100, with thinking disabled except in the thinking follow-up. Direct generation is greedy with at most 16 new tokens; thinking generation allows 4,096. Reduced-precision audits found score changes when tokens were appended after a fixed position (up to `0.843` nats with int8 and `0.875` with bfloat16), prompting the float32 runtime. The KV-cache scorer was checked against full forwards. OLMo's `2.8e-5` causality deviation failed its exactly-zero preflight rule; the balanced run's one `5.7e-5` deviation was accepted under that runner's different tolerance. The paper discloses both rules.

The evidence is limited to short synthetic tasks in five open models from three developers. Construction syntax differs, and correct retrieval does not establish which state the model used at decision time. The balanced comparison is Qwen3-8B-only; never-updated and thinking controls cover the two Qwen3 models. No internal mechanism or mitigation efficacy is established. See [limitations](../paper_v2/sections/08-limitations.tex) and [reproduction instructions](../paper_v2/README.md).
