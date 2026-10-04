
# Counterfactual Score Shifts After Context Updates

\footnotetext{A large language model assisted with drafting and revision of this manuscript. The authors are responsible for checking the accuracy of its claims, analyses, references, and final text.}

## Abstract

Do earlier values continue to shape a language model's answer scores after a contextual update? We replace a historical value with a donor while holding the current assignment fixed, then compare candidate score changes for questions about the associated entity and another entity. In Qwen3-8B and Gemma 3 4B, the entity-mention construction yields greater query relevance than the superseded construction (5.478 versus 1.974 nats for Qwen; 4.277 versus 1.688 for Gemma). For unassigned values, query relevance is positive when historical and current mention orders align and negative when they are reversed. Superseded relevance stays positive in both order groups. The three-control prediction, fixed in an internal amendment before confirmation but not externally preregistered, is not supported. The current candidate remains top-ranked in every v2 superseded trial for both models. In an exploratory Qwen-only derived-code task, current-code rank-one accuracy was 93.8% before editing and 94.8% after; paired changes in current-code log probability and margin were inconclusive. Effects on unrestricted generation and reliable practical accuracy remain unresolved.

## 1. Introduction

Conversations often update a fact while leaving its earlier version in the context. A user's preference changes, an object moves, or a task variable receives a new value. A model may rank the current value first while its scores remain sensitive to an earlier value. We ask how the measured score shift varies across prompt constructions and with the order in which values and entities appear.

Suppose a history first states that Nora's badge was amber and later updates it to jade. Replacing `amber` with `violet` can shift the scores for those two candidates even if `jade` remains the preferred answer. The shift could depend on the old badge assignment, Nora's association with `amber`, or the value's position in the history. We compare prompt conditions that separate these possibilities as far as the current designs allow.

Consider an illustrative history from our second experiment:

```text
Previously, Nora’s badge was amber.
Previously, Liam’s badge was coral.
Currently, Nora’s badge is jade.
Currently, Liam’s badge is pearl.
What is Nora’s current badge?
Respond with only the value, with no explanation.
```

The correct answer is `jade`. We replace the earlier `amber` with `violet` and leave both current assignments unchanged. For each query, we measure the edit's shift in the score difference between `violet` and `amber`. We subtract the shift for the Liam query from the shift for the Nora query. Defined formally in Section 3.3, this contrast measures how much more the edit affects a query about the entity associated with the edited value. The example illustrates the calculation; it is not a scored dataset row.

In the first experiment, superseded assignments had greater query relevance than one counterbalanced unassigned control in three eligible model configurations. The comparison did not match entity association or position. The follow-up added early and late unassigned mentions, entity-associated notes without the target assignment, and earlier assignments to another attribute. It also varied historical and current order independently.

In both tested models, the entity-mention construction yields greater query relevance than the superseded construction. Effects for unassigned values reverse sign when relative mention order is reversed; superseded effects remain positive under both orders. The order pattern in the unassigned constructions is consistent with position-based association in this task. Because we measure output scores, the experiment does not test an internal binding mechanism.

The controlled comparison shows that positive query relevance occurs in several prompt constructions. The entity-mention construction yields a larger effect than the superseded construction, while order-stratified estimates show opposing effects for unassigned values that averaging conceals. The three-control prediction tests the original interpretation and fails because superseded relevance is lower than entity-mention relevance in both models. The main v2 estimates concern fixed answer-prefix scores. In the exploratory derived-code task, some current-code rankings changed in both directions, while paired changes in score and margin were inconclusive. Whether these score shifts affect unrestricted generation or reliable practical accuracy remains unresolved.

## 2. Related work

### 2.1 Contextual updates and state tracking

Entity-tracking and belief-revision studies evaluate whether models recover a current state after updates or revise conclusions when new evidence arrives (Kim and Schuster, 2023; Wilie et al., 2024; Wang and Sun, 2025; Tang et al., 2026; Qian et al., 2026). Guo et al. (2026) test stale-answer failures, whether current values remain recoverable, and attention drift during answer selection. Yang et al. (2026) study a broader taxonomy of conflicts within context and report a bias toward earlier evidence. These studies evaluate answer correctness or contextual conflict resolution. Our v2 experiment instead measures how a matched edit to one value changes bounded answer-prefix scores across controlled prompt constructions, with entity-mention and mention-order controls. It does not establish a new stale-binding mechanism or general evidence that older facts persist.

Where these studies evaluate update and conflict behavior through answer accuracy or contextual resolution, we hold current assignments fixed and measure counterfactual score shifts caused by replacing a historical value. We ask whether query relevance is greater for the superseded prompt construction than for early-unassigned, entity-mention, and different-attribute constructions. The results show that a positive query-specific score shift does not diagnose an obsolete target assignment: the entity-mention construction has greater relevance in both models, and unassigned effects reverse with relative mention order. Because the conditions also differ in syntax and predicates, the comparisons do not isolate an effect of relation status.

### 2.2 Persistence after knowledge editing

Knowledge-editing studies ask what remains after changing a model's response to a stored fact. Srivastava, Ai, and Chatterjee (2026) test whether original facts remain linearly decodable from hidden states after successful knowledge edits, including an external-memory edit that changes no base-model weights. Xie et al. (2025) study resurfacing of original parametric knowledge after editing and use causal interventions. Here, the earlier and current assignments remain visible in the same input, and we replace the earlier input value. We measure the resulting change in output scores without editing the model or probing its hidden representations. Our experiments address score sensitivity to a value still present in the input. They do not test whether an edited parametric fact remains decodable from hidden states.

### 2.3 Entity binding and order

Feng and Steinhardt (2024), Gur-Arieh et al. (2026), and Prakash et al. (2026) analyze how models retrieve bound entities or track characters' beliefs, including causal evidence for binding and lookback mechanisms. Those studies target internal retrieval mechanisms. We manipulate prompt construction and order, then measure changes in candidate-prefix scores; our results neither identify an internal mechanism nor imply that we have found a new one.

## 3. Task and estimands

### 3.1 Histories and conditions

Here, v1 and v2 name two experiment protocols, not model versions. V1 is the earlier comparison reported in Section 5 and Appendix A.1: it contrasted a live assignment, a superseded assignment, and two counterbalanced orders of unassigned values. It did not include an entity-mention control or independently cross the order of historical and current assignments. V2 is the follow-up reported in Sections 3–6: it uses six conditions, adds early and late unassigned values, an entity-mention construction, and a different-attribute control, and independently varies historical and current entity order. Both protocols use matched source-to-donor edits and query-specific score contrasts, but v2 uses a new renderer and a different control set. The v1 and v2 estimates therefore describe different prompt constructions. A v1 superseded prompt for the same Nora/Liam example was:

```text
The badge assigned to Nora was amber.
The badge assigned to Liam was coral.
Later, Nora’s badge was changed to jade.
Later, Liam’s badge was changed to pearl.
What is Nora’s current badge?
Respond with only the value, with no explanation.
```

Its counterbalanced-unassigned control instead stated the current assignments and then mentioned `amber` and `coral` as unassigned badge values, in both mention orders. V1's primary contrast averaged those two orders. V2 replaces this single control with separate early and late unassigned constructions, plus the entity-mention and other-attribute controls shown in the table below.

In the v2 entity-mention construction, a line such as `Nora mentioned amber in an unrelated note.` names Nora and amber together, but does not say that amber is Nora's badge. A separate current-state line assigns Nora's badge as jade. This condition tests whether a value mentioned alongside an entity can affect the query-specific score contrast without an explicit assignment to the queried attribute; it does not vary entity association alone.

Each history specifies two entities, labelled analytically as $x$ and $z$, and a target attribute. Four distinct values define the earlier and current assignments, and two additional distinct values serve as donors for the edits. The six values are selected from `amber`, `coral`, `jade`, `pearl`, `slate`, `teal`, `violet`, and `ivory`. The same history supplies the values for all six v2 conditions:

| Condition | Edited occurrence | Relation to queried attribute | Position and purpose |
|---|---|---|---|
| Superseded | Earlier value assigned to an entity's target attribute | Target assignment, later updated | Historical assignment block; focal condition |
| Early unassigned | Value mentioned without an entity assignment | No target assignment | Before current assignments; order control |
| Late unassigned | Value mentioned without an entity assignment | No target assignment | After current assignments; order control |
| Entity mention | Value mentioned in an entity-associated note | No target assignment | Entity association without target relation |
| Other attribute | Earlier team or project value for the same entity | Assignment to a different relation | Different-relation control |
| Live | Current assignment itself | Live target assignment | Positive control; edit changes the answer to the query about the edited entity |

In the five non-live conditions, each context contains historical/control mentions and two current assignments. Live contains both entities' current assignments, with one edited in place. The two queries are used in every condition. Under the query about the edited entity, the live answer changes to the donor; under the other entity's query, the correct answer stays fixed. The following example uses Nora's earlier value `amber`, Liam's earlier value `coral`, and current values `jade` and `pearl`. Let $C$ denote the two sentences `Currently, Nora’s badge is jade.` and `Currently, Liam’s badge is pearl.`

| Condition | Baseline context in the displayed order |
|---|---|
| Superseded | `Previously, Nora’s badge was amber.`; `Previously, Liam’s badge was coral.`; $C$ |
| Early unassigned | `The unassigned badge value was amber.`; `The unassigned badge value was coral.`; $C$ |
| Late unassigned | $C$; `The unassigned badge value was amber.`; `The unassigned badge value was coral.` |
| Entity mention | `Nora mentioned amber in an unrelated note.`; `Liam mentioned coral in an unrelated note.`; $C$ |
| Other attribute | `Previously, Nora’s team was amber.`; `Previously, Liam’s team was coral.`; $C$ |
| Live | `Currently, Nora’s badge is amber.`; `Currently, Liam’s badge is coral.` |

Each sentence occupies a separate line. Every context is followed by `What is Nora’s current badge?` or the corresponding Liam query, then `Respond with only the value, with no explanation.` The other-attribute condition alternates between `team` and `project` across histories. These are different relations; their content is not assumed to be semantically unrelated to the target attribute. Editing the analytic $x$ value replaces `amber` with `violet` in the selected condition. Nora's correct answer remains `jade` in the five non-live conditions and changes from `amber` to `violet` in live; Liam's assignment is present in every condition, and his correct answer is unchanged by an edit to Nora's value.

V2 uses different wording from v1. Its conditions also differ in predicates, syntax, and temporal expressions. The estimates compare these prompt constructions; they do not isolate assignment status, entity association, or temporal wording as separate causal factors.

The labels $x$ and $z$ remain attached to the same entities throughout a history. In the five non-live conditions, historical order specifies whether the mention associated with $x$ or $z$ appears first; current order independently specifies which entity's current assignment appears first. The orders are aligned when both blocks use $x$ then $z$ or both use $z$ then $x$, and reversed when they use different orders. For unassigned mentions, the earlier values retain analytic labels $x$ and $z$ for constructing the contrast, but neither value is assigned to an entity in the text. Alignment in those conditions compares the order of those analytically labelled values with current entity order. Semantic orientation determines which literal name corresponds to each analytic label. Live has only one physical assignment block: the renderer uses the historical-order field to order its two current-assignment lines, while the current-order field is redundant. Thus each live prompt is duplicated across current-order labels; these duplicates do not provide an independent current-order manipulation.

### 3.2 Bounded answer-prefix score

Let $p$ denote a tokenized prompt, including the model's chat template and the assistant prefix `Answer:`. For token continuation $w=(w_1,\ldots,w_m)$, define

\[
L(w\mid p)=\sum_{t=1}^{m}\log P(w_t\mid p,w_{<t}).
\]

For value $v$, let $\mathcal{W}(v)$ be its tokenizer-validated eligible surface continuations. We score

\[
S(v\mid p)=\log\sum_{w\in\mathcal{W}(v)}\exp L(w\mid p).
\]

For each value, we propose lowercase and title-case spellings, each with either no leading space or one leading ASCII space. Tokenizer validation retains only continuations that preserve the fixed prompt prefix, removes duplicate token sequences, and rejects collisions or cases in which one candidate sequence is a prefix of another. This makes the retained token-prefix events disjoint. We sum their sequence probabilities without length normalization or an end-of-turn token. Thus $S$ is the log probability mass of a fixed set of answer prefixes. Candidate-ranking accuracy compares these masses across the eight values; it does not measure the accuracy of unrestricted generated answers.

### 3.3 Matched edits and query relevance

An edit replaces a source value $s$ with a donor value $r$ in one selected occurrence. In the five non-live conditions, both current assignments remain fixed. In live, the edited occurrence is one entity's current assignment, so its answer changes to the donor; the other entity's answer stays fixed. We compute the same two-query $R$ contrast for live as for the other conditions. The renderer orders the two live assignment lines using the historical-order field; the current-order field is redundant, so each live prompt is duplicated across current-order labels rather than receiving a second physical order manipulation.

Let $p^0_{h,c,e,q,o}$ be the baseline prompt for history $h$, condition $c$, edited entity or analytic label $e\in\{x,z\}$, query $q$, and order cell $o$. The edited prompt $p^1_{h,c,e,q,o}$ differs only at the selected value span. Define

\[
E_{h,c,e,q,o}=
\big[S(r\mid p^1_{h,c,e,q,o})-S(s\mid p^1_{h,c,e,q,o})\big]
-\big[S(r\mid p^0_{h,c,e,q,o})-S(s\mid p^0_{h,c,e,q,o})\big].
\]

A positive $E$ means that the edit shifts the donor-versus-source score difference toward the donor. For example, replacing Nora's earlier `amber` with `violet` can change their relative scores even though Nora's current answer remains `jade`.

Write the order cell as $o=(o_H,o_C)$, where each component is 0 for $x$ then $z$ and 1 for $z$ then $x$. For an order group $g$, let $\bar E^g$ be the mean over its cells: all four cells for overall, $(0,0)$ and $(1,1)$ for aligned, and $(0,1)$ and $(1,0)$ for reversed. Let $q_x$ and $q_z$ ask about the entities labelled $x$ and $z$. Define query relevance as

\[
R^g_{h,c}=\frac12\left[
\bar E^g_{h,c,x,q_x}-\bar E^g_{h,c,x,q_z}
+\bar E^g_{h,c,z,q_z}-\bar E^g_{h,c,z,q_x}
\right].
\]

This averages the difference between the matched query and the other query over edits to both values. Positive $R$ means that the shift toward the donor is greater under the matched query. In the unassigned conditions, matching is an analytic convention: the text contains no entity assignment for the edited value. Live's two-query $R$ uses both current assignments. Its duplicated order labels are averaged and are not interpreted as separate physical order conditions. We omit the superscript $g$ when referring to the overall estimate.

For v1, the primary history-level contrast was

\[
\Delta^{(v1)}_h=R_{h,\mathrm{superseded}}-R_{h,\mathrm{unassigned,CB}},
\]

where the two unassigned mention orders were averaged within history. For v2, each control contrast is

\[
\Delta^{(v2)}_{h,c}=R_{h,\mathrm{superseded}}-R_{h,c},
\qquad c\in\{\mathrm{early},\mathrm{late},\mathrm{entity},\mathrm{other\ attribute}\}.
\]

The three-control prediction, recorded in a dated internal amendment before confirmation, is that superseded relevance exceeds early-unassigned, entity-mention, and other-attribute relevance. For each model, the rule requires the lower 95% history-bootstrap bound to exceed zero for all three overall contrasts. The amendment also specified the same criterion for the reversed-order contrasts. The prediction was not externally preregistered. The reversed-order prediction is distinct from the aligned-minus-reversed comparison, which is a secondary exploratory analysis. Even if a criterion were met, it would establish an ordering among these prompt constructions, not isolate assignment status, because other linguistic features differ.

The symmetric contrast cancels additive effects of edit label and query. It retains interactions between them, including interactions that depend on mention order. We therefore report aligned and reversed estimates alongside the overall mean. Live-minus-superseded differences are descriptive comparisons, not estimates of a percentage of retained information.

## 4. Experimental protocol

### 4.1 Models and eligibility

V1 evaluated Qwen3-8B, Gemma 3 4B, Phi-4-mini, and Mistral-7B-Instruct-v0.3. Qwen, Gemma, and Phi passed the competence gate and completed confirmation. Mistral scored 281/288 on its unassigned-control gate and did not proceed to confirmation. V2 evaluated Qwen, Gemma, and Phi. Qwen and Gemma passed all six condition-level competence gates and completed confirmation; Phi scored 561/576 (97.40%) in the entity-mention condition and did not proceed. Phi's completed v1 estimate is reported separately.

Each gate condition must have at least 99% strict rank-one accuracy, with ties counted as incorrect, and all required diagnostic cells must be present. Gate accuracy counts each distinct rendered prompt once within its condition. The superseded condition must also have a positive mean current-minus-stale score margin. The lowest v2 condition accuracy was 287/288 (99.65%) for Qwen in live and 575/576 (99.83%) for Gemma in entity mention. Both models scored 576/576 in superseded, with current-minus-stale margins of 21.571 and 23.910 nats, respectively. These gates select configurations for confirmation; they do not test the three-control prediction.

**Table 1. Model coverage by protocol.** Gate eligibility is distinct from confirmation performance. A dash denotes no confirmatory estimate.

| Protocol | Model | Gate | Confirmation histories |
|---|---|---|---:|
| v1 | Qwen3-8B | Pass | 96 |
| v1 | Gemma 3 4B | Pass | 96 |
| v1 | Phi-4-mini | Pass | 96 |
| v1 | Mistral-7B-Instruct-v0.3 | Fail: 281/288 in unassigned control | — |
| v2 | Qwen3-8B | Pass | 96 |
| v2 | Gemma 3 4B | Pass | 96 |
| v2 | Phi-4-mini | Fail: 561/576 in entity mention | — |

### 4.2 Histories, inference, and uncertainty

V1 used 24 development histories, 24 gate histories, and 96 confirmation histories per eligible model. Confirmation repeated 12 scheduled combinations of entity pair, attribute, and semantic orientation eight times with different value assignments. V2 used the same stage sizes but crossed six entity pairs, four target attributes (`badge`, `color`, `code`, `label`), and two semantic orientations. Each of the resulting 48 confirmation cells had two histories, with the alternate `team` and `project` relations balanced across the design. Development and gate used fixed 24-history subsets of this allocation. V2 used one set of sentence templates throughout.

Each v2 history contributes $6\times2\times2\times2\times2\times2=192$ scored rows: six conditions, two queries, two edited values, two historical orders, two current orders, and baseline versus edit. This gives 18,432 rows and 9,216 matched edit pairs per model. Each condition contributes 3,072 rows, half baseline and half edited. Some rows share the same rendered prompt, including live's duplicated current-order variants. Confirmation accuracy uses all rows, whereas gate accuracy deduplicates prompts. Estimates and uncertainty are computed over the 96 histories, and the primary analysis retains histories containing task errors.

We aggregate edit and query contrasts within each history before averaging across histories. The reported primary percentile 95% intervals use 2,000 unrestricted history-bootstrap samples with seed 73021: each draw samples 96 histories with replacement from the full confirmation set, without resampling within the 48 entity-pair-by-attribute-by-orientation cells. This follows the dated analysis amendment's history-level bootstrap and targets variation across histories in the balanced confirmation mixture. Appendix B reports a design-preserving sensitivity analysis that resamples two histories within each of the 48 fixed cells. Qwen and Gemma share the same confirmation histories, enabling comparisons under the same assignments and donors. These intervals describe variation across histories under the fixed task design; they do not cover changes in names, vocabulary, sentence templates, numerical precision, or model selection. The exclusion audit and implementation chronology are reported in Appendix C.

Tokenizer audits checked that each baseline/edit pair differed only at the selected value span, apart from the consequences of changing its token length. The reported model configurations used int8 weights; Qwen used float16 computation and Gemma bfloat16. Model checkpoint revisions, inference settings, and the implementation and dataset hashes are reported in Appendix C. Protocol decisions were internally logged rather than externally preregistered.

## 5. Initial observation under the v1 control

In v1, the superseded-minus-counterbalanced-unassigned contrast was positive in all three gate-eligible configurations. Qwen's mean was 3.50 nats (95% CI [3.20, 3.80]), positive in 94/96 histories; Gemma's was 1.76 [1.51, 2.03], positive in 88/96; and Phi's was 5.53 [5.27, 5.79], positive in 96/96. These are estimates from the bounded surface-class continuation-mass scoring protocol described in Section 3.2. The Qwen estimate is distinct from the 4.52-logit value reported in an earlier exact-token diagnostic; the two use different scoring procedures and are not numerically comparable. These v1 estimates describe the v1 constructions and motivated the v2 comparisons. Because v2 changes both wording and order design, differences between the v1 and v2 estimates cannot be attributed to a single control change.

V1 establishes that superseded relevance exceeds relevance for that counterbalanced unassigned construction. It leaves entity mention and relative position unmatched. V2 evaluates the interpretation of this contrast using the additional controls and independent order factors.

## 6. Results: entity association and mention order

### 6.1 Main findings and current-answer ranking

Entity mentions produce greater query relevance than superseded mentions in both models: 5.478 versus 1.974 nats for Qwen and 4.277 versus 1.688 for Gemma. Effects for both early and late unassigned values are positive under aligned order and negative under reversed order. Superseded effects stay positive in both groups. Table 2 reports the condition means and order strata.

In the superseded condition, both models ranked the current candidate strictly first in all 1,536 baseline rows and all 1,536 edited rows. Mean current-minus-stale score margins were 21.734 nats for Qwen and 23.854 for Gemma. The score-sensitivity effects therefore occur while the current candidate remains top-ranked among the eight values.

| Model | Condition | Overall $R$ [95% CI] | Aligned $R$ [95% CI] | Reversed $R$ [95% CI] |
|---|---|---:|---:|---:|
| Qwen3-8B | Live | 32.598 [31.258, 33.967] | N/A | N/A |
| Qwen3-8B | Superseded | 1.974 [1.840, 2.109] | 2.144 [1.995, 2.301] | 1.804 [1.660, 1.947] |
| Qwen3-8B | Early unassigned | −0.037 [−0.203, 0.121] | 3.252 [3.046, 3.455] | −3.327 [−3.518, −3.150] |
| Qwen3-8B | Late unassigned | 0.001 [−0.208, 0.211] | 3.187 [2.948, 3.442] | −3.184 [−3.433, −2.945] |
| Qwen3-8B | Entity mention | 5.478 [5.128, 5.825] | 4.493 [4.235, 4.770] | 6.463 [5.971, 6.940] |
| Qwen3-8B | Other attribute | 1.660 [1.488, 1.835] | 1.679 [1.527, 1.835] | 1.641 [1.422, 1.863] |
| Gemma 3 4B | Live | 39.355 [38.303, 40.415] | N/A | N/A |
| Gemma 3 4B | Superseded | 1.688 [1.568, 1.817] | 1.703 [1.574, 1.841] | 1.672 [1.529, 1.831] |
| Gemma 3 4B | Early unassigned | 0.113 [−0.023, 0.245] | 2.598 [2.395, 2.798] | −2.372 [−2.567, −2.178] |
| Gemma 3 4B | Late unassigned | 0.125 [−0.055, 0.305] | 3.445 [3.202, 3.696] | −3.194 [−3.447, −2.945] |
| Gemma 3 4B | Entity mention | 4.277 [4.029, 4.545] | 5.181 [4.884, 5.496] | 3.372 [3.131, 3.633] |
| Gemma 3 4B | Other attribute | 1.683 [1.541, 1.832] | 1.763 [1.598, 1.932] | 1.603 [1.453, 1.771] |

**Table 2.** V2 condition-level query relevance $R$, in nats. Each interval is a percentile 95% history-bootstrap interval over 96 histories (2,000 draws; seed 73021). Live order labels duplicate one assignment block and do not estimate an order effect.

![Overall query relevance by condition, with 95% history-bootstrap intervals. Live is shown on a separate scale.](overall_condition_means.svg)

**Figure 1.** Overall $R$ by condition. Points show the mean of the 96 history-level estimates; horizontal bars show percentile 95% history-bootstrap intervals (2,000 draws; seed 73021). The live condition is shown on a separate scale. Positive $R$ denotes a larger donor-versus-source score shift under the matched query than under the other-entity query.

### 6.2 Frozen superseded-minus-control comparisons

The three frozen overall contrasts, in nats, are:

| Superseded minus control (nats) | Qwen3-8B [95% CI] | Gemma 3 4B [95% CI] |
|---|---:|---:|
| Early unassigned | 2.011 [1.855, 2.170] | 1.575 [1.440, 1.719] |
| Entity mention | −3.504 [−3.806, −3.198] | −2.589 [−2.829, −2.361] |
| Other attribute | 0.314 [0.147, 0.482] | 0.004 [−0.104, 0.118] |

**Table 3.** History-level difference in query relevance, superseded minus each control. Intervals use 2,000 history-bootstrap draws (seed 73021).

Superseded relevance exceeds early-unassigned relevance in both models, but is lower than entity-mention relevance. Its contrast with the other-attribute condition is positive for Qwen (0.314 nats, 95% CI [0.147, 0.482]) and close to zero for Gemma (0.004, [−0.104, 0.118]). The three-control criterion specified in the internal amendment is not met in either model because the entity-mention contrast is negative in both. Gemma's near-zero other-attribute contrast is inconclusive; no equivalence margin was specified.

The entity-mention condition places the edited value alongside the entity without assigning it to the queried attribute. Its larger $R$ means this construction produces more query relevance than a superseded assignment. The comparison cannot isolate the reason because the conditions also differ in predicate and syntax.

### 6.3 Order dependence and cancellation

For both early and late unassigned mentions, $R$ is positive under aligned order and negative under reversed order in both models. Early-unassigned $R$, for example, is 3.252 and −3.327 nats for Qwen, and 2.598 and −2.372 for Gemma. The pattern is consistent with association by relative mention order: reversing value order relative to entity order also reverses the query contrast. This is an interpretation of prompt-level behavior; it does not identify an internal binding mechanism. Superseded-minus-early-unassigned is negative under aligned order and positive under reversed order.

| Contrast | Order | Qwen3-8B [95% CI] | Gemma 3 4B [95% CI] |
|---|---|---:|---:|
| Superseded − early unassigned | Aligned | −1.108 [−1.281, −0.925] | −0.895 [−1.084, −0.719] |
| Superseded − early unassigned | Reversed | 5.130 [4.920, 5.348] | 4.044 [3.810, 4.292] |
| Aligned minus reversed, paired secondary analysis | — | −6.238 [−6.481, −5.983] | −4.939 [−5.275, −4.620] |

**Table 4.** Superseded-minus-early-unassigned contrasts in nats by order, plus the paired aligned-minus-reversed difference (aligned minus reversed). Intervals are 95% history-bootstrap intervals over 96 histories; the interaction is exploratory.

![Order-stratified superseded-minus-control contrasts for early and late unassigned conditions.](order_stratified_contrasts.svg)

**Figure 2.** Superseded-minus-control differences in $R$ by order for early and late unassigned conditions. Points show estimates and bars show 95% history-bootstrap intervals. Zero marks equal query relevance in the superseded and control conditions.

Superseded relevance is positive in both order groups (Table 2), whereas unassigned values reverse sign. Late mentions follow the same order pattern as early ones. Their overall means are near zero because the positive aligned and negative reversed estimates cancel; reporting only the mean would obscure both effects.

The paired aligned-minus-reversed differences were computed in the secondary reanalysis and are interpreted as exploratory. Separately, the amendment specified a reversed-order version of the three-control criterion. That criterion also fails in both models: the entity-mention contrast is negative and the other-attribute interval spans zero. The primary decision remains based on the overall contrasts.

### 6.4 Score decomposition and heterogeneity

For Qwen's superseded condition, mean edit effects under the matched and other-entity queries are 4.095 and 2.121 nats. Their difference gives $R=1.974$, while their average gives the mean edit effect $\overline E=3.108$ reported in Appendix B. Gemma's corresponding effects are 3.315 and 1.628 nats. For early unassigned mentions, the two query effects are similar: 8.202 and 8.239 for Qwen, and 6.347 and 6.234 for Gemma. Near-zero overall unassigned $R$ therefore reflects cancellation between query effects rather than an absence of response to the edit. All displayed values are rounded; the estimates are computed before rounding.

The donor shift reflects both a lower source score and a higher donor score. In superseded trials, Qwen's mean source log mass changes from −22.271 to −23.881 and donor log mass from −23.618 to −22.120. Gemma's corresponding changes are −24.311 to −25.552 and −25.498 to −24.268. The current candidate remains top-ranked, while the source and donor prefix events have low absolute mass. These score changes do not tell us whether unrestricted generation would change its answer.

The entity-mention construction has greater relevance than the superseded construction in all four target attributes for both models. The other-attribute contrast varies by attribute. For Qwen it is positive for badge and color, negative for code (−0.570 [−0.866, −0.280]), and inconclusive for label (0.200 [−0.042, 0.459]). Gemma's attribute-specific other-attribute intervals all include zero. Each attribute contributes 24 histories. These strata describe heterogeneity and are not additional confirmatory tests of the three-control prediction.

## 7. Discussion

The results identify two prompt-level regularities. Entity-associated mentions produce greater query relevance than superseded assignments in both tested models. Unassigned-value effects reverse sign across aligned and reversed mention orders, while superseded effects remain positive in both groups. These findings make the entity association and order of mentions central to interpreting historical-value sensitivity.

The order pattern fits a positional account: when an unassigned value and an entity occupy corresponding positions, the model's score response follows that order. Reversing it reverses the query contrast. The experiment does not intervene on internal representations and cannot identify a binding-ID mechanism. The lower superseded relevance than entity-mention relevance may also reflect the temporal marker or predicate structure. The other-attribute condition uses `Previously` too, but the superseded-minus-other contrast is positive for Qwen and near zero for Gemma. The pending marker-by-construction comparison is needed to separate these possibilities.

The initial three-control prediction fails in v2. Superseded relevance exceeds early-unassigned relevance in both models and other-attribute relevance for Qwen, but is below entity-mention relevance in both. Positive $R$ is not unique to the superseded prompt construction. The experiment cannot determine whether obsolete relation status contributes to the superseded effect because the prompt constructions do not vary that status alone.

The outcome is a score for a fixed set of answer prefixes, with no termination event. Historical source and donor events have low absolute mass, and the current answer remains top-ranked in every superseded trial. The study establishes score sensitivity under high competence. The derived-code extension is reported in Appendix A.2. Whether the score shifts change unrestricted generated answers or cause a reliable practical accuracy effect remains unresolved.

The experiments use two-entity English prompts and a fixed eight-value vocabulary. V1 and v2 use different templates; v2 varies six entity pairs and four target attributes within one renderer. Color-like values may fit some attributes more naturally than others, and the `team`/`project` conditions are different-relation controls. The v2 confirmation results cover Qwen and Gemma; Phi's gate result and Mistral's v1 gate failure are reported separately. Generalization to longer histories, distractors, multiple turns, other languages, and other model families has not been tested.

## 8. Conclusion

In Qwen3-8B and Gemma 3 4B, the entity-mention construction yields greater $R$ than the superseded construction. Unassigned-value effects reverse sign when historical and current mention order is reversed, while superseded effects stay positive in both groups. The three-control prediction specified in an internal amendment before confirmation fails because superseded relevance is lower than entity-mention relevance in both models. The estimates vary with prompt construction and mention order. The score-based design does not establish whether the effect changes unrestricted answers.

## References

Feng, J., and Steinhardt, J. (2024). *How Do Language Models Bind Entities in Context?* International Conference on Learning Representations (ICLR). https://proceedings.iclr.cc/paper_files/paper/2024/file/9d1b7fc578c0d2d6431fc26d736ecaf3-Paper-Conference.pdf

Guo, J., Fang, Y., Gu, S., Spanos, C., Demmel, J., and Lavaei, J. (2026). *When Context Changes: Understanding Update Failures in LLMs.* arXiv:2609.38866. https://arxiv.org/abs/2609.38866

Gur-Arieh, Y., Geva, M., and Geiger, A. (2026). *Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context.* ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/2eeff35664016c7f0f8aa704f0d9a83e-Abstract-Conference.html

Kim, N., and Schuster, S. (2023). *Entity Tracking in Language Models.* Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3835–3855. https://aclanthology.org/2023.acl-long.213/

Prakash, N., Shapira, N., Sen Sharma, A., Riedl, C., Belinkov, Y., Rott Shaham, T., Bau, D., and Geiger, A. (2026). *Language Models Use Lookbacks to Track Beliefs.* ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/0d1941f73833eb107939f62d323d6cc0-Abstract-Conference.html

Qian, Y., Yang, Z., Ding, W., Li, H., and Xie, Y. (2026). *Do LLMs Forget What They Should? Evaluating In-Context Forgetting in Large Language Models.* ICLR 2026. https://proceedings.iclr.cc/paper_files/paper/2026/hash/b13d00a62d438856cfe6fbd13b6b2cb8-Abstract-Conference.html

Srivastava, P., Ai, S., and Chatterjee, R. (2026). *Suppressed, Not Erased: A Representational Trace of Edited Facts Survives Even Weight-Free Knowledge Editing.* arXiv:2609.18985. https://arxiv.org/abs/2609.18985

Tang, Z., Zhao, Q., Franco, G., Wijaya, D. T., Mueller, A., Schuster, S., and Kim, N. (2026). *Do Language Models Track Entities Across State Changes?* Proceedings of the 43rd International Conference on Machine Learning, PMLR 306, pp. 119466–119503. https://proceedings.mlr.press/v306/tang26ah.html

Wang, C., and Sun, J. V. (2025). *Unable to Forget: Proactive Interference Reveals Working Memory Limits in LLMs Beyond Context Length.* arXiv:2506.08184. https://arxiv.org/abs/2506.08184

Wilie, B., Cahyawijaya, S., Ishii, E., He, J., and Fung, P. (2024). *Belief Revision: The Adaptability of Large Language Models Reasoning.* Proceedings of EMNLP 2024, pp. 10480–10496. https://aclanthology.org/2024.emnlp-main.586/

Xie, J., Cao, P., Chen, Y., Liu, K., and Zhao, J. (2025). *Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing.* Proceedings of ACL, Volume 1: Long Papers, pp. 17756–17780. https://aclanthology.org/2025.acl-long.868/

Yang, X., Liu, Z., Li, R., and Lei, Y. (2026). *Large Language Models in Resolving Contextual Knowledge Conflicts.* arXiv:2609.03148. https://arxiv.org/abs/2609.03148

## Appendix A. Initial v1 and derived-code observations

### A.1 V1 conditions and estimates

V1 used the `nora_v1` sentence templates and compared live assignments, superseded assignments, and counterbalanced unassigned mentions. Ordinary unassigned trials were diagnostic; the primary contrast used the two-order counterbalanced condition. Each history contributed 40 scored rows, giving 3,840 rows and 1,920 matched edit pairs per eligible model's 96-history confirmation set. The analysis retained all histories and bootstrapped at the history level. V1 did not match entity mention between focal and control conditions or independently cross historical and current order. Its primary estimates are reported in Section 5.

### A.2 Derived-code extension

The derived-code task mapped values to shuffled opaque codes and asked which code corresponded to an entity's current badge. The codebook stayed fixed across baseline and edit; editing an earlier badge value changed its mapped code, not the codebook or current badge assignment. We scored all 16 full code sequences by their unnormalized log probability. Qwen alone passed the separate 47/48 gate (48/48); Gemma scored 45/48 and Phi 33/48 and did not proceed. Qwen's stale-derived query relevance was 5.544 nats (95% CI [5.159, 5.941]) and live-derived relevance was 12.836 [12.418, 13.257], both positive in all 96 histories. In stale trials, the current code ranked first in 360/384 baseline and 364/384 edited prompts; 9 pairs changed from incorrect to correct and 5 from correct to incorrect. Changes in current-code log probability (0.108 [−0.050, 0.264]) and candidate margin (0.103 [−0.118, 0.325]) were inconclusive. This exploratory Qwen-only extension shows score sensitivity in a derived-code task, but it lacks v2's entity-mention and order controls. The small net increase in rank-one current-code accuracy does not establish a practical accuracy effect.

## Appendix B. V2 order and score diagnostics

### B.1 Order cells

Section 3.3 defines the overall, aligned, and reversed estimands. The table below gives the remaining superseded-minus-control contrasts in nats. Paired aligned-minus-reversed differences were computed in the secondary reanalysis and are exploratory. Live has one assignment block and is excluded from physical order comparisons.

| Control | Order | Qwen3-8B [95% CI] | Gemma 3 4B [95% CI] |
|---|---|---:|---:|
| Late unassigned | Overall | 1.973 [1.762, 2.183] | 1.562 [1.390, 1.742] |
| Late unassigned | Aligned | −1.043 [−1.281, −0.810] | −1.741 [−1.968, −1.505] |
| Late unassigned | Reversed | 4.988 [4.724, 5.266] | 4.866 [4.592, 5.155] |
| Entity mention | Aligned | −2.350 [−2.572, −2.125] | −3.478 [−3.769, −3.197] |
| Entity mention | Reversed | −4.659 [−5.119, −4.207] | −1.700 [−1.946, −1.465] |
| Other attribute | Aligned | 0.465 [0.295, 0.649] | −0.060 [−0.215, 0.099] |
| Other attribute | Reversed | 0.162 [−0.049, 0.367] | 0.069 [−0.080, 0.213] |

**Table B1.** Remaining v2 superseded-minus-control estimates by order, in nats. Intervals are 95% history-bootstrap intervals over 96 histories.

Paired aligned-minus-reversed differences in the superseded-minus-control contrast are exploratory secondary estimates, with histories as the bootstrap unit:

| Control | Qwen3-8B [95% CI], nats | Gemma 3 4B [95% CI], nats |
|---|---:|---:|
| Late unassigned | −6.031 [−6.307, −5.754] | −6.607 [−6.951, −6.243] |
| Entity mention | 2.309 [1.933, 2.707] | −1.778 [−2.049, −1.519] |
| Other attribute | 0.302 [0.103, 0.509] | −0.128 [−0.323, 0.081] |

**Table B2.** Paired aligned-minus-reversed differences in the superseded-minus-control estimate, in nats. Intervals are 95% history-bootstrap intervals over 96 histories.

### B.2 Competence and selected-history analysis

Strict rank-one accuracy across all 3,072 confirmation rows per condition, including baseline and edited rows and duplicate prompt representations, is:

| Model | Condition | Correct/total (%) |
|---|---|---:|
| Qwen3-8B | Live | 3,064/3,072 (99.7396%) |
| Qwen3-8B | Superseded | 3,072/3,072 (100%) |
| Qwen3-8B | Early unassigned | 3,072/3,072 (100%) |
| Qwen3-8B | Late unassigned | 3,069/3,072 (99.9023%) |
| Qwen3-8B | Entity mention | 3,072/3,072 (100%) |
| Qwen3-8B | Other attribute | 3,072/3,072 (100%) |
| Gemma 3 4B | Live | 3,072/3,072 (100%) |
| Gemma 3 4B | Superseded | 3,072/3,072 (100%) |
| Gemma 3 4B | Early unassigned | 3,071/3,072 (99.9674%) |
| Gemma 3 4B | Late unassigned | 3,072/3,072 (100%) |
| Gemma 3 4B | Entity mention | 3,072/3,072 (100%) |
| Gemma 3 4B | Other attribute | 3,072/3,072 (100%) |

**Table B3.** Strict rank-one candidate accuracy on all 3,072 confirmation rows per condition. Counts include baseline and edited rows and repeated representations of identical prompts.

A secondary analysis retained only histories for which every confirmation row had a correct strict rank-one candidate. This retained 91/96 Qwen histories and 95/96 Gemma histories. The superseded-minus-control estimates were:

| Control | Qwen3-8B [95% CI], nats | Gemma 3 4B [95% CI], nats |
|---|---:|---:|
| Early unassigned | 2.015 [1.861, 2.176] | 1.576 [1.433, 1.714] |
| Entity mention | −3.445 [−3.750, −3.141] | −2.591 [−2.827, −2.354] |
| Other attribute | 0.371 [0.201, 0.532] | 0.004 [−0.106, 0.119] |

**Table B4.** Superseded-minus-control contrasts among histories with correct strict rank-one candidates on every confirmation row. Intervals are 95% history-bootstrap intervals over the retained histories.

This analysis conditions on model performance and is descriptive; the primary analysis retains all 96 histories.

Phi-4-mini scored 561/576 unique prompts (97.40%) in the v2 entity-mention gate, below the frozen 99% threshold. It did not proceed to confirmation, so no confirmatory v2 $R$ estimate is available. The gate result is reported to make model coverage explicit; it is not a confirmation estimate.

### B.3 Paired score changes

The table reports mean baseline-to-edit changes in source and donor log prefix mass, averaged across matched and other-entity queries. Mean $E$ is the donor change minus the source change before rounding. It differs from $R$, which subtracts the other-query effect from the matched-query effect.

| Model | Condition | Source log-mass change | Donor log-mass change | Mean $E$ (nats) |
|---|---|---:|---:|---:|
| Qwen3-8B | Superseded | −1.610 | 1.497 | 3.108 |
| Qwen3-8B | Early unassigned | −4.208 | 4.012 | 8.221 |
| Qwen3-8B | Late unassigned | −5.901 | 5.682 | 11.583 |
| Qwen3-8B | Entity mention | −3.832 | 3.584 | 7.417 |
| Qwen3-8B | Other attribute | −1.348 | 1.331 | 2.679 |
| Gemma 3 4B | Superseded | −1.241 | 1.230 | 2.471 |
| Gemma 3 4B | Early unassigned | −3.315 | 2.975 | 6.290 |
| Gemma 3 4B | Late unassigned | −4.511 | 3.999 | 8.510 |
| Gemma 3 4B | Entity mention | −3.209 | 2.957 | 6.166 |
| Gemma 3 4B | Other attribute | −1.444 | 1.113 | 2.557 |

**Table B5.** Mean changes in source and donor log prefix mass and their mean edit effect $E$, averaged across matched and other-entity queries. Values are in nats.

These descriptive means are reported without confidence intervals; inference uses the history-bootstrap intervals for $R$ and the control contrasts. The scores are log masses for the defined prefix events. Exponentiating a mean log mass gives a geometric mean of event masses, not their arithmetic mean.

In superseded trials, mean current-answer log masses round to 0.000 before and after editing for both models. Across all six conditions, the rounded means range from −0.012 to 0.000 for Qwen and from −0.004 to 0.000 for Gemma; Qwen's −0.012 occurs in live. The current-answer rank and current-minus-stale margins are reported in Section 6.1.

### B.4 Attribute-stratified contrasts

Superseded-minus-entity-mention is negative for every target attribute in both models:

| Model | Attribute | Superseded − entity mention [95% CI] |
|---|---|---:|
| Qwen3-8B | Badge | −4.310 [−4.819, −3.787] |
| Qwen3-8B | Color | −1.794 [−2.129, −1.469] |
| Qwen3-8B | Code | −4.570 [−5.098, −4.058] |
| Qwen3-8B | Label | −3.342 [−3.668, −3.017] |
| Gemma 3 4B | Badge | −1.794 [−2.047, −1.556] |
| Gemma 3 4B | Color | −2.532 [−2.852, −2.203] |
| Gemma 3 4B | Code | −3.937 [−4.418, −3.452] |
| Gemma 3 4B | Label | −2.094 [−2.358, −1.809] |

**Table B6.** Attribute-specific superseded-minus-entity-mention contrasts in $R$, in nats. Intervals are 95% history-bootstrap intervals over the 24 histories for each attribute.

The attribute-specific superseded-minus-other-attribute estimates are:

| Attribute | Qwen3-8B [95% CI], nats | Gemma 3 4B [95% CI], nats |
|---|---:|---:|
| Badge | 0.748 [0.512, 0.969] | 0.011 [−0.206, 0.232] |
| Color | 0.876 [0.683, 1.068] | −0.028 [−0.236, 0.208] |
| Code | −0.570 [−0.866, −0.280] | −0.045 [−0.292, 0.217] |
| Label | 0.200 [−0.042, 0.459] | 0.078 [−0.110, 0.264] |

**Table B7.** Attribute-specific superseded-minus-other-attribute contrasts in $R$, in nats. Intervals are 95% history-bootstrap intervals over the 24 histories for each attribute.

Each attribute contributes 24 histories per model; these are descriptive strata rather than separate confirmatory claims.

### B.5 Cell-stratified bootstrap sensitivity

The primary intervals resample histories without restricting draws to the 48 entity-pair-by-attribute-by-orientation cells. We also computed percentile intervals by sampling two histories with replacement within each of those 48 cells on every bootstrap draw (2,000 draws, seed 73021). This holds the cell counts fixed; because each cell has one `team` and one `project` history, the resampled cell can include two histories from the same alternate relation. Means are unchanged. The cell-stratified intervals preserve the decision for all three contrasts in both models: the early-unassigned contrast is positive, the entity-mention contrast is negative, Qwen's other-attribute contrast is positive, and Gemma's other-attribute contrast includes zero.

| Contrast (`R_superseded − R_control`) | Qwen3-8B mean [cell-stratified 95% CI], nats | Gemma 3 4B mean [cell-stratified 95% CI], nats |
|---|---:|---:|
| Early unassigned | 2.011 [1.920, 2.100] | 1.575 [1.477, 1.667] |
| Entity mention | −3.504 [−3.641, −3.363] | −2.589 [−2.707, −2.475] |
| Other attribute | 0.314 [0.225, 0.403] | 0.004 [−0.079, 0.090] |

**Table B8.** Sensitivity intervals resampling histories within each fixed entity-pair-by-attribute-by-orientation cell. Primary inference continues to use the unrestricted history bootstrap specified in the dated amendment.


## Appendix C. Protocol chronology and provenance

Earlier exact-token diagnostics identified casing and tokenization problems for Phi and candidate-ranking errors for Mistral. The later protocol scored tokenizer-validated surface classes and used new competence gates. Results from these stages are reported under their respective scoring protocols. Protocol decisions were internally logged, without external preregistration.

The first Gemma v2 confirmation attempt stopped during geometry validation before model loading, so it produced no confirmation scores. Its 96 physical histories were excluded from the subsequent confirmation dataset. The repaired validator retained a separate mapping from semantic fields to text spans for each row, including rows whose rendered prompts were duplicates. A further runtime check distinguished the environment used to validate saved artifacts from the original inference environment. Saved development and gate scores were then validated under the corrected implementation and gave the same gate results. Qwen and Gemma subsequently passed the required checks and completed confirmation.

The completion audit recorded 18,432 valid score rows and 96 histories per model, no row errors, matching confirmation history assignments across models, and consistent dataset, score, and report bindings. Neither confirmation set overlapped with the stopped Gemma bundle. Qwen used model checkpoint revision `b968826d9c46dd6066d109eabc6255188de91218`; Gemma used `093f9f388b31de276ce2de164bdc2081324b9767`. Both used int8 weights and CUDA inference with Python 3.12.13. Qwen's computation dtype was float16 and Gemma's was bfloat16.

The corrected implementation freeze has SHA-256 `f2981fa0f05a5e29e9478f8a94987e7578b2ce3667d1c941e8ce31cf8cdfef9a`. The shared confirmation dataset has SHA-256 `27c1e9b80e62a0cb0266d393fd6120dff2bcf637d8e9468491748e740287c08d`.

The interpretation amendment was recorded on 3 October 2026 before confirmation results were examined, with SHA-256 `0371b4e1c9a363bbcb8dfdc0a0ccc6a3c39c08f49843b59eea9bcbde27ccd2b9`. It specified that all three overall superseded-minus-control lower 95% bounds must exceed zero to support the three-control prediction, and applied the same rule to a separate reversed-order prediction. The rule and its limitations are described in Section 3.3.
