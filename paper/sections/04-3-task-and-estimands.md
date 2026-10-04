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
