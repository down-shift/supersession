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
