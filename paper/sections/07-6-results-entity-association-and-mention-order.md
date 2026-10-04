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

![Overall query relevance by condition, with 95% history-bootstrap intervals. Live is shown on a separate scale.](../overall_condition_means.svg)

**Figure 1.** Overall $R$ by condition. Points show the mean of the 96 history-level estimates; horizontal bars show percentile 95% history-bootstrap intervals (2,000 draws; seed 73021). The live condition is shown on a separate scale. Positive $R$ denotes a larger donor-versus-source score shift under the matched query than under the other-entity query.

### 6.2 Frozen superseded-minus-control comparisons

The three frozen overall contrasts, in nats, are:

| Superseded minus control (nats) | Qwen3-8B [95% CI] | Gemma 3 4B [95% CI] |
|---|---:|---:|
| Early unassigned | 2.011 [1.855, 2.170] | 1.575 [1.440, 1.719] |
| Entity mention | −3.504 [−3.806, −3.198] | −2.589 [−2.829, −2.361] |
| Other attribute | 0.314 [0.147, 0.482] | 0.004 [−0.104, 0.118] |

**Table 3.** History-level difference in query relevance, superseded minus each control. Intervals use 2,000 history-bootstrap draws (seed 73021).

Superseded relevance exceeds early-unassigned relevance in both models, but is lower than entity-mention relevance. Its contrast with the other-attribute condition is positive for Qwen (0.314 nats, 95% CI [0.147, 0.482]) and close to zero for Gemma (0.004, [−0.104, 0.118]). The prespecified three-control criterion is not met in either model because the entity-mention contrast is negative in both. Gemma's near-zero other-attribute contrast is inconclusive; no equivalence margin was specified.

The entity-mention condition places the edited value alongside the entity without assigning it to the queried attribute. Its larger $R$ means this construction produces more query relevance than a superseded assignment. The comparison cannot isolate the reason because the conditions also differ in predicate and syntax.

### 6.3 Order dependence and cancellation

For both early and late unassigned mentions, $R$ is positive under aligned order and negative under reversed order in both models. Early-unassigned $R$, for example, is 3.252 and −3.327 nats for Qwen, and 2.598 and −2.372 for Gemma. The pattern is consistent with association by relative mention order: reversing value order relative to entity order also reverses the query contrast. This is an interpretation of prompt-level behavior; it does not identify an internal binding mechanism. Superseded-minus-early-unassigned is negative under aligned order and positive under reversed order.

| Contrast | Order | Qwen3-8B [95% CI] | Gemma 3 4B [95% CI] |
|---|---|---:|---:|
| Superseded − early unassigned | Aligned | −1.108 [−1.281, −0.925] | −0.895 [−1.084, −0.719] |
| Superseded − early unassigned | Reversed | 5.130 [4.920, 5.348] | 4.044 [3.810, 4.292] |
| Aligned minus reversed, paired secondary analysis | — | −6.238 [−6.481, −5.983] | −4.939 [−5.275, −4.620] |

**Table 4.** Superseded-minus-early-unassigned contrasts in nats by order, plus the paired aligned-minus-reversed difference (aligned minus reversed). Intervals are 95% history-bootstrap intervals over 96 histories; the interaction is exploratory.

![Order-stratified superseded-minus-control contrasts for early and late unassigned conditions.](../order_stratified_contrasts.svg)

**Figure 2.** Superseded-minus-control differences in $R$ by order for early and late unassigned conditions. Points show estimates and bars show 95% history-bootstrap intervals. Zero marks equal query relevance in the superseded and control conditions.

Superseded relevance is positive in both order groups (Table 2), whereas unassigned values reverse sign. Late mentions follow the same order pattern as early ones. Their overall means are near zero because the positive aligned and negative reversed estimates cancel; reporting only the mean would obscure both effects.

The paired aligned-minus-reversed differences were computed in the secondary reanalysis and are interpreted as exploratory. Separately, the amendment specified a reversed-order version of the three-control criterion. That criterion also fails in both models: the entity-mention contrast is negative and the other-attribute interval spans zero. The primary decision remains based on the overall contrasts.

### 6.4 Score decomposition and heterogeneity

For Qwen's superseded condition, mean edit effects under the matched and other-entity queries are 4.095 and 2.121 nats. Their difference gives $R=1.974$, while their average gives the mean edit effect $\overline E=3.108$ reported in Appendix B. Gemma's corresponding effects are 3.315 and 1.628 nats. For early unassigned mentions, the two query effects are similar: 8.202 and 8.239 for Qwen, and 6.347 and 6.234 for Gemma. Near-zero overall unassigned $R$ therefore reflects cancellation between query effects rather than an absence of response to the edit. All displayed values are rounded; the estimates are computed before rounding.

The donor shift reflects both a lower source score and a higher donor score. In superseded trials, Qwen's mean source log mass changes from −22.271 to −23.881 and donor log mass from −23.618 to −22.120. Gemma's corresponding changes are −24.311 to −25.552 and −25.498 to −24.268. The current candidate remains top-ranked, while the source and donor prefix events have low absolute mass. These score changes do not tell us whether unrestricted generation would change its answer.

The entity-mention construction has greater relevance than the superseded construction in all four target attributes for both models. The other-attribute contrast varies by attribute. For Qwen it is positive for badge and color, negative for code (−0.570 [−0.866, −0.280]), and inconclusive for label (0.200 [−0.042, 0.459]). Gemma's attribute-specific other-attribute intervals all include zero. Each attribute contributes 24 histories. These strata describe heterogeneity and are not additional confirmatory tests of the three-control prediction.

