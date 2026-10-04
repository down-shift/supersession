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


