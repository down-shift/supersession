## Appendix D. Marker × construction confirmation

### D.1 Scope and verification

This confirmation is separate from relational v2 and the two-history marker pilot. The 24-history dataset explicitly excludes both pilot histories; there is no full-history or target-history signature overlap. Each model has 3,072/3,072 scored members over 2,272 unique prompts, covering every factorial cell. Both run records and sealed score provenance report completion. Dataset, score, report, and provenance hashes agree with the sealed analyses, and recomputing the analyses from the saved scores reproduces the saved estimates. The generation report was written before inference and therefore records evaluation as pending; completed score sidecars and analyses establish the final status. The original report is preserved.

The models share the same dataset (generation seed 23261011; scoring seed 20261006), Python 3.13.5, and the repository's locked environment. Qwen/Qwen3-8B used model and tokenizer revision `b968826d9c46dd6066d109eabc6255188de91218`; google/gemma-3-4b-it used `093f9f388b31de276ce2de164bdc2081324b9767`. The runtime was an NVIDIA GeForce RTX 5080 with CUDA 13.0. Qwen used int8 weights with float16 computation; Gemma used int8 weights with bfloat16 computation. Each scored 2,272 new unique prompts with no resumed prompts. Scoring took 1,551.08 seconds for Qwen and 657.75 seconds for Gemma; these candidate-only runtimes do not estimate harder-task runs that generate answers.

### D.2 Marker effects

Marker effect is $R$ with `Previously` minus $R$ without it. The interaction is the superseded marker effect minus the entity-mention marker effect. Aligned order means historical and current entity orders agree; reversed order means they differ. Intervals are pointwise 95% percentile history-bootstrap intervals (2,000 draws, seed 73021; 24 histories per estimate).

| Model | Order | Superseded marker effect | Entity-mention marker effect | Interaction |
|---|---|---:|---:|---:|
| Qwen | All | −0.750 [−0.917, −0.590] | −2.087 [−2.410, −1.772] | 1.337 [0.976, 1.712] |
| Qwen | Aligned | −0.372 [−0.552, −0.187] | −1.298 [−1.580, −0.979] | 0.926 [0.581, 1.274] |
| Qwen | Reversed | −1.127 [−1.324, −0.939] | −2.876 [−3.354, −2.429] | 1.749 [1.236, 2.274] |
| Gemma | All | −1.085 [−1.308, −0.864] | −0.960 [−1.183, −0.727] | −0.125 [−0.390, 0.136] |
| Gemma | Aligned | −1.100 [−1.359, −0.851] | −1.155 [−1.384, −0.915] | 0.055 [−0.233, 0.328] |
| Gemma | Reversed | −1.069 [−1.372, −0.785] | −0.765 [−1.123, −0.387] | −0.304 [−0.753, 0.116] |

**Table D1.** Marker effects and marker-by-construction interactions in nats. All within-construction marker-effect intervals are below zero. Qwen's interaction interval is above zero overall and in both order strata. Gemma's interaction intervals include zero; this is inconclusive about a construction difference and does not establish equivalence. $R$ remains positive with the marker in every model, construction, and order cell.

The Qwen interaction agrees in direction with the exploratory two-history pilot. Gemma's overall interaction remains inconclusive, while its pilot aligned-stratum interaction does not persist as a clear difference in confirmation. Pilot histories are not pooled with confirmation.

### D.3 R decomposition

For each baseline-to-edit pair, $E$ is the change in replacement log mass minus the change in source log mass. $R$ subtracts $E$ for the other query entity from $E$ for the matching query entity, then symmetrizes over edited entities. The same transformation is applied to each component, so $R=R_{replacement}-R_{source}$. Table D2 reports means and 95% history-bootstrap intervals (2,000 draws, seed 73021; 24 histories).

| Model | Construction | `Previously` | $R$ | Replacement component | Source component |
|---|---|---|---:|---:|---:|
| Qwen | Superseded | Absent | 3.738 [3.536, 3.934] | 1.590 [1.118, 2.005] | −2.149 [−2.623, −1.705] |
| Qwen | Superseded | Present | 2.989 [2.791, 3.186] | 1.283 [0.832, 1.746] | −1.705 [−2.151, −1.242] |
| Qwen | Entity mention | Absent | 7.304 [6.629, 7.966] | 3.875 [3.477, 4.277] | −3.429 [−4.135, −2.764] |
| Qwen | Entity mention | Present | 5.217 [4.749, 5.705] | 2.540 [2.171, 2.917] | −2.677 [−3.193, −2.165] |
| Gemma | Superseded | Absent | 2.631 [2.409, 2.863] | 1.413 [1.170, 1.652] | −1.218 [−1.501, −0.934] |
| Gemma | Superseded | Present | 1.546 [1.379, 1.714] | 0.685 [0.445, 0.932] | −0.861 [−1.115, −0.610] |
| Gemma | Entity mention | Absent | 3.313 [2.975, 3.669] | 1.732 [1.427, 2.048] | −1.582 [−1.915, −1.276] |
| Gemma | Entity mention | Present | 2.353 [2.114, 2.605] | 1.079 [0.874, 1.287] | −1.274 [−1.468, −1.068] |

**Table D2.** Overall $R$ and its source/replacement components by model, construction, and marker condition. Marker presence lowers the replacement component and makes the source component less negative at the mean level in both constructions and models. Paired component changes, rather than overlap between marginal intervals, are the relevant decomposition of marker effects; their order-stratified estimates are included in the audit report.

The paired changes in the components, where $\Delta R=\Delta R_{replacement}-\Delta R_{source}$, are:

| Model | Order | Construction | $\Delta$ replacement component | $\Delta$ source component |
|---|---|---|---:|---:|
| Qwen | All | Superseded | −0.306 [−0.812, 0.237] | 0.443 [−0.100, 1.040] |
| Qwen | All | Entity mention | −1.335 [−1.640, −1.013] | 0.752 [0.323, 1.140] |
| Qwen | Aligned | Superseded | −0.286 [−0.840, 0.267] | 0.086 [−0.522, 0.704] |
| Qwen | Aligned | Entity mention | −0.810 [−1.328, −0.330] | 0.488 [−0.179, 1.070] |
| Qwen | Reversed | Superseded | −0.327 [−0.995, 0.399] | 0.800 [0.088, 1.557] |
| Qwen | Reversed | Entity mention | −1.859 [−2.283, −1.435] | 1.017 [0.480, 1.492] |
| Gemma | All | Superseded | −0.727 [−1.046, −0.420] | 0.357 [−0.009, 0.720] |
| Gemma | All | Entity mention | −0.653 [−1.003, −0.310] | 0.307 [−0.021, 0.645] |
| Gemma | Aligned | Superseded | −0.547 [−1.049, −0.015] | 0.553 [0.062, 1.085] |
| Gemma | Aligned | Entity mention | −0.679 [−1.101, −0.245] | 0.475 [0.056, 0.909] |
| Gemma | Reversed | Superseded | −0.907 [−1.298, −0.486] | 0.162 [−0.301, 0.629] |
| Gemma | Reversed | Entity mention | −0.626 [−1.085, −0.207] | 0.139 [−0.285, 0.532] |

**Table D3.** Paired marker changes in the $R$ decomposition, with 95% history-bootstrap intervals (2,000 draws, seed 73021; 24 histories). For Qwen entity mention, both component changes exclude zero overall; the source-component interval includes zero when aligned and excludes zero when reversed. In Qwen superseded, both overall component intervals cross zero; only the reversed source-component interval excludes zero. For Gemma, replacement-component changes exclude zero in every stratum for both constructions. Source-component changes exclude zero in aligned order for both constructions; the overall and reversed intervals include zero. The combined $R$ effects in Table D1 remain the primary interpretation.

### D.4 Artifact hashes

| Artifact | SHA-256 |
|---|---|
| Confirmation dataset | `af1b33a73e4235a4e1bc0550ea2bea008a652972f299be0e5da21e8e880b299b` |
| Generation report | `3aae4c6419e2301c44c84410208a137cece6f278a5fda07e1218db51d34dba71` |
| Excluded marker pilot dataset | `f42cc115be1715b46d70587fd5a8e311eed33c8868e34d516e1454774aa4fec4` |
| Qwen scores | `ddf600c12b6d58b84779ffa48571f29dd9f2e64ecc99dbd77b81aceb2461c866` |
| Qwen score provenance | `2627828f16bdf08ffe538883d609f59e72d552593e601744f185046ce94b002e` |
| Gemma scores | `a329720a535ae7ad60db8727d1112173ddedb51b1f1ca2882624df1302914bb1` |
| Gemma score provenance | `3842e40ffd4f42b284692e73d00cb00104360b3da63b396b545cf20e05ca2822` |
| Qwen analysis | `60383f56fe6f29cbe04d7c1c1c75f661d7bd15be6a42b836461d4b160988e9bf` |
| Gemma analysis | `5d2c0fc95c74f080e32fa7cc191d00383132440b830e1ca0dec96d0693648019` |
| uv.lock | `55c8fc8089a4258c33ee2c6ee054abf210ff2006dcdca956b83aebc72cfa14e4` |
| Scoring and analysis code | `1ec47a290a3ceb2a31e0f8d622d8328aa23096f92bd08670bb7aa42956473bc8` |

The complete detailed audit, including order-stratified component changes and the full matching-query/other-query $E$ decomposition, is in [marker_construction_confirmation.md](../docs/marker_construction_confirmation.md). The saved artifacts are [dataset](../outputs/followups/marker_confirmatory.jsonl), [generation report](../outputs/followups/marker_confirmatory_report.json), [Qwen analysis](../outputs/followups/qwen_marker_confirmatory_analysis.json), and [Gemma analysis](../outputs/followups/gemma_marker_confirmatory_analysis.json). Score files have matching `.provenance.json` and `.run.json` sidecars.
