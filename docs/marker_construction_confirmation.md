# Marker × construction confirmation

Verified against saved artifacts on 2026-10-04. This is the prespecified 24-history confirmation, separate from the two-history marker pilot and relational v2.

Adding `Previously` reduces the v2 R sensitivity measure within both constructions for both models. Qwen shows a positive marker × construction interaction: the reduction is larger in entity-mention controls. Gemma shows reductions of similar magnitude; its interaction interval includes zero, so the construction difference is unresolved for Gemma. This is evidence that marker presence affects measured sensitivity. It does not establish active suppression, a stale-answer effect, or a construction contrast that isolates conflict from syntax.

## Verification and scope

- Both models have 3,072/3,072 scored members, covering all 24 histories and all factorial cells; there are 2,272 unique prompts per model.
- Both run records and sealed score provenance report `complete`. Dataset, score, report, and provenance hashes agree with the sealed analyses; analysis results were recomputed from the saved scores and matched exactly.
- The dataset explicitly excludes both rerun1 marker-pilot histories, with zero full-history or target-history signature overlap. Older unscored preparation artifacts and pilot results are excluded from these estimates.
- Both models use the same dataset (generation seed 23261011, scoring seed 20261006), pinned model/tokenizer revisions, and Python 3.13.5. The recorded uv.lock hash matches the current repository lock.
- Scoring mode is `candidate`: these confirmation files contain continuation masses and no generated-answer categories. Accuracy, stale-answer frequency, and answer transitions cannot be estimated from these files.

The generation report says model evaluation was pending because it was written before inference. The completed score sidecars and analyses establish the current execution status; the original generation report is preserved.

## Marker effects and interaction

Marker effect = R with `Previously` minus R without it. Interaction = superseded marker effect minus entity-mention marker effect. Aligned means historical and current entity order agree; reversed means they differ. Intervals are 95% percentile history-bootstrap intervals, using 2,000 draws and seed 73021. Each estimate uses all 24 histories. These are pointwise intervals for the prespecified contrasts and strata.

| Model | Order | Superseded marker effect | Entity-mention marker effect | Interaction |
|---|---|---:|---:|---:|
| Qwen | All | −0.750 [−0.917, −0.590] | −2.087 [−2.410, −1.772] | 1.337 [0.976, 1.712] |
| Qwen | Aligned | −0.372 [−0.552, −0.187] | −1.298 [−1.580, −0.979] | 0.926 [0.581, 1.274] |
| Qwen | Reversed | −1.127 [−1.324, −0.939] | −2.876 [−3.354, −2.429] | 1.749 [1.236, 2.274] |
| Gemma | All | −1.085 [−1.308, −0.864] | −0.960 [−1.183, −0.727] | −0.125 [−0.390, 0.136] |
| Gemma | Aligned | −1.100 [−1.359, −0.851] | −1.155 [−1.384, −0.915] | 0.055 [−0.233, 0.328] |
| Gemma | Reversed | −1.069 [−1.372, −0.785] | −0.765 [−1.123, −0.387] | −0.304 [−0.753, 0.116] |

For Qwen, the interaction is positive in both order strata, and every marker-effect interval is below zero. For Gemma, every within-construction marker-effect interval is below zero, while the overall and both order-stratum interaction intervals include zero. A zero-crossing interaction interval is inconclusive about a construction difference and does not establish equivalence. R remains positive with the marker in every model/construction/order cell.

The Qwen interaction agrees in direction with its exploratory pilot. Gemma's inconclusive overall pilot interaction remains inconclusive in confirmation; its pilot aligned-stratum interaction does not persist as a clear difference in the 24-history confirmation. Pilot histories are not pooled with confirmation.

## R and its source/replacement decomposition

For each baseline→historical-value edit, E = change in replacement log mass minus change in source log mass. R subtracts E for the nonmatching query entity from E for the matching query entity, then symmetrizes across edited entities. The same transformation is applied to each E component, so R = R_replacement − R_source. Values are means with 95% history-bootstrap intervals (2,000 draws, seed 73021; 24 histories).

| Model | Order | Construction | Previously | R | R replacement component | R source component |
|---|---|---|---|---:|---:|---:|
| Qwen | All | Superseded | Absent | 3.738 [3.536, 3.934] | 1.590 [1.118, 2.005] | −2.149 [−2.623, −1.705] |
| Qwen | All | Superseded | Present | 2.989 [2.791, 3.186] | 1.283 [0.832, 1.746] | −1.705 [−2.151, −1.242] |
| Qwen | All | Entity-mention | Absent | 7.304 [6.629, 7.966] | 3.875 [3.477, 4.277] | −3.429 [−4.135, −2.764] |
| Qwen | All | Entity-mention | Present | 5.217 [4.749, 5.705] | 2.540 [2.171, 2.917] | −2.677 [−3.193, −2.165] |
| Qwen | Aligned | Superseded | Absent | 3.547 [3.320, 3.761] | 1.640 [1.190, 2.082] | −1.908 [−2.384, −1.428] |
| Qwen | Aligned | Superseded | Present | 3.175 [2.957, 3.385] | 1.354 [0.854, 1.878] | −1.821 [−2.370, −1.248] |
| Qwen | Aligned | Entity-mention | Absent | 5.841 [5.303, 6.399] | 3.107 [2.553, 3.651] | −2.734 [−3.533, −1.969] |
| Qwen | Aligned | Entity-mention | Present | 4.543 [4.141, 4.962] | 2.296 [1.934, 2.644] | −2.246 [−2.763, −1.723] |
| Qwen | Reversed | Superseded | Absent | 3.929 [3.692, 4.165] | 1.540 [0.958, 2.108] | −2.390 [−2.947, −1.865] |
| Qwen | Reversed | Superseded | Present | 2.802 [2.589, 3.019] | 1.213 [0.637, 1.809] | −1.589 [−2.169, −0.997] |
| Qwen | Reversed | Entity-mention | Absent | 8.766 [7.913, 9.630] | 4.642 [4.204, 5.065] | −4.124 [−4.840, −3.377] |
| Qwen | Reversed | Entity-mention | Present | 5.890 [5.326, 6.489] | 2.783 [2.304, 3.307] | −3.107 [−3.721, −2.508] |
| Gemma | All | Superseded | Absent | 2.631 [2.409, 2.863] | 1.413 [1.170, 1.652] | −1.218 [−1.501, −0.934] |
| Gemma | All | Superseded | Present | 1.546 [1.379, 1.714] | 0.685 [0.445, 0.932] | −0.861 [−1.115, −0.610] |
| Gemma | All | Entity-mention | Absent | 3.313 [2.975, 3.669] | 1.732 [1.427, 2.048] | −1.582 [−1.915, −1.276] |
| Gemma | All | Entity-mention | Present | 2.353 [2.114, 2.605] | 1.079 [0.874, 1.287] | −1.274 [−1.468, −1.068] |
| Gemma | Aligned | Superseded | Absent | 2.817 [2.609, 3.038] | 1.354 [1.020, 1.677] | −1.463 [−1.822, −1.121] |
| Gemma | Aligned | Superseded | Present | 1.717 [1.510, 1.932] | 0.807 [0.449, 1.164] | −0.910 [−1.253, −0.546] |
| Gemma | Aligned | Entity-mention | Absent | 4.352 [4.014, 4.723] | 2.275 [1.929, 2.622] | −2.076 [−2.457, −1.726] |
| Gemma | Aligned | Entity-mention | Present | 3.197 [2.867, 3.536] | 1.596 [1.263, 1.932] | −1.601 [−1.869, −1.326] |
| Gemma | Reversed | Superseded | Absent | 2.445 [2.139, 2.759] | 1.471 [1.108, 1.797] | −0.974 [−1.334, −0.612] |
| Gemma | Reversed | Superseded | Present | 1.376 [1.136, 1.628] | 0.564 [0.270, 0.858] | −0.812 [−1.097, −0.550] |
| Gemma | Reversed | Entity-mention | Absent | 2.275 [1.891, 2.630] | 1.188 [0.763, 1.637] | −1.087 [−1.506, −0.656] |
| Gemma | Reversed | Entity-mention | Present | 1.510 [1.234, 1.780] | 0.562 [0.337, 0.790] | −0.947 [−1.232, −0.651] |

The paired marker changes in the R components by construction and order are below. ΔR = Δreplacement − Δsource; bootstrap each history-level component change with the same 2,000 draws and seed 73021.

| Model | Order | Construction | Δreplacement component | Δsource component |
|---|---|---|---:|---:|
| Qwen | All | Superseded | −0.306 [−0.812, 0.237] | 0.443 [−0.100, 1.040] |
| Qwen | All | Entity-mention | −1.335 [−1.640, −1.013] | 0.752 [0.323, 1.140] |
| Qwen | Aligned | Superseded | −0.286 [−0.840, 0.267] | 0.086 [−0.522, 0.704] |
| Qwen | Aligned | Entity-mention | −0.810 [−1.328, −0.330] | 0.488 [−0.179, 1.070] |
| Qwen | Reversed | Superseded | −0.327 [−0.995, 0.399] | 0.800 [0.088, 1.557] |
| Qwen | Reversed | Entity-mention | −1.859 [−2.283, −1.435] | 1.017 [0.480, 1.492] |
| Gemma | All | Superseded | −0.727 [−1.046, −0.420] | 0.357 [−0.009, 0.720] |
| Gemma | All | Entity-mention | −0.653 [−1.003, −0.310] | 0.307 [−0.021, 0.645] |
| Gemma | Aligned | Superseded | −0.547 [−1.049, −0.015] | 0.553 [0.062, 1.085] |
| Gemma | Aligned | Entity-mention | −0.679 [−1.101, −0.245] | 0.475 [0.056, 0.909] |
| Gemma | Reversed | Superseded | −0.907 [−1.298, −0.486] | 0.162 [−0.301, 0.629] |
| Gemma | Reversed | Entity-mention | −0.626 [−1.085, −0.207] | 0.139 [−0.285, 0.532] |

The E table reports the matching and other query entities separately by construction, marker, and history-current order. Values average the balanced per-edit/per-query rows across histories; the full edit-level rows are retained in each analysis JSON. Each cell gives E / replacement component / source component.

| Model | Order | Construction | Previously | Query | E | Replacement component | Source component |
|---|---|---|---|---|---:|---:|---:|
| Qwen | All | Superseded | Absent | Matching | 7.205 | 3.119 | −4.086 |
| Qwen | All | Superseded | Absent | Other | 3.466 | 1.529 | −1.937 |
| Qwen | All | Superseded | Present | Matching | 5.355 | 2.524 | −2.831 |
| Qwen | All | Superseded | Present | Other | 2.367 | 1.241 | −1.126 |
| Qwen | All | Entity-mention | Absent | Matching | 13.026 | 6.680 | −6.346 |
| Qwen | All | Entity-mention | Absent | Other | 5.722 | 2.805 | −2.917 |
| Qwen | All | Entity-mention | Present | Matching | 9.379 | 4.682 | −4.697 |
| Qwen | All | Entity-mention | Present | Other | 4.162 | 2.143 | −2.020 |
| Qwen | Aligned | Superseded | Absent | Matching | 6.754 | 3.055 | −3.699 |
| Qwen | Aligned | Superseded | Absent | Other | 3.207 | 1.415 | −1.791 |
| Qwen | Aligned | Superseded | Present | Matching | 5.534 | 2.710 | −2.825 |
| Qwen | Aligned | Superseded | Present | Other | 2.359 | 1.356 | −1.003 |
| Qwen | Aligned | Entity-mention | Absent | Matching | 11.240 | 5.787 | −5.452 |
| Qwen | Aligned | Entity-mention | Absent | Other | 5.399 | 2.680 | −2.718 |
| Qwen | Aligned | Entity-mention | Present | Matching | 8.210 | 4.124 | −4.085 |
| Qwen | Aligned | Entity-mention | Present | Other | 3.667 | 1.828 | −1.839 |
| Qwen | Reversed | Superseded | Absent | Matching | 7.655 | 3.182 | −4.473 |
| Qwen | Reversed | Superseded | Absent | Other | 3.726 | 1.643 | −2.083 |
| Qwen | Reversed | Superseded | Present | Matching | 5.176 | 2.339 | −2.838 |
| Qwen | Reversed | Superseded | Present | Other | 2.374 | 1.126 | −1.248 |
| Qwen | Reversed | Entity-mention | Absent | Matching | 14.812 | 7.572 | −7.239 |
| Qwen | Reversed | Entity-mention | Absent | Other | 6.045 | 2.930 | −3.116 |
| Qwen | Reversed | Entity-mention | Present | Matching | 10.549 | 5.241 | −5.308 |
| Qwen | Reversed | Entity-mention | Present | Other | 4.658 | 2.457 | −2.201 |
| Gemma | All | Superseded | Absent | Matching | 6.529 | 3.335 | −3.193 |
| Gemma | All | Superseded | Absent | Other | 3.898 | 1.923 | −1.975 |
| Gemma | All | Superseded | Present | Matching | 2.958 | 1.429 | −1.529 |
| Gemma | All | Superseded | Present | Other | 1.412 | 0.744 | −0.668 |
| Gemma | All | Entity-mention | Absent | Matching | 6.435 | 3.475 | −2.960 |
| Gemma | All | Entity-mention | Absent | Other | 3.121 | 1.743 | −1.378 |
| Gemma | All | Entity-mention | Present | Matching | 5.263 | 2.667 | −2.596 |
| Gemma | All | Entity-mention | Present | Other | 2.910 | 1.588 | −1.321 |
| Gemma | Aligned | Superseded | Absent | Matching | 6.299 | 3.149 | −3.150 |
| Gemma | Aligned | Superseded | Absent | Other | 3.482 | 1.795 | −1.687 |
| Gemma | Aligned | Superseded | Present | Matching | 2.911 | 1.495 | −1.416 |
| Gemma | Aligned | Superseded | Present | Other | 1.194 | 0.688 | −0.506 |
| Gemma | Aligned | Entity-mention | Absent | Matching | 7.225 | 3.788 | −3.437 |
| Gemma | Aligned | Entity-mention | Absent | Other | 2.873 | 1.512 | −1.361 |
| Gemma | Aligned | Entity-mention | Present | Matching | 5.730 | 2.947 | −2.782 |
| Gemma | Aligned | Entity-mention | Present | Other | 2.532 | 1.352 | −1.181 |
| Gemma | Reversed | Superseded | Absent | Matching | 6.758 | 3.522 | −3.236 |
| Gemma | Reversed | Superseded | Absent | Other | 4.314 | 2.051 | −2.263 |
| Gemma | Reversed | Superseded | Present | Matching | 3.006 | 1.364 | −1.642 |
| Gemma | Reversed | Superseded | Present | Other | 1.630 | 0.800 | −0.830 |
| Gemma | Reversed | Entity-mention | Absent | Matching | 5.645 | 3.163 | −2.482 |
| Gemma | Reversed | Entity-mention | Absent | Other | 3.370 | 1.974 | −1.395 |
| Gemma | Reversed | Entity-mention | Present | Matching | 4.797 | 2.387 | −2.409 |
| Gemma | Reversed | Entity-mention | Present | Other | 3.287 | 1.825 | −1.462 |

At the mean level, marker presence lowers the replacement component and makes the source component less negative in both constructions. For Qwen entity-mention, the replacement-component change excludes zero overall and in both order strata; the source-component change excludes zero overall and in reversed order, but its aligned interval includes zero. In Qwen superseded, both overall component intervals cross zero; the reversed stratum has a positive source-component change with an interval excluding zero, while aligned intervals include zero. For Gemma, replacement-component changes exclude zero overall and in both order strata for both constructions. Source-component changes exclude zero in aligned order for both constructions, while the overall and reversed intervals include zero. The combined R marker effects remain the primary interpretation.


## Runtime and provenance

Runs used the NVIDIA GeForce RTX 5080 with CUDA 13.0. Qwen used int8 weights/float16 activations; Gemma used int8 weights/bfloat16 activations. Both scored 2,272 new unique prompts with no resumed prompts. Runtime is the recorded elapsed time for the scoring invocation.

| Model | Elapsed | Seconds/unique prompt | Completed UTC |
|---|---:|---:|---|
| Qwen | 1551.08 s (25.85 min) | 0.683 | 2026-10-04T17:37:05Z |
| Gemma | 657.75 s (10.96 min) | 0.290 | 2026-10-04T17:48:13Z |

These candidate-only timings should not be used as runtimes for the harder-task tests, which also generate answers.

| Item | Revision or SHA-256 |
|---|---|
| Qwen/Qwen3-8B model/tokenizer | `b968826d9c46dd6066d109eabc6255188de91218` |
| google/gemma-3-4b-it model/tokenizer | `093f9f388b31de276ce2de164bdc2081324b9767` |
| Confirmation dataset | `af1b33a73e4235a4e1bc0550ea2bea008a652972f299be0e5da21e8e880b299b` |
| Generation report | `3aae4c6419e2301c44c84410208a137cece6f278a5fda07e1218db51d34dba71` |
| Excluded marker pilot dataset | `f42cc115be1715b46d70587fd5a8e311eed33c8868e34d516e1454774aa4fec4` |
| Qwen scores | `ddf600c12b6d58b84779ffa48571f29dd9f2e64ecc99dbd77b81aceb2461c866` |
| Gemma scores | `a329720a535ae7ad60db8727d1112173ddedb51b1f1ca2882624df1302914bb1` |
| Qwen analysis | `60383f56fe6f29cbe04d7c1c1c75f661d7bd15be6a42b836461d4b160988e9bf` |
| Gemma analysis | `5d2c0fc95c74f080e32fa7cc191d00383132440b830e1ca0dec96d0693648019` |
| uv.lock | `55c8fc8089a4258c33ee2c6ee054abf210ff2006dcdca956b83aebc72cfa14e4` |
| Scoring/analysis code | `1ec47a290a3ceb2a31e0f8d622d8328aa23096f92bd08670bb7aa42956473bc8` |

Full artifacts: [dataset](../outputs/followups/marker_confirmatory.jsonl), [generation report](../outputs/followups/marker_confirmatory_report.json), [Qwen analysis](../outputs/followups/qwen_marker_confirmatory_analysis.json), [Gemma analysis](../outputs/followups/gemma_marker_confirmatory_analysis.json). Score files have matching `.provenance.json` and `.run.json` sidecars.

## Remaining work

The marker × construction confirmation is complete for both models. The separate harder-task n=4/n=6 development and any eligible harder-task test remain pending in this checkout. No harder-task selection or test freeze is inferred from the marker results. Relational v2 results and the paper are unchanged.
