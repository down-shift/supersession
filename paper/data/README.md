# Per-history result data

These compact derived tables preserve the history-level estimates needed to regenerate the manuscript's main figures and recompute their percentile history-bootstrap intervals (2,000 draws; seed 73021). They contain no prompts or token scores and do not replace the sealed analyses. `phi_followup_output_diagnostic.csv` records the strict exact-answer counts and a separate post hoc candidate-prefix count for the Phi-4-mini development runs; its prefix diagnostic does not redefine correctness.

`exp1_history_estimates.csv` contains 96 histories per model, the target attribute and orientation, the five non-live construction-specific aligned/reversed $R$ estimates, their paired order differences, and the overall and order-stratified superseded-minus-control contrasts. `exp2_marker_history_estimates.csv` contains 24 histories per model, construction- and marker-specific $R$ estimates by order, paired marker effects/interactions, and entity-mention-minus-superseded gaps by marker and order.

The Experiment 1 and marker tables were exported by `export_history_estimates.py` from these local analysis JSONs:

| Source | SHA-256 |
|---|---|
| `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004/qwen3_8b/confirmatory_analysis.json` | `b04596b0b8e2fa1c7387da9f92fac7cd2192c2b213943495a7a3d2e42601ec20` |
| `outputs/cross_model_relational_v2/factorial_relation_counterbalanced_runtimefix_20261004/gemma3_4b/confirmatory_analysis.json` | `145771c4a8cdb487c58627168fd3cacd96b269bc0afd92055faa5aaa15e0ec7e` |
| `outputs/followups/qwen_marker_confirmatory_analysis.json` | `60383f56fe6f29cbe04d7c1c1c75f661d7bd15be6a42b836461d4b160988e9bf` |
| `outputs/followups/gemma_marker_confirmatory_analysis.json` | `5d2c0fc95c74f080e32fa7cc191d00383132440b830e1ca0dec96d0693648019` |

CSV SHA-256: `exp1_history_estimates.csv` = `938332f6d1ddf9087a55bebbd14e91386527eff6081e9d3ae0f883978f3e6799`; `exp2_marker_history_estimates.csv` = `074f480020acafa7a70e6d5f07773a2187eb0d16eacccaa501f0206b399b4fb2`.

`export_phi_followup_diagnostic.py` creates `phi_followup_output_diagnostic.csv` from `outputs/followups/phi4_mini_development_n{2,4,6}_scores.jsonl`; those source score hashes are recorded in `docs/followup_experiments.md`. The CSV SHA-256 is `8b8921ca53ab88ab65db2fb2c5a52da9cb3991ee754ac559ff4e614718a5e58b`.

The upstream source analysis JSONs and raw prompt/score artifacts are excluded from Git. These derived files make the estimates and intervals used in the manuscript inspectable and reproducible, but they do not permit independent reconstruction of scoring or all raw-data checks.

Added 2026-10-05 by the same export script: `exp2_marker96_history_estimates.csv` (96-history marker run; from `outputs/followups/marker96/{qwen,gemma}_marker_analysis.json`, SHA-256 `0ee53192…` and `0dbc4d61…`) and `distance_history_estimates.csv` (from `outputs/followups/distance_v1/{qwen,gemma}_distance_analysis.json`, `351be488…` and `d1adea1a…`). Re-running the script reproduces the two earlier CSVs byte for byte. `derived_estimates.py` bootstraps contrasts quoted in the text but not stored in the sealed analyses (marker-absent construction gaps) into `derived_estimates.json`, and also reproduces the sealed marker interaction as a check.

CSV SHA-256: `exp2_marker96_history_estimates.csv` = `7b9c33f37e7665889bcdb9af1cd00585136e44dbb90db234717701fba84642ea`; `distance_history_estimates.csv` = `8e07569a6ccc9768850564bee77c85f554f46274065b92a0eb60288579f909e9`.

