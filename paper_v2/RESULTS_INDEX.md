# Stale-decision results and reproducibility index

This index maps the current `paper_v2` manuscript to its committed evidence. The [manuscript guide](../docs/stale_decision_v2.md) explains the estimates and populations; the [README](README.md) gives build and regeneration commands. The earlier TMLR results have their own [index](../paper/RESULTS_INDEX.md).

All data keys below refer to [`data/derived.json`](data/derived.json). Source-artifact paths and SHA-256 hashes are stored under its `inputs` key. The sealed input reports and raw score files are absent from this checkout; the committed derived data support checking displayed results and rebuilding tables and figures.

| Manuscript evidence | Committed data keys | Display or source |
|---|---|---|
| Qwen3-8B competence, current and initial retrieval | `confirmation.competence`, `confirmation.historical_retrieval`, `confirmation.member_accuracy` | [Results](sections/05-results.tex), [competence figure](figures/competence.pdf) |
| H1 obsolete-state preference shift and generated switches | `confirmation.primary`, `confirmation.switches` | [Results](sections/05-results.tex) |
| Construction effects S, O, M and U | `confirmation.constructions`, `confirmation.primary`, `confirmation.switches` | [Construction figure](figures/constructions.pdf) |
| Histories satisfying joint tracking checks | `confirmation.joint_correctness`, `models.<model>.confirmation.joint_correctness` | [Per-model tracked table](data/tables/joint_by_model.tex); per-cell Qwen3-8B table in [Appendix B](sections/B-tables.tex) |
| Common tracked set across constructions | `confirmation.joint_intersection`, `models.<model>.confirmation.joint_intersection` | [Qwen3-8B table](data/tables/tracked.tex), [per-model table](data/tables/tracked_models.tex) |
| Never-updated other attribute N | `static_other` | [N comparison table](data/tables/static_other.tex); N appears in the Qwen3 construction figures |
| Balanced other-entity cue and thinking generation | `balanced`, `think`, `confirmation.constructions`, `confirmation.switches` | [Conditions figure](figures/conditions.pdf), [conditions table](data/tables/conditions.tex) |
| Five-model compact estimates on confirmation-competent cells | `models.<model>.confirmation.competent_confirmation` | [Models figure](figures/models.pdf), [cross-model table](data/tables/cross_model.tex), [intervals](data/tables/cross_model_ci.tex) |
| Original gate-based compact estimates | `models.<model>.confirmation.competent` | Gate-rule comparison in [Appendix B](sections/B-tables.tex) |
| Failed gates and all attempted models with gate data | `models.<model>.gate_cells`, `gates` | [Gate table](data/tables/models_gate.tex) |
| H3 grouped prediction and easy-query sensitivity | `confirmation.H3`, `confirmation.R_easy_local_superseded`, `models.<model>.confirmation.H3` | H3 table in [Appendix B](sections/B-tables.tex), [main results](sections/05-results.tex) |
| Template dependence across non-confirmatory splits | `replication`, `replication_switches`, `confirmation.primary` | [Split-replication figure](figures/replication.pdf) |
| Vocabulary/order-cell strata and access error diagnosis | `confirmation.strata`, `models.<model>.confirmation.access_diagnosis` | [Discussion](sections/06-discussion.tex), [limitations](sections/08-limitations.tex), [Appendix B](sections/B-tables.tex) |
| Runtime and total recorded compute | `models.<model>.confirmation.run`, `think`, `compute` | [Reproducibility appendix](sections/C-reproducibility.tex) |

[`data/tables.py`](data/tables.py) regenerates the external table files, and [`figures/make_figures.py`](figures/make_figures.py) regenerates all figure PDFs from the committed derived data. Some tables and estimates are written directly in the section sources; [`data/check_numbers.py`](data/check_numbers.py) provides numeric traces for review. A matching rounded number alone does not establish that a sentence refers to the correct estimand.

The compact cross-model table uses confirmation-competent cells; the tracked-history table applies a separate post hoc restriction to all routing histories, with retained sets varying by construction. Common-set comparisons use `joint_intersection`. The tracked analyses and confirmation-based competence reporting are post hoc; original gate-rule estimates remain in the appendix. The balanced-history comparison is descriptive, thinking runs have no preference scores, and access is confirmation-competent only in Gemma 4 12B. See [protocol history](sections/A-protocol-history.tex) for dated decisions and [limitations](sections/08-limitations.tex) for scope.
