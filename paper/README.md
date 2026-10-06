# What a Stale Value Still Does: Mention Structure and Order Shape Score Sensitivity After In-Context Updates

TMLR manuscript on how mention construction, relative order, and explicit temporal marking affect candidate-score sensitivity to edited historical values. The main paper reports seven screened models from five families; the core relational/order and marker experiments are supplemented by three controlled follow-ups and exploratory generated-answer analyses. These results concern the tested prompts and score estimand, not an identified internal mechanism. `main.tex` holds the preamble and pulls in one file per section from `sections/`. Tables, figures, source analyses, and evidence limits are mapped in [RESULTS_INDEX.md](RESULTS_INDEX.md).

Build: `latexmk -pdf main.tex` (requires a TeX distribution).

Figures: run `python paper/data/export_history_estimates.py` to export available per-history estimates from sealed output JSONs, then run `MPLBACKEND=Agg python paper/figures/make_figures.py` from the repository root with Matplotlib installed. Figure 1 is generated separately by `python scripts/fig_decomposition.py`. The companion CSV/JSON files preserve estimates used for figures and derived contrasts. Exporters require the local `outputs/` artifacts; figure builders use the exported data.
