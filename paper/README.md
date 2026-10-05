# TMLR manuscript

Manuscript in the TMLR template. `main.tex` holds the preamble and pulls in one file per section from `sections/`. The paper's tables, figures, source analyses, and their limits are indexed in [RESULTS_INDEX.md](RESULTS_INDEX.md).

Build: `latexmk -pdf main.tex` (requires a TeX distribution).

Figures: `python paper/data/export_history_estimates.py` exports per-history estimates from the sealed output JSONs; then run `MPLBACKEND=Agg python paper/figures/make_figures.py` from the repository root with Matplotlib installed. The companion CSVs preserve the estimates used for the figures and paired contrasts. The exporter requires the local `outputs/` artifacts; the figure builder does not.
