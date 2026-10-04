# TMLR manuscript

Manuscript in the TMLR template. `main.tex` holds the preamble and pulls in one file per section from `sections/`.

Build: `latexmk -pdf main.tex`

Figures: `cd figures && uv run --no-project --with matplotlib python make_figures.py` (it plots the estimates reported in Tables 4, 6, and B1).
