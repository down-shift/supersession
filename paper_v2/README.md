# paper_v2 — stale-decision manuscript (ACL format)

Separate from the submitted TMLR source in `paper/`, which is unchanged.

    uv run --extra dev python -m paper_v2.data.derive          # numbers -> paper_v2/data/derived.json
    uv run --extra dev python -m paper_v2.figures.make_figures # figures/*.pdf
    cd paper_v2 && latexmk -pdf main.tex

`derive.py` reads the sealed reports under `outputs/stale_decision_{v1b,v2,v2m}` and the confirmation score
file. It checks the score file against its completion record and stores the SHA-256 of every input. Style files:
see STYLE_SOURCE.txt. Assessment: docs/rescue/PUBLICATION_ASSESSMENT.md.
