# paper_v2 — stale-decision manuscript (ACL format)

This is a separate manuscript and evidence line from the TMLR source in `paper/`. The paper studies whether editing an obsolete state changes policy decisions, using synthetic routing and access records. In the frozen Qwen3-8B confirmation, routing met the competence criteria and access did not. The obsolete-state edit shifted routing wrong-action log-odds by 3.133 nats and produced a net wrong-action switch rate of 0.185. Four further models replicated routing preference shifts; matched incidental mentions also shifted preferences across models, showing that the effect is not unique to superseded assignments. See [`docs/paper_claims.md`](../docs/paper_claims.md) for claim boundaries and [`sections/08-limitations.tex`](sections/08-limitations.tex) for scope.

The results are behavioral evidence from short synthetic tasks. The paper does not identify an internal mechanism or test a mitigation. Access results for Qwen3-8B are below the competence criteria, and two screened models have no confirmation because they failed the gate.

    uv run --extra dev python -m paper_v2.data.derive          # numbers -> paper_v2/data/derived.json
    uv run --extra dev python -m paper_v2.figures.make_figures # figures/*.pdf
    cd paper_v2 && latexmk -pdf main.tex

`derive.py` reads the sealed reports under `outputs/stale_decision_{v1b,v2,v2m}` and the confirmation score
file. It checks the score file against its completion record and stores the SHA-256 of every input. Style files:
see STYLE_SOURCE.txt. Assessment: docs/rescue/PUBLICATION_ASSESSMENT.md.
