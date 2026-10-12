# paper_v2 — stale-decision manuscript (ACL format)

**Obsolete and Incidental Words Shift Language-Model Decisions, but Change Them Mainly When Another Cue Agrees** ([source](main.tex), [PDF](main.pdf)).

Obsolete-state edits shift routing action preferences in five open models, including histories with correct current and initial retrieval. Generated decisions change mainly when the other entity's current state also implies the wrong action. In Qwen3-8B, the superseded net switch rate falls from `0.185` to `0.031` on balanced histories; in both Qwen3 models it is zero with thinking enabled. Incidental mentions also shift preferences in every model. The never-updated other-attribute control shows that interference requires neither supersession nor assignment to the queried attribute.

See the [manuscript guide](../docs/stale_decision_v2.md) for the design, model results, controls, reporting choices and limits. Access meets confirmation competence only in Gemma 4 12B. The tracked-history analyses and confirmation-based selection of competent cells are post hoc; the appendix also gives estimates under the original gate-based rule. The balanced comparison is descriptive and limited to Qwen3-8B. Evidence comes from short synthetic tasks and does not identify a mechanism or establish mitigation efficacy.

[RESULTS_INDEX.md](RESULTS_INDEX.md) maps the manuscript's results, tables and figures to their committed data keys and source artifacts.

## Build from committed assets

From the repository root, with `latexmk` and the required TeX packages installed:

```bash
cd paper_v2
latexmk -pdf main.tex
```

The ACL style and bibliography style are included as [`acl.sty`](acl.sty) and [`acl_natbib.bst`](acl_natbib.bst). The manuscript reads committed tables from `data/tables/` and figures from `figures/`.

## Regenerate tables and figures

From the repository root, using the committed [`data/derived.json`](data/derived.json):

```bash
uv run --extra dev python -m paper_v2.data.tables
uv run --extra dev python -m paper_v2.figures.make_figures
```

`models.pdf` summarizes confirmation-competent routing cells; `conditions.pdf` compares main, balanced and thinking conditions. The construction plots include the never-updated other attribute where available. `competence.pdf` and `replication.pdf` are appendix diagnostics.

## Re-derive evidence from sealed artifacts

The raw experiment inputs are absent from this checkout. Re-derivation requires the sealed reports, datasets, scores, completion records and timings at the paths recorded in `data/derived.json` and [`data/derive.py`](data/derive.py):

| Output namespace | Evidence |
|---|---|
| `outputs/stale_decision_v1b/` | Earlier development and failed gate |
| `outputs/stale_decision_v2/` | Qwen3-8B development, gate and confirmation |
| `outputs/stale_decision_v2m/` | Gemma 3 4B development and failed gate |
| `outputs/stale_decision_v2r/` | Qwen3-14B replication |
| `outputs/stale_decision_v2f/` | Further families and their gates/confirmations |
| `outputs/stale_decision_v2s/` | Never-updated other attribute in both Qwen3 models |
| `outputs/stale_decision_v2b/` | Balanced routing histories in Qwen3-8B |
| `outputs/stale_decision_think/` | Thinking-enabled generation in both Qwen3 models |

With those artifacts available, run this before regenerating tables and figures:

```bash
uv run --extra dev python -m paper_v2.data.derive
```

The derivation records input SHA-256 hashes and checks confirmation scores against completion records. Model and follow-up loaders skip absent reports, so reproducing the full current paper requires every input listed in the committed `derived.json`. [`data/check_numbers.py`](data/check_numbers.py) traces numeric literals to derived values or recorded design constants; its matches require human review because rounded values can coincide.

The [protocol history](sections/A-protocol-history.tex) documents the event-format corrections, numerical precision checks and dated amendments. The [reproducibility appendix](sections/C-reproducibility.tex) records model revisions, float32 runtime, provenance and compute.

The documentation cleanup changes source/protocol hashes used for new runs. Verify or resume historical frozen artifacts from their recorded Git revision, with its original protocol files; the current guide does not replace their hash-bound records.
