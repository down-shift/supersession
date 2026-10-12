# Obsolete and Incidental Words Shift Language-Model Decisions, but Change Them Mainly When Another Cue Agrees

The ACL-format manuscript in [`paper_v2/`](paper_v2/) studies policy decisions after an in-context update. In five open models, obsolete-state edits shift routing preferences and change generated decisions even on histories the models retrieve correctly. Decision switches occur mainly when the other entity's current state also implies the wrong action; they fall sharply on Qwen3-8B's balanced histories and disappear for superseded assignments with thinking enabled in both Qwen3 models. Incidental mentions also shift preferences in every model.

## Documentation

- [Manuscript guide](docs/stale_decision_v2.md): design, model results, controls and claim boundaries.
- [Build and reproduction](paper_v2/README.md): ACL build, tables, figures and required artifacts.
- [Results index](paper_v2/RESULTS_INDEX.md): evidence behind each result, table and figure.
- [Protocol history](paper_v2/sections/A-protocol-history.tex): dated revisions and analysis choices.

## Environment

The repository uses uv, with Python selected by `.python-version`. Install the analysis and development environment:

```bash
uv sync --extra dev
```

For model inference:

```bash
uv sync --extra model --extra dev
```

The reported runs use pinned model/tokenizer revisions, unquantized float32, SDPA, and one 80 GB H100. Configurations are under `configs/stale_decision_*`.

## Reproduction

Tables and figures can be regenerated from the committed `paper_v2/data/derived.json`. Re-deriving raw-score results requires the sealed experiment artifacts listed in the [reproduction instructions](paper_v2/README.md).

Historical frozen artifacts require their recorded Git revision and original protocol files for hash verification or resume. The documentation cleanup changes provenance hashes for new runs.
