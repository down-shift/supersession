# Obsolete and Incidental Words Shift Language-Model Decision Preferences

The ACL-format manuscript in [`paper_v2/`](paper_v2/) studies policy decisions after an in-context update. In five open models, obsolete-state edits shift routing preferences and change generated decisions even on histories the models retrieve correctly. An exploratory Qwen3-8B run has fewer superseded switches when the other entity's state supports the correct action, but different histories prevent isolating cue agreement. Thinking-enabled generation produces no superseded switches in either Qwen3 model. Incidental mentions also shift preferences in every model.

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
