# When Old Values and Incidental Mentions Change LLM Routing Decisions

The current manuscript studies whether an edit to an earlier line changes a model's routing decision when the current state, policy, and correct route stay fixed. On layouts selected by a separate screening split, all five models chose the wrong route more often after the edit, both for former states and incidental mentions. The mechanism remains unresolved.

- [Current paper and build instructions](paper_dai/README.md)
- [Manuscript guide](docs/stale_decision_v2.md)
- [Earlier ACL manuscript and results index](paper_v2/RESULTS_INDEX.md)

## Environment

The repository uses uv and the Python version in `.python-version`.

```bash
uv sync --extra dev
```

For model inference, use `uv sync --extra model --extra dev`. The reported runs used unquantized float32, SDPA, and one 80 GB H100. Configurations are under `configs/stale_decision_*`.

## Reproduction

The committed `paper_v2/data/derived.json` contains aggregate results. Reproducing the original analysis from raw outputs requires the sealed experiment artifacts. They were unavailable during the manuscript audit. See the [author evidence audit](paper_dai/README.md).
