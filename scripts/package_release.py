"""Build the anonymized code-and-records archive for the paper_v2 submission.

    uv run --extra dev python -m scripts.package_release [--output dist/anonymous_release.zip]

Contents: the import closure of the stale-decision pipeline, its tests and configs, current documentation,
manuscript protocol sources, sealed analysis reports, confirmation datasets and scores, and the paper's
derivation and figure scripts. Model weights are excluded. The archive is scanned for identifying strings;
the build fails if any is found.
"""
import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = """
src/__init__.py src/utils.py src/models/__init__.py src/models/loader.py
src/data/__init__.py src/data/generate.py src/data/io.py src/data/stale_decision_v1.py src/data/stale_decision_v2.py
src/data/status_focal.py src/data/status_prompt_gate.py src/data/supersession_behavior.py src/data/token_validation.py
src/analysis/__init__.py src/analysis/natural_competence.py src/analysis/stale_decision_v1.py
src/analysis/stale_decision_v1b.py src/analysis/stale_decision_v2.py
src/cross_model/__init__.py src/cross_model/adapters.py src/cross_model/progress.py src/cross_model/protocol.py
src/cross_model/runtime.py src/cross_model/scoring.py src/cross_model/tokens.py
src/experiments/__init__.py src/experiments/stale_decision_v1.py src/experiments/stale_decision_v1b.py
src/experiments/stale_decision_v2.py src/experiments/stale_decision_v2m.py src/experiments/stale_decision_v2r.py
scripts/__init__.py scripts/stale_decision_v1.py scripts/stale_decision_v1b.py scripts/stale_decision_v1b_summary.py
scripts/stale_decision_v2.py scripts/stale_decision_v2m.py scripts/stale_decision_v2r.py
tests/__init__.py tests/test_stale_decision_v1.py tests/test_stale_decision_v1b.py tests/test_stale_decision_v2.py
tests/test_stale_decision_v2m.py tests/test_stale_decision_v2r.py
pyproject.toml uv.lock
docs/stale_decision_v2.md paper_v2/README.md paper_v2/RESULTS_INDEX.md
paper_v2/main.tex paper_v2/acl.sty paper_v2/acl_natbib.bst paper_v2/references.bib
paper_v2/__init__.py paper_v2/data/__init__.py paper_v2/data/derive.py paper_v2/data/check_numbers.py
paper_v2/data/derived.json paper_v2/data/tables.py paper_v2/figures/__init__.py paper_v2/figures/make_figures.py
""".split()
SOURCE_GLOBS = ['src/data/stale_decision*.py', 'src/experiments/stale_decision*.py',
                'src/analysis/stale_decision*.py', 'scripts/stale_decision*.py',
                'tests/test_stale_decision*.py', 'paper_v2/sections/*.tex',
                'paper_v2/data/tables/*.tex', 'paper_v2/figures/*.pdf']
CONFIG_GLOBS = ['configs/stale_decision_*/*']
# Sealed analysis reports and run records (small JSON); raw score files are excluded.
RECORD_GLOBS = ['outputs/stale_decision_*/**/*report.json',
                'outputs/stale_decision_*/**/*.manifest.json', 'outputs/stale_decision_*/**/*.complete.json',
                'outputs/stale_decision_*/**/freeze.json', 'outputs/stale_decision_*/**/scorer_check.json',
                'outputs/stale_decision_*/**/causality_check.json', 'outputs/stale_decision_*/**/*.timing.json']
# Confirmation datasets and per-record scores, so that paper_v2.data.derive runs from the archive.
RAW = [f'{directory}/{name}'
       for directory in ('outputs/stale_decision_v2/confirmation',
                         'outputs/stale_decision_v2r/qwen3_14b/confirmation',
                         'outputs/stale_decision_v2f/gemma4_12b/confirmation',
                         'outputs/stale_decision_v2f/qwen35_9b/confirmation',
                         'outputs/stale_decision_v2f/granite41_8b/confirmation')
       for name in ('confirmation.jsonl', 'confirmation_scores.jsonl')]
EXTRA = {'scripts/causality_check.py': 'outputs/_dgx_final/causality_check_v2m.py'}
# Identifying strings are redacted in archive copies only.
REDACT = [(re.compile(r'/Users/[A-Za-z0-9_.-]+'), '/Users/<user>'), (re.compile(r'\bgpubox\b'), 'a workstation')]
FORBIDDEN = re.compile(r'kaluzh|jerzy|jrzkaminski|d\.dgx|gpubox|A100-ya|ysda|/Users/(?!<user>)|/data/storage|privaterelay|'
                       r'/work/stale|jhub', re.I)
README = """# Anonymized release: stale-state decisions

Code, current documentation, manuscript protocol sources and experiment artifacts for the five-model submission. Python >= 3.11 with uv.

```bash
uv sync --extra dev
uv run --extra dev python -m paper_v2.data.derive
uv run --extra dev python -m paper_v2.data.tables
uv run --extra dev python -m paper_v2.figures.make_figures
```

The archive includes confirmation datasets and per-record scores for the five confirmed models, sealed analysis reports, and available manifests, completion records, freezes, scorer/causality checks and timings. Derivation checks confirmation scores against their completion records. Model weights are excluded.

See `docs/stale_decision_v2.md` for the design, controls, model results and reporting limits; `paper_v2/RESULTS_INDEX.md` maps the evidence and `paper_v2/README.md` gives build and reproduction instructions. Protocol revisions are recorded in `paper_v2/sections/A-protocol-history.tex`.

GPU inference requires the model extra, pinned revisions, unquantized float32 and a single GPU with sufficient memory. The reported runs used an 80 GB H100. The runners hash their implementation and retained manuscript protocol sources. Documentation cleanup changes these hashes; verifying or resuming historical freezes requires the original code revision and protocol records. New runs must create fresh freezes and output paths.

Anonymization redacts identifying paths and hostnames only in archive copies listed in `REDACTED.txt`. Verifying hashes of affected historical records requires the original files.
"""


def collect():
    files = {}
    for p in CODE:
        if (ROOT / p).exists():
            files[p] = ROOT / p
    for g in SOURCE_GLOBS + CONFIG_GLOBS + RECORD_GLOBS:
        for p in ROOT.glob(g):
            if p.is_file() and 'invalid' not in p.parts and '_dgx_final' not in p.parts:
                files[str(p.relative_to(ROOT))] = p
    for p in RAW:
        files[p] = ROOT / p
    for dest, src in EXTRA.items():
        files[dest] = ROOT / src
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='dist/anonymous_release.zip')
    a = ap.parse_args()
    files = collect()
    hits, texts, redacted = [], {}, []
    for name, path in files.items():
        text = path.read_text(errors='ignore')
        new = text
        for pat, rep in REDACT:
            new = pat.sub(rep, new)
        if new != text:
            redacted.append(name)
            texts[name] = new
        text = new
        for m in FORBIDDEN.finditer(text):
            hits.append(f'{name}: {text[max(0, m.start() - 40):m.end() + 40]!r}')
    if FORBIDDEN.search(README):
        hits.append('README')
    if hits:
        raise SystemExit('identifying strings found:\n' + '\n'.join(hits[:40]))
    out = ROOT / a.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('README.md', README)
        z.writestr('REDACTED.txt', '\n'.join(sorted(redacted)) + '\n')
        for name, path in sorted(files.items()):
            if name in texts:
                z.writestr(name, texts[name])
            else:
                z.write(path, name)
    print(f'{out}: {len(files) + 2} files, {out.stat().st_size / 1e6:.1f} MB; redacted: {redacted}')


if __name__ == '__main__':
    main()
