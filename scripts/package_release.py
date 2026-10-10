"""Build the anonymized code-and-records archive for the paper_v2 submission.

    uv run --extra dev python -m scripts.package_release [--output dist/anonymous_release.zip]

Contents: the import closure of the stale-decision pipeline, its tests and configs, the dated protocol
records, the sealed analysis reports and the paper's derivation and figure scripts. Raw score files and model
weights are excluded (datasets are regenerated deterministically from the code). The archive is scanned for
identifying strings; the build fails if any is found.
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
docs/stale_decision_v1.md docs/stale_decision_v1_stage1_report.md docs/stale_decision_v1b.md
docs/stale_decision_v1b_development_report.md docs/stale_decision_v2.md docs/stale_decision_v2_confirmation_report.md
docs/stale_decision_v2_models.md docs/stale_decision_v2m_gemma_report.md docs/stale_decision_v2_replication.md
paper_v2/__init__.py paper_v2/data/__init__.py paper_v2/data/derive.py paper_v2/data/check_numbers.py
paper_v2/data/derived.json paper_v2/data/tables.py paper_v2/data/tables/cross_model.tex paper_v2/data/tables/models_gate.tex paper_v2/data/tables/joint_by_model.tex paper_v2/data/tables/gate_rule.tex paper_v2/figures/__init__.py paper_v2/figures/make_figures.py
""".split()
CONFIG_GLOBS = ['configs/stale_decision_v1/*', 'configs/stale_decision_v1b/*', 'configs/stale_decision_v2/*',
                'configs/stale_decision_v2m/*', 'configs/stale_decision_v2r/*']
# Sealed analysis reports and run records (small JSON); raw score files are excluded.
RECORD_GLOBS = ['outputs/stale_decision_v1b/*/*report.json', 'outputs/stale_decision_v2/*/*report.json',
                'outputs/stale_decision_v2m/gemma3_4b/*report.json', 'outputs/stale_decision_v2m/gemma3_4b/gate/*report.json',
                'outputs/stale_decision_v2r/*/*report.json', 'outputs/stale_decision_v2r/*/*/*report.json',
                'outputs/stale_decision_v*/**/*.manifest.json', 'outputs/stale_decision_v*/**/*.complete.json',
                'outputs/stale_decision_v*/**/freeze.json', 'outputs/stale_decision_v*/**/scorer_check.json',
                'outputs/stale_decision_v*/**/causality_check.json', 'outputs/stale_decision_v*/**/*.timing.json']
# Confirmation datasets and per-record scores, so that paper_v2.data.derive runs from the archive.
RAW = ['outputs/stale_decision_v2/confirmation/confirmation.jsonl',
       'outputs/stale_decision_v2/confirmation/confirmation_scores.jsonl',
       'outputs/stale_decision_v2r/qwen3_14b/confirmation/confirmation.jsonl',
       'outputs/stale_decision_v2r/qwen3_14b/confirmation/confirmation_scores.jsonl']
EXTRA = {'scripts/causality_check.py': 'outputs/_dgx_final/causality_check_v2m.py'}
# Identifying strings found in dated protocol records; redacted in the archive copy only (the repository records,
# and the code hashes computed over them, are unchanged).
REDACT = [(re.compile(r'/Users/[A-Za-z0-9_.-]+'), '/Users/<user>'), (re.compile(r'\bgpubox\b'), 'a workstation')]
FORBIDDEN = re.compile(r'kaluzh|jerzy|jrzkaminski|d\.dgx|gpubox|A100-ya|ysda|/Users/(?!<user>)|/data/storage|privaterelay|'
                       r'/work/stale|jhub', re.I)
README = """# Anonymized release: stale-state decisions

Code, protocol records and sealed analysis reports for the submission. Python >= 3.11 with uv.

    uv sync --extra dev                       # CPU: tests, analysis, paper numbers and figures
    uv run --extra dev pytest -q tests        # unit tests
    uv run --extra dev python -m paper_v2.data.derive   # every number in the paper -> paper_v2/data/derived.json
    uv run --extra dev python -m paper_v2.data.tables   # generated LaTeX tables
    uv run --extra dev python -m paper_v2.figures.make_figures

Reproducing a run (GPU, float32, one 80 GB GPU; `uv sync --extra model --extra dev`). Each command validates the
dataset, freeze, scorer check and completion records against the code hash in docs/; see the CLI docstrings.

    python -m scripts.stale_decision_v2 generate --output DIR/development.jsonl
    python -m scripts.stale_decision_v2 token-audit --dataset DIR/development.jsonl --output DIR/token_audit.json
    python scripts/causality_check.py DIR/development.jsonl DIR/causality_check.json configs/stale_decision_v2/protocol.json
    python -m scripts.stale_decision_v2 verify-scorer --dataset DIR/development.jsonl --output DIR/scorer_check.json
    python -m scripts.stale_decision_v2 freeze --output DIR/freeze.json
    python -m scripts.stale_decision_v2 score --dataset DIR/development.jsonl --freeze DIR/freeze.json \\
        --scorer-check DIR/scorer_check.json --output DIR/development_scores.jsonl
    python -m scripts.stale_decision_v2 analyze --dataset DIR/development.jsonl --scores DIR/development_scores.jsonl \\
        --output DIR/development_report.json
    # gate: generate --split frozen_gate --exclude DIR/development.jsonl, then score and analyze
    # confirmation: generate --split confirmation --exclude <dev> --exclude <gate> --gate-dataset <gate>
    #               --gate-scores <gate scores>  (refuses unless the recomputed gate passes), then score and analyze
    # other models: scripts.stale_decision_v2m / v2r with --config configs/stale_decision_v2{m,r}/<model>.json

Protocol records (docs/) are dated; amendments are appended, never rewritten. outputs/ holds the sealed
analysis reports and run records (manifests, completion records, freezes, scorer and causality checks), and
the confirmation datasets and per-record score files of both confirmed models (checked against their
completion records by derive.py). Development and gate score files are available on request after review.

Anonymization: identifying strings (a local home path and a hostname) were redacted in the archive copies of
the files listed in REDACTED.txt. Code hashes in the sealed reports were computed over the unredacted
records, so re-verifying those hashes needs the originals; regenerated runs are self-consistent.
"""


def collect():
    files = {}
    for p in CODE:
        if (ROOT / p).exists():
            files[p] = ROOT / p
    for g in CONFIG_GLOBS + RECORD_GLOBS:
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
