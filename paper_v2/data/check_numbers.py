"""List numeric literals in paper_v2/sections/*.tex that cannot be traced to derived.json.

    uv run --extra dev python -m paper_v2.data.check_numbers

A literal is traced if some value in derived.json rounds to it at the literal's displayed precision, or if
it appears in ALLOW with the record it comes from. Integers below 13 (section numbers, small counts in prose)
are skipped. Because derived.json holds thousands of values, short literals can match by coincidence
(a random 3-decimal value in [0, 1] matches ~28% of the time), so the script also writes
paper_v2/data/number_trace.txt: every literal with the derived.json paths it matched, for a human to check
that each path is the quantity the sentence describes.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTIONS = HERE.parent / 'sections'

# Values quoted from protocol records rather than derived.json, with their source.
ALLOW = {
    # design constants: docs/stale_decision_v1.md, v1b.md, v2.md, v2_replication.md
    '192': 'histories per replicate', '384': 'gate / per-task histories', '768': 'confirmation histories',
    '8448': 'confirmation records', '4224': 'gate records', '2000': 'bootstrap draws', '0.95': 'threshold',
    '0.90': 'threshold', '0.05': 'threshold', '0.02': 'scorer tolerance', '97.5': 'interval level',
    '95': 'interval level', '16': 'max new tokens', '20': 'top-20 causality check', '24': 'smoke records',
    '64': 'histories per stratum', '128': 'histories per cell', '82003': 'confirmation seed', '80': 'GPU memory GB',
    '2026': 'year', '10': 'exponent', '5': 'folds', '1': 'C', '3': 'misc',
    # docs/stale_decision_v1b.md and RESEARCH_LOG (numerical causality, v1b gate failure)
    '0.843': 'int8 causality', '0.875': 'bf16 causality', '0.502': 'bf16+fp32 head', '3.406': 'int8 KV scorer',
    '5.08': 'fp32 KV scorer (x1e-5)', '0.851': 'v1b gate direct min', '0.946': 'v1b gate direct max',
    '260': 'v1b Answer: echoes', '100': 'v1b label answers', '64.': '', '6.27': 'confirmation GPU-h',
    '2.67': 's per record', '24.': '', '0.984': 'Gemma dev historical retrieval (v2m report)',
    # derived in prose from derived.json values (ranges written as lo--hi are traced endpoint by endpoint)
    '3.1': 'routing D_superseded 3.133 rounded in prose', '1.2': 'routing Delta_semantic 1.224 rounded',
    '0.98': 'retrieval >= 0.98 (routing cells 0.982-0.990)', '0.99': 'routing retrieval rounded',
    '3': 'misc', '4': 'misc', '2.5': 'unattached D range 2.496', '3.0': 'unattached D range 3.043',
    '9': 'nine of ten', '12': '12 strata',
}


def values(x, out, path=''):
    """Flat list of numbers; with path_out, also (path, value) pairs."""
    if isinstance(x, dict):
        for k, v in x.items():
            values(v, out, f'{path}/{k}')
    elif isinstance(x, list):
        for i, v in enumerate(x):
            values(v, out, f'{path}[{i}]')
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out.append(float(x))
        PATHS.append((path, float(x)))
    return out


PATHS = []


def matches(lit):
    s = lit.lstrip('+-').replace('{,}', '').replace(',', '')
    decimals = len(s.split('.')[1]) if '.' in s else 0
    return [p for p, v in PATHS if round(abs(v), decimals) == float(s)]


def traced(lit, pool):
    s = lit.lstrip('+-').replace('{,}', '').replace(',', '')
    if s in ALLOW or s.rstrip('0').rstrip('.') in ALLOW:
        return True
    decimals = len(s.split('.')[1]) if '.' in s else 0
    target = float(s)
    return any(round(abs(v), decimals) == target for v in pool)


def main():
    derived = json.loads((HERE / 'derived.json').read_text())
    pool = values(derived, [])
    pool += [p * 100 for p in pool if 0 <= p <= 1]  # percentages
    untraced, report = [], []
    for tex in sorted(SECTIONS.glob('*.tex')):
        text = re.sub(r'\\(label|ref|cite[pt]?|citealp|includegraphics)\{[^}]*\}', '', tex.read_text())
        text = re.sub(r'\\lit\{[^}]*\}', '', text)
        for n, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith('%'):
                continue
            for lit in re.findall(r'(?<![\w.])[+-]?\d+(?:\{,\}\d+)?(?:\.\d+)?', line):
                s = lit.lstrip('+-').replace('{,}', '')
                if '.' not in s and float(s) < 13:
                    continue
                if not traced(lit, pool):
                    untraced.append(f'{tex.name}:{n}: {lit}')
                key = lit.lstrip('+-').replace('{,}', '')
                where = ALLOW.get(key) or ALLOW.get(key.rstrip('0').rstrip('.'))
                hits = matches(lit)
                report.append(f"{tex.name}:{n}: {lit:>10}  " + (f"[record] {where}" if where else
                              f"{len(hits)} match(es): " + '; '.join(hits[:3])))
    (HERE / 'number_trace.txt').write_text('\n'.join(report) + '\n')
    print('\n'.join(untraced) if untraced else 'all numeric literals traced')
    print(f'{len(untraced)} untraced')


if __name__ == '__main__':
    main()
