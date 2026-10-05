#!/usr/bin/env python3
"""History-bootstrap summaries of derived contrasts quoted in the paper but not stored in the sealed
analyses (2,000 draws, seed 73021, the same procedure as the sealed reports). Reads only the CSVs in
this directory; writes derived_estimates.json.

    uv run --no-project --with numpy python paper/data/derived_estimates.py
"""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QUOTED = {
    'exp2_marker96_history_estimates.csv': [
        'entity_minus_superseded_marker0_all', 'entity_minus_superseded_marker1_all',
        'entity_minus_superseded_marker0_aligned', 'entity_minus_superseded_marker0_reversed',
        'R_superseded_marker0_reversed', 'R_entity_mention_marker0_reversed',
        'marker_interaction_all'],  # the last reproduces the sealed analysis (check)
}


def summary(values):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(73021)
    means = x[rng.integers(0, len(x), size=(2000, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {'mean': float(x.mean()), 'ci95': [float(lo), float(hi)], 'n_histories': len(x)}


def independent_difference(a, b):
    """Mean of a minus mean of b for independent history sets, each resampled separately."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(73021)
    da = a[rng.integers(0, len(a), size=(2000, len(a)))].mean(axis=1)
    db = b[rng.integers(0, len(b), size=(2000, len(b)))].mean(axis=1)
    lo, hi = np.quantile(da - db, [0.025, 0.975])
    return {'mean': float(a.mean() - b.mean()), 'ci95': [float(lo), float(hi)], 'n_histories': [len(a), len(b)]}


def chain_depth_contrasts():
    path = HERE / 'chain_history_estimates.csv'
    rows = list(csv.DictReader(path.open()))
    out = {}
    for model in sorted({r['model'] for r in rows}):
        by = {d: [float(r['R']) for r in rows if r['model'] == model and r['depth'] == str(d)] for d in (3, 4, 5)}
        out[model] = {f'R_depth{d}': summary(by[d]) for d in (3, 4, 5)}
        out[model]['R_depth5_minus_depth3'] = independent_difference(by[5], by[3])
        out[model]['R_depth4_minus_depth3'] = independent_difference(by[4], by[3])
        out[model]['R_depth5_minus_depth4'] = independent_difference(by[5], by[4])
    return hashlib.sha256(path.read_bytes()).hexdigest(), out


def main():
    out = {'bootstrap': 'history; 2,000 draws; seed 73021', 'sources': {}, 'estimates': {}}
    for name, keys in QUOTED.items():
        path = HERE / name
        out['sources'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = list(csv.DictReader(path.open()))
        for model in sorted({r['model'] for r in rows}):
            out['estimates'].setdefault(name, {})[model] = {
                k: summary([float(r[k]) for r in rows if r['model'] == model]) for k in keys}
    out['sources']['chain_history_estimates.csv'], out['estimates']['chain_depth'] = chain_depth_contrasts()
    (HERE / 'derived_estimates.json').write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
