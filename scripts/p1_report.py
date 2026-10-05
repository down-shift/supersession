#!/usr/bin/env python3
"""Collect P1/P2 results from the sealed analyses into one JSON for the paper (no recomputation).

Every number is copied from an analysis file written by scripts/run_followups.py or
scripts/score_exp1_model.py; the output records the SHA-256 of each source.

    PYTHONPATH=. python scripts/p1_report.py --root outputs --output paper/figures/p1_estimates.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.io import sha256_file

MODELS = {'qwen': 'Qwen3-8B', 'gemma': 'Gemma 3 4B'}


def pick(summary, keys):
    return {k: {'mean': summary[k]['mean'], 'ci95': summary[k]['ci95_history_bootstrap'],
                'n_histories': summary[k]['n_histories']} for k in keys if k in summary}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root) / 'followups'
    out = {'sources': {}, 'distance': {}, 'marker96': {}, 'chain': {}, 'p2': {}}

    def load(path):
        out['sources'][str(path)] = sha256_file(path)
        return json.loads(path.read_text())

    for key, name in MODELS.items():
        f = root / 'distance_v1' / f'{key}_distance_analysis.json'
        if f.exists():
            s = load(f)['summary']
            cells = [f'{c}_{d}_{o}_R' for c in ('superseded', 'entity_mention') for d in ('near', 'far')
                     for o in ('all', 'aligned', 'reversed')]
            effects = [f'{c}_distance_effect_{o}' for c in ('superseded', 'entity_mention') for o in ('all', 'aligned', 'reversed')]
            gaps = [f'construction_gap_{d}_{o}' for d in ('near', 'far') for o in ('all', 'aligned', 'reversed')]
            inter = [f'distance_by_construction_interaction_{o}' for o in ('all', 'aligned', 'reversed')]
            out['distance'][name] = pick(s, cells + effects + gaps + inter)
        f = root / 'marker96' / f'{key}_marker_analysis.json'
        if f.exists():
            s = load(f)['summary']
            keys = [f'{c}_m{m}_all_R' for c in ('superseded', 'entity_mention') for m in (0, 1)]
            keys += [f'{c}_marker_effect_{o}' for c in ('superseded', 'entity_mention') for o in ('all', 'aligned', 'reversed')]
            keys += [f'marker_by_construction_interaction_{o}' for o in ('all', 'aligned', 'reversed')]
            keys += [f'{c}_m{m}_all_R_{part}_component' for c in ('superseded', 'entity_mention') for m in (0, 1)
                     for part in ('replacement', 'source')]
            out['marker96'][name] = pick(s, keys)
    chain_dir = root / 'chain_v1'
    for f in sorted(chain_dir.glob('*_analysis.json')):
        d = load(f)
        model = d['inference_provenance']['fingerprint']['model_id']
        out['chain'][f.stem] = {'model': model, 'stage': d['stage'], 'depth': d['depth'],
                                **pick(d['summary'], list(d['summary']))}
    selection = chain_dir / 'depth_selection.json'
    if selection.exists():
        out['chain']['depth_selection'] = load(selection)
    for f in sorted((Path(a.root) / 'p2').glob('*/gate_report.json')):
        gate = load(f)['evaluation']
        out['p2'].setdefault(f.parent.name, {})['screen'] = {
            'pass': gate['pass'], 'failed_conditions': gate['failed_conditions'],
            'accuracy': {c: m['semantic_accuracy'] for c, m in gate['condition_summary'].items()}}
    for f in sorted((Path(a.root) / 'p2').glob('*/confirmatory_analysis.json')):
        d = load(f)
        res = d['sequence_mass']['results']
        keys = [k for k in res if k.startswith('R_') and k.count('_h') == 0]
        out['p2'].setdefault(f.parent.name, {})['confirmation'] = {
            'descriptive_only': d.get('descriptive_only', False),
            **{k: {'mean': res[k]['mean'], 'ci95': res[k]['ci95_cluster_bootstrap']} for k in keys}}
    Path(a.output).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: list(v) for k, v in out.items() if k != 'sources'}, indent=1))


if __name__ == '__main__':
    main()
