#!/usr/bin/env python3
"""Audited raw matched effects and paired history-level controls/status summaries."""
import argparse
import json
import hashlib
from pathlib import Path

import pandas as pd
from src.data.io import read_jsonl, sha256_file
from src.analysis.supersession_behavior import (audited_effects, history_contrasts, summarize_histories,
                                               competence, DEFINITIONS)
from src.utils import provenance, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--behavior', required=True)
    p.add_argument('--kind', choices=('controls', 'status'), required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seed', type=int, default=73021)
    p.add_argument('--bootstrap-draws', type=int, default=2000)
    p.add_argument('--competence-threshold', type=float, default=.99, help='diagnostic flag only; never filters primary trials')
    a = p.parse_args()
    if a.bootstrap_draws < 1 or not 0 <= a.competence_threshold <= 1:
        raise ValueError('invalid bootstrap count or diagnostic competence threshold')
    out = Path(a.output_dir)
    if any((out/name).exists() for name in ('matched_edit_effects.csv', 'history_relevance.csv', 'competence.csv', 'summary.json')):
        raise FileExistsError('analysis outputs exist; choose a fresh output directory')
    dataset, scored = read_jsonl(a.dataset), read_jsonl(a.behavior)
    manifest = Path(a.behavior+'.run.json')
    if manifest.exists() and json.loads(manifest.read_text())['dataset_sha256'] != sha256_file(a.dataset):
        raise ValueError('scoring manifest dataset hash differs from analysis dataset')
    effects = audited_effects(dataset, scored, a.kind)
    histories = history_contrasts(effects, a.kind)
    comp = competence(scored, effects, a.competence_threshold)
    results = {key: summarize_histories([h[key] for h in histories], a.seed, a.bootstrap_draws)
               for key in histories[0] if key != 'history_id'}
    scoring_provenance = Path(a.behavior+'.provenance.json')
    summary = {'kind': a.kind, 'primary_filter': 'all_valid_trials', 'all_contrasts_history_paired': True,
               'n_histories': len(histories), 'n_pairs': len(effects), 'n_scored_members': len(scored),
               'definitions': DEFINITIONS, 'bootstrap_unit': 'history_id', 'bootstrap_seed': a.seed,
               'bootstrap_draws': a.bootstrap_draws, 'results': results, 'competence': comp,
               'low_competence_conditions': sorted({r['condition'] for r in comp if r['low_competence']}),
               'competence_weighting': 'all scored pair members; identical baselines repeated across independent edits',
               'provenance': {**provenance({'seed': a.seed}, a.dataset), 'behavior_sha256': sha256_file(a.behavior),
                              'analysis_code_sha256': hashlib.sha256(Path(__file__).read_bytes()+Path('src/analysis/supersession_behavior.py').read_bytes()+Path('src/analysis/metrics.py').read_bytes()).hexdigest(),
                              'scoring_provenance': json.loads(scoring_provenance.read_text()) if scoring_provenance.exists() else None}}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(effects).to_csv(out/'matched_edit_effects.csv', index=False)
    pd.DataFrame(histories).to_csv(out/'history_relevance.csv', index=False)
    pd.DataFrame(comp).to_csv(out/'competence.csv', index=False)
    save_json(summary, out/'summary.json')
    print(f'wrote {len(effects)} raw effects and {len(histories)} paired history contrasts to {out}')
    if summary['low_competence_conditions']:
        print('Low full-vocabulary task competence: '+', '.join(summary['low_competence_conditions'])+'; primary analysis retained every valid pair.')


if __name__ == '__main__':
    main()
