#!/usr/bin/env python3
"""Generate chain datasets and apply the recorded depth-selection rule (docs/chain_v1.md).

    python -m scripts.chains generate --stage development --depth 3 --histories 12 --seed 20261107 \
        --vocabulary-audit AUDIT.json --output D.jsonl --report D_report.json [--prior-dataset P.jsonl ...]
    python -m scripts.chains select-depth --analyses A_qwen_d3.json ... --output selection.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.cross_model.chains import CHAIN_VERSION, DEPTHS, chain_vocabulary, generate_chain
from src.cross_model.protocol import digest, read_sealed, write_new
from src.data.io import read_jsonl, sha256_file, write_jsonl

ACCURACY_WINDOW = (0.70, 0.95)


def generate(a):
    audit = json.loads(Path(a.vocabulary_audit).read_text())
    if not audit.get('complete'):
        raise ValueError('vocabulary audit did not choose four extension values')
    vocabulary = chain_vocabulary(audit['extension'])
    exclusions, priors = set(), []
    for prior in a.prior_dataset:
        rows = read_jsonl(prior)
        exclusions.update(r['history_signature'] for r in rows)
        priors.append({'path': str(Path(prior).resolve()), 'sha256': sha256_file(prior)})
    rows = generate_chain(a.stage, a.histories, a.depth, a.seed, vocabulary, exclusions)
    if Path(a.output).exists() or Path(a.report).exists():
        raise FileExistsError('refusing to overwrite chain dataset or report')
    write_jsonl(rows, a.output)
    report = {'protocol': CHAIN_VERSION, 'experiment': 'chain', 'stage': a.stage, 'depth': a.depth,
              'n_histories': a.histories, 'n_members': len(rows), 'n_unique_prompts': len({r['prompt'] for r in rows}),
              'dataset_seeds': sorted({r['seed'] for r in rows}), 'vocabulary': vocabulary,
              'vocabulary_audit_sha256': sha256_file(a.vocabulary_audit),
              'history_signatures': sorted({r['history_signature'] for r in rows}),
              'target_history_signatures': sorted({r['target_history_signature'] for r in rows}),
              'exclusions': sorted(exclusions), 'prior_datasets': priors,
              'dataset_sha256': sha256_file(a.output),
              'code_sha256': digest({p: sha256_file(p) for p in ('src/cross_model/chains.py', 'scripts/chains.py')})}
    write_new(a.report, report)
    print(json.dumps({k: report[k] for k in ('stage', 'depth', 'n_histories', 'n_members', 'n_unique_prompts')}))


def select_depth(a):
    """Shallowest depth at which at least one model's development accuracy lies in the window."""
    table = {}
    for path in a.analyses:
        analysis = read_sealed(path)
        model = analysis['inference_provenance']['fingerprint']['model_id']
        table.setdefault(analysis['depth'], {})[model] = analysis['summary']['correct']['mean']
    if set(table) != set(DEPTHS) or len({len(v) for v in table.values()}) != 1:
        raise ValueError('need development analyses for every depth and the same models at each depth')
    qualifying = [d for d in DEPTHS if any(ACCURACY_WINDOW[0] <= acc <= ACCURACY_WINDOW[1]
                                           for acc in table[d].values())]
    result = {'rule': f'shallowest depth with at least one model accuracy in {list(ACCURACY_WINDOW)}',
              'development_accuracy': table, 'qualifying_depths': qualifying,
              'selected_depth': qualifying[0] if qualifying else None,
              'decision': 'run confirmation' if qualifying else 'stop: no depth produced below-ceiling accuracy',
              'analyses_sha256': {p: sha256_file(p) for p in a.analyses}}
    write_new(a.output, result)
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--stage', choices=('development', 'confirmatory'), required=True)
    g.add_argument('--depth', type=int, choices=DEPTHS, required=True)
    g.add_argument('--histories', type=int, required=True)
    g.add_argument('--seed', type=int, required=True)
    g.add_argument('--vocabulary-audit', required=True)
    g.add_argument('--prior-dataset', action='append', default=[])
    g.add_argument('--output', required=True)
    g.add_argument('--report', required=True)
    s = sub.add_parser('select-depth')
    s.add_argument('--analyses', nargs='+', required=True)
    s.add_argument('--output', required=True)
    a = p.parse_args()
    (generate if a.command == 'generate' else select_depth)(a)


if __name__ == '__main__':
    main()
