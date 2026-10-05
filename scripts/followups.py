#!/usr/bin/env python3
"""Generate and audit the separate relational follow-up pilot datasets."""
import argparse
import json
from pathlib import Path

from src.cross_model.update_control import UPDATE_VERSION, generate_update
from src.cross_model.followups import (VERSION, DISTANCE_VERSION, generate_distance, generate_harder,
                                       generate_marker, validate_marker_templates)
from src.cross_model.protocol import digest
from src.cross_model.protocol import read_sealed
from src.data.io import write_jsonl, sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment', choices=('marker', 'harder', 'distance', 'update'))
    p.add_argument('--superseded-near', help='distance: the near superseded template chosen by the audit')
    p.add_argument('--stage', default='pilot', choices=('pilot', 'confirmatory', 'development', 'test'))
    p.add_argument('--histories', type=int, default=2)
    p.add_argument('--distractors', type=int, default=2)
    p.add_argument('--freeze')
    p.add_argument('--prior-dataset', action='append', default=[])
    p.add_argument('--seed', type=int, default=20261004)
    p.add_argument('--output', required=True)
    p.add_argument('--report')
    a = p.parse_args()
    valid_stages = {'marker': ('pilot', 'confirmatory'), 'harder': ('pilot', 'development', 'test'),
                    'distance': ('pilot', 'confirmatory'), 'update': ('pilot', 'confirmatory')}
    if a.stage not in valid_stages[a.experiment]:
        p.error(f'{a.experiment} does not define stage {a.stage!r}')
    exclusions = set()
    prior_details = []
    for prior in a.prior_dataset:
        prior_rows = __import__('src.data.io', fromlist=['read_jsonl']).read_jsonl(prior)
        prior_stages = {r['stage'] for r in prior_rows}
        if not prior_rows or len(prior_stages) != 1:
            raise ValueError(f'prior dataset must contain exactly one stage: {prior}')
        prior_stage = next(iter(prior_stages))
        expected_prior = {'marker': {'confirmatory': {'pilot', 'confirmatory'}, 'pilot': set()},
            'distance': {'confirmatory': {'pilot', 'confirmatory'}, 'pilot': set()},
            'update': {'confirmatory': {'pilot', 'confirmatory'}, 'pilot': set()},
            'harder': {'development': {'pilot'}, 'test': {'pilot', 'development'}, 'pilot': set()}}
        if prior_stage not in expected_prior[a.experiment][a.stage]:
            raise ValueError(f'{a.experiment}/{a.stage} cannot exclude prior {prior_stage} data')
        prior_details.append({'path': str(Path(prior).resolve()), 'sha256': sha256_file(prior), 'stage': prior_stage})
        exclusions.update(r['history_signature'] for r in prior_rows)
        exclusions.update(r['target_history_signature'] for r in prior_rows)
    if a.experiment == 'update':
        template_audit = {'status': 'not applicable'}
        rows = generate_update(a.stage, a.histories, a.seed,
                               excluded_signatures=exclusions if a.prior_dataset else None)
    elif a.experiment == 'distance':
        if not a.superseded_near:
            raise ValueError('distance generation requires --superseded-near from the tokenizer audit')
        template_audit = {'superseded_near': a.superseded_near, 'status': 'chosen by scripts/distance_audit.py'}
        rows = generate_distance(a.stage, a.histories, a.seed, a.superseded_near,
                                 excluded_signatures=exclusions if a.prior_dataset else None)
    elif a.experiment == 'marker':
        template_audit = validate_marker_templates()
        if a.stage == 'confirmatory' and not a.prior_dataset:
            raise ValueError('confirmatory generation requires --prior-dataset pilot.jsonl')
        rows = generate_marker(a.stage, a.histories, a.seed,
                               excluded_signatures=exclusions if a.prior_dataset else None)
    else:
        if a.stage == 'test':
            if not a.freeze:
                raise ValueError('test generation requires --freeze from select-hard-difficulty')
            frozen = read_sealed(a.freeze)
            from scripts.run_followups import _code_hash, validate_test_freeze
            validate_test_freeze(frozen)
            if frozen.get('prompt_code_sha256') != _code_hash():
                raise ValueError('test protocol freeze code hash differs from current code')
            if a.distractors != frozen.get('selected_n_distractors'):
                raise ValueError('test distractor count differs from frozen development selection')
            if a.seed != frozen.get('test_dataset_seed'):
                raise ValueError('test seed differs from frozen development protocol')
            if a.histories != frozen.get('test_histories'):
                raise ValueError('test history count differs from frozen development protocol')
            exclusions.update(frozen.get('excluded_history_signatures', []))
            if not exclusions:
                raise ValueError('test freeze has no development history exclusions')
        if a.stage == 'development' and not a.prior_dataset:
            raise ValueError('development generation requires --prior-dataset pilot.jsonl')
        if a.stage == 'test' and not a.prior_dataset:
            raise ValueError('test generation requires development --prior-dataset files')
        if a.stage == 'test' and not {'pilot', 'development'} <= {d['stage'] for d in prior_details}:
            raise ValueError('test generation requires both pilot and development history exclusions')
        template_audit = validate_marker_templates()
        rows = generate_harder(a.stage, a.histories, a.distractors, a.seed,
                               excluded_signatures=exclusions if a.prior_dataset else None)
    if Path(a.output).exists():
        raise FileExistsError(f'refusing to overwrite follow-up dataset: {a.output}')
    if a.report and Path(a.report).exists():
        raise FileExistsError(f'refusing to overwrite follow-up report: {a.report}')
    write_jsonl(rows, a.output)
    unique = {r['prompt'] for r in rows}
    report = {'protocol': {'distance': DISTANCE_VERSION, 'update': UPDATE_VERSION}.get(a.experiment, VERSION),
        'experiment': a.experiment, 'stage': a.stage,
        'n_histories': a.histories, 'n_members': len(rows), 'n_unique_prompts': len(unique),
        'dataset_seeds': sorted({r['seed'] for r in rows}),
        'history_signatures': sorted({r['history_signature'] for r in rows}),
        'target_history_signatures': sorted({r['target_history_signature'] for r in rows}),
        'exclusions': sorted(exclusions), 'prior_datasets': prior_details,
        'freeze_sha256': sha256_file(a.freeze) if a.freeze else None,
        'dataset_bytes': Path(a.output).stat().st_size, 'dataset_sha256': sha256_file(a.output),
        'code_sha256': digest({str(p): sha256_file(p) for p in
                         (Path('src/cross_model/followups.py'), Path('scripts/followups.py'))}),
        'template_audit': template_audit,
        'generation_status': {'pilot': 'dataset generation/storage only; no model inference',
            'development': 'development histories generated; model evaluation and difficulty selection pending',
            'test': 'frozen test histories generated; model evaluation pending',
            'confirmatory': 'marker confirmatory histories generated; model evaluation pending'}[a.stage],
        'failed_gates': [],
        'runtime_estimate_method': 'infer from measured per-prompt inference on the target runtime before confirmatory scoring'}
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
