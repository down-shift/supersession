#!/usr/bin/env python3
"""Generate and audit the separate relational follow-up pilot datasets."""
import argparse
import json
from pathlib import Path

from src.cross_model.followups import (VERSION, generate_harder, generate_marker,
                                       validate_marker_templates)
from src.cross_model.protocol import digest
from src.data.io import write_jsonl, sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('experiment', choices=('marker', 'harder'))
    p.add_argument('--stage', default='pilot', choices=('pilot', 'development', 'test'))
    p.add_argument('--histories', type=int, default=2)
    p.add_argument('--distractors', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261004)
    p.add_argument('--output', required=True)
    p.add_argument('--report')
    a = p.parse_args()
    if a.experiment == 'marker':
        template_audit = validate_marker_templates()
        rows = generate_marker(a.stage if a.stage == 'pilot' else 'confirmatory', a.histories, a.seed)
    else:
        template_audit = validate_marker_templates()
        rows = generate_harder(a.stage if a.stage in ('development', 'test') else 'development',
                               a.histories, a.distractors, a.seed)
    write_jsonl(rows, a.output)
    unique = {r['prompt'] for r in rows}
    report = {'protocol': VERSION, 'experiment': a.experiment, 'stage': a.stage,
        'n_histories': a.histories, 'n_members': len(rows), 'n_unique_prompts': len(unique),
        'dataset_bytes': Path(a.output).stat().st_size, 'dataset_sha256': sha256_file(a.output),
        'code_sha256': digest({str(p): sha256_file(p) for p in
                         (Path('src/cross_model/followups.py'), Path('scripts/followups.py'))}),
        'template_audit': template_audit,
        'pilot_status': 'design/storage pilot only; no model inference or outcome-based selection',
        'runtime_estimate_method': 'infer from measured per-prompt inference on the target runtime before confirmatory scoring'}
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
