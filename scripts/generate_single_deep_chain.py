#!/usr/bin/env python3
"""Generate fresh single-deep-chain histories with competence-qualified full depths."""
import argparse
import json
from pathlib import Path
from src.data.io import read_jsonl, write_jsonl, sha256_file
from src.data.single_deep_chain import generate, signature, template_hash, DEPTHS
from src.analysis.single_deep_chain_gate import verified_pilot
from src.utils import save_json



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('pilot','full'), required=True)
    p.add_argument('--config', default='configs/four_query_288.yaml')
    p.add_argument('--token-ids', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--depths', default='2,3,4')
    p.add_argument('--n', type=int, help='histories per depth; pilot 8, full 96 by default')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--pilot-audit')
    p.add_argument('--exclude-dataset', action='append', default=[])
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() or Path(str(out)+'.provenance.json').exists():
        raise FileExistsError('preserve existing datasets; choose a fresh output')
    token_doc = json.loads(Path(a.token_ids).read_text())
    if token_doc.get('config_sha256') != sha256_file(a.config):
        raise ValueError('use a validated candidate token map for this exact configuration')
    depths = [int(d) for d in a.depths.split(',')]
    if len(set(depths)) != len(depths) or any(d not in DEPTHS for d in depths):
        raise ValueError('depths must be distinct members of 2,3,4')
    excluded = []
    if a.stage == 'full':
        if not a.pilot_audit:
            raise ValueError('full generation requires --pilot-audit')
        doc, pilot = verified_pilot(a.pilot_audit,a.config,a.token_ids)
        if a.seed == pilot[0]['seed']:
            raise ValueError('full generation needs a fresh seed')
        depths = [d for d in depths if d in doc['competence']['qualified_depths']]
        if not depths:
            raise ValueError('pilot qualified no depths; full generation blocked')
        excluded.extend(pilot)
    paths = list(dict.fromkeys(a.exclude_dataset + [str(p) for p in out.parent.glob('single_deep_chain_*.jsonl')]))
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(f'excluded dataset does not exist: {path}')
        excluded.extend(read_jsonl(path))
    sigs = {signature(r) for r in excluded}
    if any(r['seed'] == a.seed for r in excluded):
        raise ValueError('this seed was already used by an excluded chain run')
    n = a.n if a.n is not None else (8 if a.stage=='pilot' else 96)
    rows = generate(a.stage,n,depths,
                    list(token_doc['token_ids']),a.seed,sigs)
    write_jsonl(rows,out)
    save_json({'experiment_kind':'single_deep_chain','schema':rows[0]['schema'],'stage':a.stage,'seed':a.seed,'depths':depths,
        'n_histories_per_depth':n,
        'dataset_sha256':sha256_file(out),'config_sha256':sha256_file(a.config),
        'token_map_sha256':sha256_file(a.token_ids),'template_sha256':template_hash(),
        'excluded_history_count':len(sigs),'excluded_datasets':[{ 'path':path,'sha256':sha256_file(path)} for path in paths],
        'pilot_audit_path':str(Path(a.pilot_audit).resolve()) if a.pilot_audit else None,
        'pilot_audit_sha256':sha256_file(a.pilot_audit) if a.pilot_audit else None,
        'generator_sha256':sha256_file(__file__)},str(out)+'.provenance.json')
    print(f'wrote {len(rows)} members at depths {depths}; histories are fresh')


if __name__ == '__main__':
    main()
