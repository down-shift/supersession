#!/usr/bin/env python3
"""Generate explicit paired behavioral controls/status inputs or legacy chains."""
import argparse
import hashlib
import json
from pathlib import Path

from src.data.io import write_jsonl, sha256_file
from src.data.supersession import make_version_chain, render_version_chain
from src.data.supersession_behavior import generate_behavior_pairs, generate_status_2x2_gate, SCHEMA
from src.utils import provenance, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('controls', 'status', 'controls_counterbalanced', 'status_2x2', 'status_2x2_gate', 'chains'), required=True)
    p.add_argument('--values', required=True, help='JSON array of candidate strings')
    p.add_argument('--token-ids', help='restrict controls/status proposals to this validated token map')
    p.add_argument('--output', required=True)
    p.add_argument('--n', type=int, help='number of histories (defaults: 96, or fixed 24 for status_2x2_gate)')
    p.add_argument('--seed', type=int)
    p.add_argument('--frozen-gate', help='passing frozen competence artifact required for new status_2x2 confirmatory data')
    p.add_argument('--depths', default='0,1,2,4,8')
    a = p.parse_args()
    if a.kind == 'status_2x2':
        if not a.frozen_gate or not a.token_ids:
            raise ValueError('status_2x2 generation requires --frozen-gate and --token-ids; use generate_status_prompt_stage.py for redevelopment')
        from scripts.generate_status_prompt_stage import generate_stage
        generate_stage('confirmatory', a.values, a.token_ids, a.output, a.n, a.seed,
                       frozen_gate_path=a.frozen_gate)
        return
    seed = a.seed if a.seed is not None else (20261001 if a.kind == 'status_2x2_gate' else 73021)
    n = a.n if a.n is not None else (24 if a.kind == 'status_2x2_gate' else 96)
    if a.kind == 'status_2x2_gate' and n != 24:
        raise ValueError('status_2x2_gate requires --n 24')
    out = Path(a.output)
    if out.exists() or Path(str(out) + '.provenance.json').exists():
        raise FileExistsError(f'{out} exists; choose a fresh output path')
    proposals = json.loads(Path(a.values).read_text())
    values = proposals
    if a.token_ids:
        if a.kind == 'chains':
            raise ValueError('--token-ids is supported only for controls/status generation')
        token_map = json.loads(Path(a.token_ids).read_text())['token_ids']
        values = [v for v in proposals if v in token_map]
        print(f'using {len(values)} validated proposal values; excluded {len(proposals) - len(values)}')
    if a.kind == 'status_2x2_gate':
        rows = generate_status_2x2_gate(n, values, seed)
    elif a.kind in ('controls', 'status', 'controls_counterbalanced', 'status_2x2'):
        rows = generate_behavior_pairs(a.kind, n, values, seed)
    else:
        rows = []
        for i in range(n):
            for depth in map(int, a.depths.split(',')):
                h = make_version_chain(f'chain{i:05d}:d{depth}', values, depth, seed+i*19+depth)
                h['rendered_prompt_x'] = render_version_chain(h, 'x')
                h['rendered_prompt_z'] = render_version_chain(h, 'z')
                rows.append(h)
    write_jsonl(rows, out)
    save_json({**provenance({'seed': seed}, out), 'schema': SCHEMA if a.kind != 'chains' else 'legacy_chains',
               'kind': a.kind, 'n_matched_histories': n, 'n_records': len(rows),
               'values_sha256': sha256_file(a.values), 'token_map_sha256': sha256_file(a.token_ids) if a.token_ids else None,
               'selected_values': values, 'excluded_proposals': [v for v in proposals if v not in values],
               'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()+Path('src/data/supersession_behavior.py').read_bytes()).hexdigest(),
               'update_block_order_balance': ({**__import__('src.data.supersession_behavior',fromlist=['audit_status_2x2_update_order']).audit_status_2x2_update_order(rows)} if a.kind in ('status_2x2','status_2x2_gate') else None)}, str(out) + '.provenance.json')
    print(f'wrote {len(rows)} records for {n} histories to {out}')


if __name__ == '__main__':
    main()
