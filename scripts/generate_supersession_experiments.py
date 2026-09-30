#!/usr/bin/env python3
"""Generate explicit paired behavioral controls/status inputs or legacy chains."""
import argparse
import hashlib
import json
from pathlib import Path

from src.data.io import write_jsonl, sha256_file
from src.data.supersession import make_version_chain, render_version_chain
from src.data.supersession_behavior import generate_behavior_pairs, SCHEMA
from src.utils import provenance, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=('controls', 'status', 'chains'), required=True)
    p.add_argument('--values', required=True, help='JSON array of candidate strings')
    p.add_argument('--token-ids', help='restrict controls/status proposals to this validated token map')
    p.add_argument('--output', required=True)
    p.add_argument('--n', type=int, default=96, help='number of matched histories')
    p.add_argument('--seed', type=int, default=73021)
    p.add_argument('--depths', default='0,1,2,4,8')
    a = p.parse_args()
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
    if a.kind in ('controls', 'status'):
        rows = generate_behavior_pairs(a.kind, a.n, values, a.seed)
    else:
        rows = []
        for i in range(a.n):
            for depth in map(int, a.depths.split(',')):
                h = make_version_chain(f'chain{i:05d}:d{depth}', values, depth, a.seed+i*19+depth)
                h['rendered_prompt_x'] = render_version_chain(h, 'x')
                h['rendered_prompt_z'] = render_version_chain(h, 'z')
                rows.append(h)
    write_jsonl(rows, out)
    save_json({**provenance({'seed': a.seed}, out), 'schema': SCHEMA if a.kind != 'chains' else 'legacy_chains',
               'kind': a.kind, 'n_matched_histories': a.n, 'n_records': len(rows),
               'values_sha256': sha256_file(a.values), 'token_map_sha256': sha256_file(a.token_ids) if a.token_ids else None,
               'selected_values': values, 'excluded_proposals': [v for v in proposals if v not in values],
               'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()+Path('src/data/supersession_behavior.py').read_bytes()).hexdigest()}, str(out) + '.provenance.json')
    print(f'wrote {len(rows)} records for {a.n} matched histories to {out}')


if __name__ == '__main__':
    main()
