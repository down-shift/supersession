#!/usr/bin/env python3
"""Generate fresh pilot/full chains; full depths require a recomputed pilot audit."""
import argparse
import json
from pathlib import Path
from src.data.io import read_jsonl, write_jsonl, sha256_file
from src.data.version_chain import generate, signature, template_hash, DEPTHS
from src.analysis.version_chain import competence
from src.utils import save_json
from src.utils import load_config


def verified_pilot(path, config, tokens):
    doc = json.loads(Path(path).read_text())
    for key in ('dataset','scores'):
        if sha256_file(doc[key+'_path']) != doc[key+'_sha256']:
            raise ValueError('pilot artifact input hash mismatch')
    ds, ss = read_jsonl(doc['dataset_path']), read_jsonl(doc['scores_path'])
    if ds[0]['stage'] != 'pilot' or doc['competence'] != competence(ds,ss):
        raise ValueError('pilot competence does not match recomputation')
    if (doc['template_sha256'] != template_hash() or doc['config_sha256'] != sha256_file(config)
            or doc['token_map_sha256'] != sha256_file(tokens)):
        raise ValueError('pilot config/template/token map differs')
    prov = json.loads(Path(doc['scores_path']+'.provenance.json').read_text())
    if sha256_file(doc['scores_path']+'.provenance.json') != doc['scores_provenance_sha256']:
        raise ValueError('pilot score provenance changed')
    if any(prov[k] != doc[k] for k in ('template_sha256','config_sha256','token_map_sha256','dataset_sha256')):
        raise ValueError('pilot score provenance differs')
    if not prov['alignment_audit']['exact_one_input_token_edit']:
        raise ValueError('pilot lacks the paired input audit')
    td = json.loads(Path(tokens).read_text())
    cfg = load_config(config)['model']
    for k in ('model_revision','tokenizer_revision'):
        if prov[k] != doc[k] or prov[k] != td[k] or prov[k] != (cfg.get(k) or cfg['revision']):
            raise ValueError('pilot exact model/tokenizer revision differs')
    return doc, ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('pilot','full'), required=True)
    p.add_argument('--config', default='configs/four_query_288.yaml')
    p.add_argument('--token-ids', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--depths', default='1,2,4,8')
    p.add_argument('--n', type=int, help='histories per depth; pilot 8, full 96 by default')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--pilot-audit')
    p.add_argument('--exclude-dataset', action='append', default=[])
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() or Path(str(out)+'.provenance.json').exists():
        raise FileExistsError('preserve existing datasets; choose a fresh output')
    token_doc = json.loads(Path(a.token_ids).read_text())
    if token_doc.get('config_sha256') != sha256_file(a.config) or token_doc.get('template_sha256') != template_hash():
        raise ValueError('use a chain token map validated for this configuration/template')
    depths = [int(d) for d in a.depths.split(',')]
    if len(set(depths)) != len(depths) or any(d not in DEPTHS for d in depths):
        raise ValueError('depths must be distinct members of 1,2,4,8')
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
    paths = list(dict.fromkeys(a.exclude_dataset + [str(p) for p in out.parent.glob('version_chain_*.jsonl')]))
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
    save_json({'schema':rows[0]['schema'],'stage':a.stage,'seed':a.seed,'depths':depths,
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
