#!/usr/bin/env python3
"""Audited, resumable version-chain scoring with raw candidate logits."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from tqdm.auto import tqdm
from src.data.io import read_jsonl, sha256_file
from src.data.progress import prepare_jsonl_progress, append_jsonl_record
from src.data.version_chain import audit, audit_tokens, render, template_hash
from src.experiments.behavior import score_example
from src.models.loader import load_model
from src.utils import load_config, provenance, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/four_query_288.yaml')
    p.add_argument('--dataset', required=True)
    p.add_argument('--token-ids', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    rows = read_jsonl(a.dataset)
    audit(rows)
    td = json.loads(Path(a.token_ids).read_text())
    ids = td['token_ids']
    dp = json.loads(Path(a.dataset+'.provenance.json').read_text())
    bindings = {'dataset_sha256':sha256_file(a.dataset),'config_sha256':sha256_file(a.config),
                'token_map_sha256':sha256_file(a.token_ids),'template_sha256':template_hash()}
    if any(dp.get(k) != v for k,v in bindings.items()):
        raise ValueError('dataset provenance/freshness mismatch')
    if rows[0]['stage'] == 'full':
        from src.analysis.version_chain_gate import verified_pilot
        pilot_path = dp.get('pilot_audit_path')
        if not pilot_path or sha256_file(pilot_path) != dp.get('pilot_audit_sha256'):
            raise ValueError('full dataset passing pilot artifact changed or is missing')
        pilot_doc, pilot_rows = verified_pilot(pilot_path,a.config,a.token_ids)
        if not {r['depth'] for r in rows} <= set(pilot_doc['competence']['qualified_depths']):
            raise ValueError('full dataset includes unqualified depths')
        from src.data.version_chain import signature
        if {signature(r) for r in rows} & {signature(r) for r in pilot_rows}:
            raise ValueError('full dataset reuses pilot histories')
    if not a.resume and any(Path(a.output+s).exists() for s in ('','.run.json','.provenance.json')):
        raise FileExistsError('preserve scores; choose fresh output or --resume')
    c = load_config(a.config)
    model, tok = load_model(c)
    for key in ('model_revision','tokenizer_revision'):
        requested = c['model'].get(key) or c['model'].get('revision')
        if not c.get('resolved_'+key) and re.fullmatch(r'[0-9a-f]{40}',str(requested)):
            c['resolved_'+key] = requested
        if not re.fullmatch(r'[0-9a-f]{40}',str(c.get('resolved_'+key))) or td.get(key) != c['resolved_'+key]:
            raise ValueError('exact model/tokenizer revisions must match validated token map')
    chat = c['model'].get('chat_template',True)
    alignment = audit_tokens(rows,tok,ids,chat)
    chat_hash = hashlib.sha256(str(tok.chat_template).encode()).hexdigest()
    if td.get('chat_template_sha256') != chat_hash:
        raise ValueError('token map chat template differs')
    paths = [__file__,'src/data/version_chain.py','src/experiments/behavior.py',
             'src/analysis/metrics.py','src/analysis/version_chain.py','src/analysis/version_chain_gate.py','src/data/token_validation.py','src/data/progress.py','src/models/loader.py']
    code_hash = hashlib.sha256(b''.join(Path(p).read_bytes() for p in paths)).hexdigest()
    run = provenance(c,a.dataset)
    fp = {**c,'version_chain':{**bindings,'code_sha256':code_hash,
                              'chat_template_sha256':chat_hash,'packages':run['packages'],'python':run['python']}}
    done = prepare_jsonl_progress(a.output,a.dataset,a.token_ids,fp,rows,resume=a.resume)
    side = Path(a.output+'.provenance.json')
    if done and not side.exists():
        raise ValueError('checkpoint score provenance missing')
    if not side.exists():
        save_json({**run,**bindings,'stage':rows[0]['stage'],'code_sha256':code_hash,
            'chat_template_sha256':chat_hash,'alignment_audit':alignment,
            'dataset_path':str(Path(a.dataset).resolve()),'candidate_token_ids':ids},side)
    expected = {r['example_id']:r for r in rows}
    cache = {}
    for s in read_jsonl(a.output) if Path(a.output).exists() else []:
        if any(s.get(k) != v for k,v in expected[s['example_id']].items()) or s['prompt'] != render(expected[s['example_id']],tok,chat):
            raise ValueError('checkpoint prompt/metadata differs')
        cache[s['prompt']] = s
    for r in tqdm(sorted(rows,key=lambda r:(r['pair_id'],r['pair_direction'])),desc='Version-chain scoring'):
        if r['example_id'] in done:
            continue
        prompt = render(r,tok,chat)
        if prompt in cache:
            s = {**r,**{k:v for k,v in cache[prompt].items() if k not in r}}
        else:
            s = score_example(model,tok,r,ids,chat=chat,renderer=render,preserve_metadata=True)
            cache[prompt] = s
        l, chain = s['candidate_logits'],r['versions'][r['query']]
        s.update(current_minus_previous=l[chain[-1]]-l[chain[-2]],
                 current_minus_oldest=l[chain[-1]]-l[chain[0]])
        append_jsonl_record(a.output,s)
    print(f'scored/resumed {len(rows)} members, {len(done)} loaded; no mechanistic interventions')


if __name__ == '__main__':
    main()
