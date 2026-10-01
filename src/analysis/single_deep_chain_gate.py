"""Recompute single-chain depth eligibility from immutable pilot inputs."""
import json
from pathlib import Path
from src.data.io import read_jsonl,sha256_file
from src.data.single_deep_chain import template_hash
from src.analysis.single_deep_chain import competence
from src.utils import load_config


def verified_pilot(path,config,tokens):
    doc = json.loads(Path(path).read_text())
    if doc.get('experiment_kind')!='single_deep_chain' or doc.get('stage')!='pilot':
        raise ValueError('requires a single-chain pilot artifact')
    for key in ('dataset','scores'):
        if sha256_file(doc[key+'_path'])!=doc[key+'_sha256']:
            raise ValueError('pilot input hash mismatch')
    ds,ss = read_jsonl(doc['dataset_path']),read_jsonl(doc['scores_path'])
    if ds[0]['stage']!='pilot' or doc['competence']!=competence(ds,ss):
        raise ValueError('pilot competence does not match recomputation')
    bindings = {'config_sha256':sha256_file(config),'token_map_sha256':sha256_file(tokens),'template_sha256':template_hash()}
    if any(doc[k]!=v for k,v in bindings.items()):
        raise ValueError('pilot config/token map/template differs')
    prov_path = doc['scores_path']+'.provenance.json'
    prov = json.loads(Path(prov_path).read_text())
    if sha256_file(prov_path)!=doc['scores_provenance_sha256'] or any(prov[k]!=doc[k] for k in (*bindings,'dataset_sha256')):
        raise ValueError('pilot provenance differs')
    if prov.get('experiment_kind')!='single_deep_chain' or not prov['alignment_audit']['exact_one_input_token_edit']:
        raise ValueError('pilot lacks audited single-chain score provenance')
    td = json.loads(Path(tokens).read_text()); cfg = load_config(config)['model']
    for k in ('model_revision','tokenizer_revision'):
        if prov[k]!=doc[k] or prov[k]!=td[k] or prov[k]!=(cfg.get(k) or cfg['revision']):
            raise ValueError('pilot exact revisions differ')
    return doc,ds
