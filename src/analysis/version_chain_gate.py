"""Shared verification of immutable version-chain pilot artifacts."""
import json
from pathlib import Path
from src.data.io import read_jsonl, sha256_file
from src.data.version_chain import template_hash
from src.analysis.version_chain import competence
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
