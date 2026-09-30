#!/usr/bin/env python3
"""Tokenizer-only validation of a larger vocabulary for distinct deep chains."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from transformers import AutoTokenizer
from src.data.io import sha256_file
from src.data.token_validation import continuation_token_id
from src.data.version_chain import render, template_hash
from src.utils import load_config, save_json, provenance


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/four_query_288.yaml')
    p.add_argument('--values', default='configs/supersession_values.json')
    p.add_argument('--extra-values', default='configs/replication_values.json')
    p.add_argument('--output', required=True)
    p.add_argument('--local-files-only', action='store_true')
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError('preserve existing token maps; choose a fresh output')
    c = load_config(a.config)
    m = c['model']
    revision = m.get('tokenizer_revision') or m.get('revision')
    if not all(re.fullmatch(r'[0-9a-f]{40}', str(r)) for r in (revision,m.get('revision'))):
        raise ValueError('pin exact model and tokenizer revisions')
    tok = AutoTokenizer.from_pretrained(m.get('tokenizer_id') or m['id'], revision=revision,
                                      use_fast=True, local_files_only=a.local_files_only)
    values = list(dict.fromkeys(json.loads(Path(a.values).read_text()) + json.loads(Path(a.extra_values).read_text())))
    probe = {'versions':{'x':['amber','denim'], 'z':['coral','elm']},
             'literal_names':{'x':'x','z':'z'}, 'block_order':['x','z'], 'query':'x'}
    text = render(probe, tok, m.get('chat_template', True))
    accepted, rejected = {}, {}
    for value in values:
        try:
            token = continuation_token_id(tok, text, ' '+value)
            assignment = tok('x = '+value, add_special_tokens=False, return_offsets_mapping=True)
            start = len('x = ')
            positions = [i for i,(x,y) in enumerate(assignment['offset_mapping']) if x<len('x = '+value) and y>start]
            if len(positions) != 1 or assignment['input_ids'][positions[0]] != token:
                raise ValueError('not one stable assignment token')
            if token in accepted.values():
                raise ValueError('token ID collision')
            accepted[value] = token
        except ValueError as exc:
            rejected[value] = str(exc)
    if len(accepted) < 19:
        raise ValueError(f'only {len(accepted)} validated values; depth 8 requires 19')
    save_json({'token_ids':accepted,'rejected':rejected,'model_revision':m['revision'],
        'tokenizer_revision':revision,'config_sha256':sha256_file(a.config),
        'values_sha256':sha256_file(a.values),'extra_values_sha256':sha256_file(a.extra_values),
        'template_sha256':template_hash(),'chat_template_sha256':hashlib.sha256(str(tok.chat_template).encode()).hexdigest(),
        'validation_scope':'probe continuations and assignment slots; every actual pair is audited before scoring',
        'preparation_code_sha256':sha256_file(__file__),'provenance':provenance(c)},a.output)
    print(f'validated {len(accepted)} distinct chain candidates; no model weights loaded')


if __name__ == '__main__':
    main()
