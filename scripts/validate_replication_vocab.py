#!/usr/bin/env python3
"""Validate a disjoint lexical replication pool against exact chat prefixes."""
import argparse, json
from pathlib import Path
from transformers import AutoTokenizer
from src.data.generate import make_histories, expand_history_queries
from src.data.token_validation import validate_candidate_vocabulary, validate_assignment_patching
from src.utils import load_config, provenance, save_json

p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--original-token-ids',required=True); p.add_argument('--values',required=True,help='JSON list of new value strings'); p.add_argument('--variables',required=True,help='JSON list of new two-variable pairs'); p.add_argument('--output',required=True); a=p.parse_args()
c=load_config(a.config); m=c['model']; tok=AutoTokenizer.from_pretrained(m.get('tokenizer_id') or m['id'],revision=m.get('tokenizer_revision') or m.get('revision'))
orig=json.loads(Path(a.original_token_ids).read_text())['token_ids']; values=json.loads(Path(a.values).read_text()); variables=json.loads(Path(a.variables).read_text())
original_vocab=set(orig)|set(c['dataset'].get('values',[]))
overlap=set(values)&original_vocab
if overlap: raise ValueError(f'replication vocabulary overlaps original values: {sorted(overlap)}')
if not variables or any(set(pair)&{v for pair in c['dataset']['variables'] for v in pair} for pair in variables): raise ValueError('replication variable names must be disjoint from original names')
histories=make_histories(48,seed=c['seed']+9187,values=values[:16],variables=variables,partition='confirmatory')
examples=[q for h in histories for q in expand_history_queries(h)]
accepted,rejected=validate_candidate_vocabulary(tok,examples,values,chat=m.get('chat_template',True))
if set(accepted)&set(orig): raise ValueError('validated token map unexpectedly overlaps original values')
audit=validate_assignment_patching(tok,histories,list(accepted),chat=m.get('chat_template',True))
save_json({'values':accepted,'rejected':rejected,'variables':variables,'original_value_overlap':sorted(overlap),'original_token_id_overlap':sorted(set(accepted.values())&set(orig.values())), 'assignment_alignment_audit':audit,'provenance':provenance(c)},a.output)
