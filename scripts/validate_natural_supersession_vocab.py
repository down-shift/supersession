#!/usr/bin/env python3
"""Validate this model's answer tokens and exact natural-prompt edit alignment."""
import argparse, hashlib, json, re
from pathlib import Path
from transformers import AutoTokenizer

from src.data.io import sha256_file
from src.data.supersession_behavior import (generate_behavior_pairs,
                                            render_behavior_example, audit_tokenized_pairs)
from src.data.token_validation import continuation_token_id
from src.utils import load_config, provenance, save_json

def annotate(rows):
    names=('Nora','Liam','Ava','Omar','Mila','Eli','Iris','Noah','Zoe','Theo','Maya','Leo')
    attrs=('badge','color','code','label')
    for row in rows:
        i=row['history_index']
        entities=[names[(2*i)%len(names)],names[(2*i+1)%len(names)]]
        row.update(prompt_family='natural_entity_attribute_v1',prompt_variant='nora_v1',
                   entities=entities,attribute=attrs[i%len(attrs)])
        row['variables']=entities if row.get('orientation',0)==0 else entities[::-1]
    return rows

p=argparse.ArgumentParser()
p.add_argument('--config',required=True); p.add_argument('--values',required=True)
p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=20261022)
p.add_argument('--validation-histories',type=int,default=4)
p.add_argument('--match-token-ids',help='prefer the existing first-model candidate universe where tokenizer validation permits')
a=p.parse_args()
if Path(a.output).exists(): raise FileExistsError('preserve the existing token map; choose a fresh output path')
if a.validation_histories<2: raise ValueError('use at least two validation histories')
c=load_config(a.config); model=c['model']; requested=model.get('tokenizer_revision') or model['revision']
if any(not re.fullmatch(r'[0-9a-f]{40}', str(x)) for x in (requested, model['revision'])):
    raise ValueError('model and tokenizer revisions must be immutable 40-character commit hashes')
tok=AutoTokenizer.from_pretrained(model.get('tokenizer_id') or model['id'],revision=requested,
                                  trust_remote_code=model.get('trust_remote_code',False))
resolved=getattr(tok,'_commit_hash',None) or getattr(tok,'init_kwargs',{}).get('_commit_hash')
if resolved and resolved != requested: raise ValueError(f'tokenizer resolved to {resolved}, expected pinned commit {requested}')
# AutoTokenizer does not expose _commit_hash consistently. The immutable
# revision passed to from_pretrained still pins the fetched tokenizer files.
resolved = resolved or requested
proposals=json.loads(Path(a.values).read_text())
if len(proposals)!=len(set(proposals)): raise ValueError('candidate proposal list contains duplicates')
contexts=annotate(generate_behavior_pairs('controls_counterbalanced',a.validation_histories,proposals,a.seed))
prompts=sorted({render_behavior_example(r,tok,True) for r in contexts if r['pair_direction']==0})
accepted, rejected={},{}
for value in proposals:
    ids=set(); failure=None
    for prompt in prompts:
        try: ids.add(continuation_token_id(tok,prompt,' '+value))
        except ValueError as exc:
            failure=str(exc); break
    if failure is None and len(ids)==1: accepted[value]=ids.pop()
    else: rejected[value]=failure or 'token ID varies across exact natural-language prompts'
owners={}
for value,token_id in accepted.items(): owners.setdefault(token_id,[]).append(value)
for token_id,words in owners.items():
    if len(words)>1:
        for value in words:
            accepted.pop(value,None); rejected[value]=f'candidate token collision ({token_id})'
first_model_values=[]
if a.match_token_ids:
    first_model_values=list(json.loads(Path(a.match_token_ids).read_text())['token_ids'])
shared=[v for v in first_model_values if v in accepted]
selected_values=shared if len(shared)>=5 else [v for v in proposals if v in accepted]
if len(selected_values)<5:
    raise ValueError(f'only {len(selected_values)} distinct one-token candidates remain; at least five are required by the shared behavioral design')
selected={v:accepted[v] for v in selected_values[:12]}
audit_rows=annotate(generate_behavior_pairs('controls_counterbalanced',4,list(selected),a.seed+1))
alignment=audit_tokenized_pairs(audit_rows,tok,selected,chat=True)
chat_hash=hashlib.sha256(str(getattr(tok,'chat_template',None)).encode()).hexdigest()
doc={'model_id':model['id'],'model_revision':model['revision'],'tokenizer_id':model.get('tokenizer_id') or model['id'],
     'tokenizer_revision':resolved,'chat_template_sha256':chat_hash,'prompt_family':'natural_entity_attribute_v1',
     'template':'nora_v1','token_ids':selected,'validated_values':list(selected),
     'first_model_shared_values':shared,'first_model_shared_count':len(shared),
     'cross_model_unmatched_values':[v for v in selected if v not in first_model_values] if a.match_token_ids else None,
     'additional_validated_not_selected':[v for v in accepted if v not in selected],
     'rejected':rejected,'validation_seed':a.seed,'validation_histories':a.validation_histories,
     'unique_prompt_prefixes_checked':len(prompts),'exact_one_token_pair_audit':alignment,
     'values_sha256':sha256_file(a.values),'config_sha256':sha256_file(a.config),
     'renderer_sha256':sha256_file('src/data/supersession_behavior.py'),
     'validator_sha256':sha256_file(__file__),
     'first_model_token_map_sha256':sha256_file(a.match_token_ids) if a.match_token_ids else None,
     'provenance':provenance(c)}
save_json(doc,a.output)
print(f'validated {len(selected)} model-specific candidates; exact edit audit {alignment["status"]}; wrote {a.output}')
