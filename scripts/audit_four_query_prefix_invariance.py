#!/usr/bin/env python3
"""Audit that later query text cannot change old-x residual activations."""
import argparse, collections, json
from pathlib import Path
from src.data.io import read_jsonl
from src.data.generate import render_example
from src.experiments.prefix_invariance import audit_prefix_invariance
from src.models.loader import load_model
from src.utils import load_config, save_json

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/four_query_288.yaml'); p.add_argument('--pairs',required=True); p.add_argument('--discovery-history-ids',required=True); p.add_argument('--output',required=True); p.add_argument('--n-histories',type=int,default=2); p.add_argument('--tolerance',type=float,default=1e-4); p.add_argument('--behavior-records',required=True)
a=p.parse_args()
history_ids=json.loads(Path(a.discovery_history_ids).read_text())[:a.n_histories]
by=collections.defaultdict(dict)
for ex in read_jsonl(a.pairs):
    if ex.get('edited_binding')=='old_x' and int(ex.get('pair_direction',-1))==0:
        by[ex['history_id']][ex['query_id']]=ex
queries=('current_x','current_z','initial_x','initial_z')
for hid in history_ids:
    if set(queries)-set(by[hid]): raise ValueError(f'missing old_x baseline query records for {hid}')
c=load_config(a.config); model,tok=load_model(c)
expected='b968826d9c46dd6066d109eabc6255188de91218'
if c.get('resolved_model_revision')!=expected or c.get('resolved_tokenizer_revision')!=expected:
    raise ValueError(f'model/tokenizer revision mismatch; expected {expected}, got {c.get("resolved_model_revision")}/{c.get("resolved_tokenizer_revision")}')
scored={}
if a.behavior_records:
    scored={r['example_id']:r for r in read_jsonl(a.behavior_records)}
device=next(model.parameters()).device; report=[]
for hid in history_ids:
    result=audit_prefix_invariance(model,tok,by[hid],queries,chat=c['model'].get('chat_template',True),behavior_records=scored,tolerance=a.tolerance)
    report.extend(dict(history_id=hid,**row) for row in result['per_layer'])
result={'histories':history_ids,'queries':list(queries),'tolerance':a.tolerance,'execution':'four prompts batched with right padding to equal sequence length','verified_identical_token_prefix_through_old_x':True,'per_layer':report,'passed':all(x['passed'] for x in report)}
save_json(result,a.output)
if not result['passed']: raise SystemExit(f'prefix-invariance audit exceeded tolerance {a.tolerance}; see {a.output}')
print(f'Prefix-invariance audit passed for {len(history_ids)} histories; wrote {a.output}')
