#!/usr/bin/env python3
"""POST-HOC / EXPLORATORY analysis of existing four-query matched edits."""
import argparse, json, hashlib
from pathlib import Path
import pandas as pd
from src.data.io import read_jsonl, sha256_file
from src.analysis.query_reactivation import history_contrasts, summarize
from src.utils import save_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--behavior',required=True,help='existing four-query unedited competence JSONL')
p.add_argument('--pairs',required=True,help='existing matched causal-edit score JSONL')
p.add_argument('--output-dir',required=True)
p.add_argument('--stage',choices=('posthoc_exploratory','confirmatory'),default='posthoc_exploratory')
p.add_argument('--seed',type=int,default=20261002); p.add_argument('--bootstrap-draws',type=int,default=10000)
a=p.parse_args(); out=Path(a.output_dir)
if out.exists() and any(out.iterdir()): raise FileExistsError('preserve prior analyses; choose a new output directory')
out.mkdir(parents=True,exist_ok=True)
behavior_all=read_jsonl(a.behavior); pairs=read_jsonl(a.pairs)
if a.stage=='confirmatory':
 if any(r.get('stage')!='confirmatory' for r in behavior_all): raise ValueError('confirmatory analysis requires confirmatory query_reactivation_v1 scores')
 side=Path(a.behavior+'.provenance.json')
 if not side.exists(): raise ValueError('confirmatory score provenance sidecar is required')
 prov=json.loads(side.read_text())
 if prov.get('protocol')!='query_reactivation_v1' or prov.get('stage')!='confirmatory' or prov.get('dataset_sha256')!=sha256_file(prov.get('dataset_path','')): raise ValueError('confirmatory score provenance/hash mismatch')
if a.stage=='confirmatory':
 if any(r.get('stage')!='confirmatory' for r in behavior_all):
  raise ValueError('confirmatory analysis requires confirmatory query_reactivation_v1 scores')
 baseline_behavior=[r for r in behavior_all if r.get('pair_direction')==0]
 if len(baseline_behavior)==0 or len(baseline_behavior)*2!=len(behavior_all):
  raise ValueError('confirmatory competence summaries require exactly the baseline member of each pair')
else:
 baseline_behavior=behavior_all
behavior_by_cell={}
for r in baseline_behavior:
 key=(r['history_id'],r['query_id'])
 previous=behavior_by_cell.get(key)
 if previous is not None:
  fields=('prompt','answer','full_vocab_next_token_accuracy','full_vocab_rank','generated_first_token')
  if any(previous.get(field)!=r.get(field) for field in fields):
   raise ValueError(f'baseline behavior differs across matched bindings: {key}')
 else:
  behavior_by_cell[key]=r
behavior=list(behavior_by_cell.values()); ids={r['history_id'] for r in behavior}
if len(behavior)!=4*len(ids): raise ValueError('behavior must contain all four unique query records per history')
if {r['query_id'] for r in behavior}!={'current_x','initial_x','current_z','initial_z'}: raise ValueError('expected four-query behavior cells')
rows=history_contrasts(pairs)
if {r['history_id'] for r in rows} != ids: raise ValueError('behavior and edit histories must match exactly')
pd.DataFrame(rows).to_csv(out/'query_reactivation_per_history.csv',index=False)
result=summarize(rows,a.seed,a.bootstrap_draws)
result.update({'exploratory':a.stage=='posthoc_exploratory','analysis_stage':a.stage,
 'input_behavior':str(Path(a.behavior).resolve()),'input_behavior_sha256':sha256_file(a.behavior),
 'input_pairs':str(Path(a.pairs).resolve()),'input_pairs_sha256':sha256_file(a.pairs),
 'analysis_script_sha256':sha256_file(__file__),
 'competence':{'by_query':{},'by_query_variable_orientation':{}}})
for r in behavior:
 key=r['query_id']; bucket=result['competence']['by_query'].setdefault(key,[]); bucket.append(r)
 query_var=r.get('variables',['x','z'])[0 if r['query_id'].endswith('_x') else 1]
 key=(r['query_id'],query_var,r.get('orientation'),r.get('prompt_variant',r.get('prompt_family')),str(r.get('variable_pair')))
 result['competence']['by_query_variable_orientation'].setdefault('|'.join(map(str,key)),[]).append(r)
for table in result['competence'].values():
 for key,rs in list(table.items()):
  table[key]={'n':len(rs),'candidate_accuracy':sum(x.get('accuracy',0) for x in rs)/len(rs),
    'full_vocab_next_token_accuracy':sum(x.get('full_vocab_next_token_accuracy',0) for x in rs)/len(rs)}
save_json(result,out/'query_reactivation_summary.json')
print(f"wrote exploratory summaries for {len(rows)} histories to {out}")
