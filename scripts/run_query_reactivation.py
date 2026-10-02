#!/usr/bin/env python3
"""Resumable query_reactivation_v1 scoring; development/gate are competence-only."""
import argparse, json, re
import hashlib
from pathlib import Path
from tqdm.auto import tqdm
from src.data.io import read_jsonl,sha256_file
from src.data.progress import prepare_jsonl_progress,append_jsonl_record
from src.data.query_reactivation import validated_stage,render,template_hash,validate_dataset_provenance,verify_sealed_artifact,seal_artifact,candidate_values_digest,verify_gate_values
from src.data.token_validation import continuation_token_id
from src.analysis.query_reactivation import score_protocol_record,evaluate_competence_gate
from src.experiments.behavior import score_example
from src.models.loader import load_model
from src.utils import load_config,provenance,save_json

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--token-ids',required=True)
p.add_argument('--config',default='configs/four_query_288.yaml');p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--gate')
a=p.parse_args(); rows=read_jsonl(a.dataset);stage=validated_stage(rows);dsprov=validate_dataset_provenance(a.dataset,a.dataset+'.provenance.json')
score_code_paths=(Path(__file__),Path('src/analysis/query_reactivation.py'),Path('src/data/query_reactivation.py'),
 Path('src/experiments/behavior.py'),Path('src/data/progress.py'),Path('src/models/loader.py'))
scoring_code_sha256=hashlib.sha256(b''.join(path.read_bytes() for path in score_code_paths)).hexdigest()
if stage=='confirmatory':
 if not a.gate: raise ValueError('confirmatory scoring requires frozen gate')
 gate=verify_sealed_artifact(json.loads(Path(a.gate).read_text()))
 if gate.get('pass') is not True: raise ValueError('invalid/failed gate seal')
if (gate['template_sha256']!=template_hash() or gate.get('scores_sha256')!=sha256_file(gate.get('scores_path',''))
     or gate.get('dataset_sha256')!=sha256_file(gate.get('dataset_path',''))
     or dsprov.get('gate_sha256')!=sha256_file(a.gate)
     or gate.get('scoring_code_sha256')!=scoring_code_sha256):
  raise ValueError('gate dataset/scores/template/hash lineage changed')
c=load_config(a.config);token_doc=json.loads(Path(a.token_ids).read_text());ids=token_doc['token_ids']
for key in ('revision','tokenizer_revision'):
 if not re.fullmatch('[0-9a-f]{40}',str(c['model'].get(key))): raise ValueError(f'frozen query_reactivation_v1 requires immutable {key}')
 if token_doc.get(key) and token_doc[key]!=c['model'][key]: raise ValueError(f'token map {key} differs from scoring config')
if token_doc.get('tokenizer_id') and token_doc['tokenizer_id']!=c['model'].get('tokenizer_id',c['model']['id']): raise ValueError('token map tokenizer ID differs from scoring config')
if token_doc.get('chat_template') is not None and token_doc['chat_template']!=c['model'].get('chat_template',True): raise ValueError('token map chat-template setting differs from scoring config')
if stage=='confirmatory' and (gate.get('token_map_sha256')!=sha256_file(a.token_ids) or gate.get('config_sha256')!=sha256_file(a.config)):
 raise ValueError('confirmatory token map/config differs from frozen gate')
if stage=='confirmatory': verify_gate_values(rows[0]['candidate_values'],gate)
if stage=='confirmatory':
 gate_side=json.loads(Path(gate['scores_path']+'.provenance.json').read_text())
 if (gate_side.get('protocol')!='query_reactivation_v1' or gate_side.get('stage')!='frozen_gate'
     or gate_side.get('causal_effects_computed') is not False
     or gate_side.get('dataset_sha256')!=gate.get('dataset_sha256')
     or gate_side.get('token_map_sha256')!=gate.get('token_map_sha256')
     or gate_side.get('scoring_code_sha256')!=scoring_code_sha256):
  raise ValueError('frozen gate score provenance is invalid')
if not a.resume and any(Path(a.output+s).exists() for s in ('','.run.json','.provenance.json')): raise FileExistsError('choose a fresh score path')
model,tok=load_model(c); chat=c['model'].get('chat_template',True)
for r in rows:
 if r['answer'] not in ids or (stage=='confirmatory' and (r['source_value'] not in ids or r['replacement_value'] not in ids)): raise ValueError('missing answer/edit value from frozen token map')
 for value in rows[0]['candidate_values']:
  if value not in ids or continuation_token_id(tok,render(rows[0],tok,chat),' '+value)!=ids[value]:
   raise ValueError(f'candidate {value!r} is not a stable one-token continuation in this scorer')
fingerprint={**c,'query_reactivation_v1':{'template_sha256':template_hash(),'stage':stage,'dataset_sha256':sha256_file(a.dataset),'token_ids_sha256':sha256_file(a.token_ids),'scoring_code_sha256':scoring_code_sha256}}
done=prepare_jsonl_progress(a.output,a.dataset,a.token_ids,fingerprint,rows,resume=a.resume)
side=Path(a.output+'.provenance.json')
if not side.exists():
 side_data={**provenance(c,a.dataset),'protocol':'query_reactivation_v1','stage':stage,'dataset_sha256':sha256_file(a.dataset),
  'dataset_path':str(Path(a.dataset).resolve()),'token_map_sha256':sha256_file(a.token_ids),'template_sha256':template_hash(),'scoring_code_sha256':scoring_code_sha256,
  'causal_effects_computed':stage=='confirmatory','scoring_mode':'competence_only' if stage!='confirmatory' else 'paired_baseline_counterfactual'}
 save_json(side_data,side)
completed=read_jsonl(a.output) if Path(a.output).exists() else []; cache={r['example_id']:r for r in completed}
for r in tqdm(rows,desc='query_reactivation scoring'):
 if r['example_id'] in done: continue
 s=score_protocol_record(r,lambda row:score_example(model,tok,row,ids,chat=chat,renderer=render,preserve_metadata=True),cache,stage)
 append_jsonl_record(a.output,s)
if stage=='frozen_gate':
 scores={r['example_id']:r for r in read_jsonl(a.output)}
 if set(scores)!={r['example_id'] for r in rows}: raise ValueError('gate scores do not exactly cover dataset')
 gate_result=evaluate_competence_gate(rows,list(scores.values()))
 artifact={'protocol':'query_reactivation_v1','stage':'frozen_gate',**gate_result,
  'candidate_values':rows[0]['candidate_values'],'candidate_values_sha256':candidate_values_digest(rows[0]['candidate_values']),
  'dataset_path':str(Path(a.dataset).resolve()),
  'dataset_sha256':sha256_file(a.dataset),'scores_path':str(Path(a.output).resolve()),'scores_sha256':sha256_file(a.output),
  'template_sha256':template_hash(),'scoring_code_sha256':scoring_code_sha256,'token_map_sha256':sha256_file(a.token_ids),'config_sha256':sha256_file(a.config)}
 artifact=seal_artifact(artifact)
 gate_path=Path(a.output+'.gate.json')
 with gate_path.open('x') as f: json.dump(artifact,f,indent=2)
 print(f'gate pass={gate_result["pass"]}; artifact={gate_path}')
print(f'scored/resumed {len(rows)} records; {len(done)} loaded')
