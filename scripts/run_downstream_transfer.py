#!/usr/bin/env python3
"""Resumable complete-sequence scoring; development/gate are competence-only."""
import argparse,json,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tqdm.auto import tqdm
from src.data.io import read_jsonl,sha256_file
from src.data.progress import prepare_jsonl_progress,append_jsonl_record
from src.data.downstream_transfer import validated_stage,render,template_hash,seal_artifact,verify_sealed_artifact
from src.analysis.downstream_transfer import sequence_logprob,score_protocol_record,evaluate_competence_gate
from src.models.loader import load_model
from src.utils import load_config
SCORING_VERSION='complete_sequence_logprob_v1'
codehash=hashlib.sha256(Path(__file__).read_bytes()+Path('src/analysis/downstream_transfer.py').read_bytes()+SCORING_VERSION.encode()).hexdigest()
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--config',default='configs/downstream_transfer_v1.yaml');p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--gate');p.add_argument('--token-audit',required=True);p.add_argument('--values-json',default='configs/downstream_transfer_values.json');p.add_argument('--codes-json',default='configs/downstream_transfer_codes.json')
a=p.parse_args();rows=read_jsonl(a.dataset);stage=validated_stage(rows);prov=json.loads(Path(a.dataset+'.provenance.json').read_text())
if prov.get('protocol')!='downstream_transfer_v1' or prov.get('dataset_sha256')!=sha256_file(a.dataset) or prov.get('template_sha256')!=template_hash(): raise ValueError('dataset provenance/hash mismatch')
c=load_config(a.config);frozen={}
token_audit=json.loads(Path(a.token_audit).read_text())
if (token_audit.get('audit')!='tokenizer_only_no_model_inference' or
 token_audit.get('tokenizer_id')!=(c['model'].get('tokenizer_id') or c['model']['id']) or
 token_audit.get('tokenizer_revision')!=c['model']['tokenizer_revision'] or
 token_audit.get('chat_template')!=c['model'].get('chat_template',True) or
 set(token_audit.get('codes',{}))!=set(rows[0]['code_vocabulary'])): raise ValueError('tokenizer audit does not match frozen tokenizer/config/code vocabulary')
for key,path in [('config_sha256',a.config),('values_sha256',a.values_json),('codes_sha256',a.codes_json)]:
 if prov.get(key)!=sha256_file(path): raise ValueError(f'dataset provenance {key} mismatch')
if rows[0]['value_vocabulary']!=json.loads(Path(a.values_json).read_text()) or rows[0]['code_vocabulary']!=json.loads(Path(a.codes_json).read_text()): raise ValueError('dataset vocabularies differ from frozen files')
if stage=='confirmatory':
 if not a.gate: raise ValueError('confirmatory scoring requires frozen --gate')
 frozen=verify_sealed_artifact(json.loads(Path(a.gate).read_text()))
 if frozen.get('pass') is not True: raise ValueError('gate did not pass')
 for key,path in [('config_sha256',a.config),('values_sha256',a.values_json),('codes_sha256',a.codes_json)]:
  if frozen.get(key)!=sha256_file(path): raise ValueError(f'frozen {key} mismatch')
 if frozen.get('template_sha256')!=template_hash() or frozen.get('dataset_sha256')!=sha256_file(frozen.get('dataset_path','')): raise ValueError('frozen gate dataset/template mismatch')
 if frozen.get('scoring_code_sha256')!=codehash or frozen.get('scoring_version')!=SCORING_VERSION: raise ValueError('frozen scoring implementation differs from gate')
 if frozen.get('token_audit_sha256')!=sha256_file(a.token_audit): raise ValueError('tokenizer audit differs from frozen gate')
 if frozen.get('scores_sha256')!=sha256_file(frozen.get('scores_path','')): raise ValueError('frozen gate score file changed')
 if prov.get('gate_sha256')!=sha256_file(a.gate): raise ValueError('dataset is not bound to supplied gate')
 if frozen.get('model_revision')!=c['model']['revision'] or frozen.get('tokenizer_revision')!=c['model']['tokenizer_revision']: raise ValueError('model/tokenizer revision differs from gate')
model,tok=load_model(c);candidates=rows[0]['code_vocabulary'];chat=c['model'].get('chat_template',True)
def scorer(r):
 prompt=render(r,tok,chat);lp={code:sequence_logprob(model,tok,prompt,' '+code) for code in candidates};best=max(lp,key=lp.get)
 return {**r,'candidate_logprobs':lp,'current_code_accuracy':int(best==r['answer_code']),'candidate_accuracy':int(best==r['answer_code'])}
finger={**c,'downstream_transfer_v1':{'dataset_sha256':sha256_file(a.dataset),'template_sha256':template_hash(),'scoring_code_sha256':codehash,'token_audit_sha256':sha256_file(a.token_audit),'stage':stage,'scoring_version':SCORING_VERSION}}
done=prepare_jsonl_progress(a.output,a.dataset,a.codes_json,finger,rows,resume=a.resume);side=Path(a.output+'.provenance.json')
side_data={'protocol':'downstream_transfer_v1','stage':stage,'dataset_path':str(Path(a.dataset).resolve()),'dataset_sha256':sha256_file(a.dataset),'template_sha256':template_hash(),'scoring_code_sha256':codehash,'scoring_version':SCORING_VERSION,'config_sha256':sha256_file(a.config),'values_sha256':sha256_file(a.values_json),'codes_sha256':sha256_file(a.codes_json),'token_audit_sha256':sha256_file(a.token_audit),'model_revision':c['model']['revision'],'tokenizer_revision':c['model']['tokenizer_revision'],'causal_effects_computed':stage=='confirmatory','gate_sha256':sha256_file(a.gate) if a.gate else None}
if side.exists():
 if json.loads(side.read_text())!=side_data: raise ValueError('score provenance differs; refuse resume')
else:
 with side.open('x') as f:f.write(json.dumps(side_data,indent=2)+'\n')
completed=read_jsonl(a.output) if Path(a.output).exists() else [];cache={r['example_id']:r for r in completed}
for r in tqdm(rows,desc='downstream_transfer scoring'):
 if r['example_id'] in done: continue
 append_jsonl_record(a.output,score_protocol_record(r,scorer,cache,stage))
if stage=='frozen_gate':
 result=evaluate_competence_gate(rows,read_jsonl(a.output),.97)
 artifact={'protocol':'downstream_transfer_v1','stage':'frozen_gate',**result,'dataset_path':str(Path(a.dataset).resolve()),'dataset_sha256':sha256_file(a.dataset),'scores_path':str(Path(a.output).resolve()),'scores_sha256':sha256_file(a.output),'template_sha256':template_hash(),'scoring_code_sha256':codehash,'scoring_version':SCORING_VERSION,'config_sha256':sha256_file(a.config),'values_sha256':sha256_file(a.values_json),'codes_sha256':sha256_file(a.codes_json),'token_audit_path':str(Path(a.token_audit).resolve()),'token_audit_sha256':sha256_file(a.token_audit),'model_revision':c['model']['revision'],'tokenizer_revision':c['model']['tokenizer_revision']}
 Path(a.output+'.gate.json').open('x').write(json.dumps(seal_artifact(artifact),indent=2)+'\n');print(f"gate pass={result['pass']}")
print(f'scored/resumed {len(rows)} records')
