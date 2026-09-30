#!/usr/bin/env python3
"""Resumable 24-history residual localization for focal update validity."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
import torch
from tqdm.auto import tqdm
from src.data.io import read_jsonl,sha256_file
from src.data.status_focal import (audit,render_prompt,audit_validity_prompt_alignment,
    focal_template_hash,verify_competence_artifact,focal_history_signatures)
from src.data.token_validation import continuation_token_id
from src.experiments.patching import (partition_history_ids,capture_run,patched_logits_positions,
    focal_status_patch_delta,recover_patch_checkpoint,commit_patch_pair)
from src.models.loader import load_model
from src.utils import load_config

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/four_query_288.yaml');p.add_argument('--dataset',required=True);p.add_argument('--token-ids',required=True);p.add_argument('--output',required=True);p.add_argument('--layers',required=True);p.add_argument('--positions',default='focal_status',choices=('focal_status','status_and_query','answer','all'));p.add_argument('--position-batch-size',type=int,default=32);p.add_argument('--seed',type=int,default=20261015);p.add_argument('--resume',action='store_true');a=p.parse_args()
 if a.position_batch_size<1:raise ValueError('position batch size must be positive')
 rows=read_jsonl(a.dataset);audit(rows,'confirmatory');dp=json.loads(Path(a.dataset+'.provenance.json').read_text())
 if dp.get('dataset_sha256')!=sha256_file(a.dataset) or dp.get('template_sha256')!=focal_template_hash():raise ValueError('confirmatory dataset provenance mismatch')
 gate_path=dp.get('frozen_gate_path')
 if not gate_path:raise ValueError('confirmatory data lacks frozen gate provenance')
 gate,gate_rows=verify_competence_artifact(gate_path,'frozen_gate',a.config,a.token_ids)
 if dp.get('frozen_gate_sha256')!=sha256_file(gate_path) or rows[0]['prompt_variant']!=gate['selected_variant'] or rows[0]['candidate_values']!=gate['candidate_values']:raise ValueError('patch source wording/candidates are not bound to the passing gate')
 gate_gen=json.loads(Path(gate['dataset_path']+'.provenance.json').read_text());selection_path=gate_gen.get('selection_path')
 if not selection_path:raise ValueError('frozen gate lacks development selection provenance')
 selection,dev_rows=verify_competence_artifact(selection_path,'development',a.config,a.token_ids)
 if focal_history_signatures(rows)&(focal_history_signatures(gate_rows)|focal_history_signatures(dev_rows)):raise ValueError('mechanistic confirmatory source overlaps prompt development/gate histories')
 if dp.get('token_map_sha256')!=sha256_file(a.token_ids) or dp.get('config_sha256')!=sha256_file(a.config):raise ValueError('confirmatory config/token map mismatch')
 cfg=load_config(a.config);token_doc=json.loads(Path(a.token_ids).read_text());token_ids=token_doc['token_ids'];model,tok=load_model(cfg)
 for k in ('model_revision','tokenizer_revision'):
  req=cfg['model'].get(k) or cfg['model'].get('revision')
  if not cfg.get('resolved_'+k) and req and re.fullmatch(r'[0-9a-f]{40}',req):cfg['resolved_'+k]=req
  if not re.fullmatch(r'[0-9a-f]{40}',str(cfg.get('resolved_'+k))):raise ValueError(f'exact {k} required')
  if cfg['resolved_'+k]!=gate['scoring_provenance'].get('resolved_'+k):raise ValueError(f'loaded {k} differs from frozen competence gate')
  if token_doc.get(k) and token_doc[k]!=cfg.get('resolved_'+k):raise ValueError(f'frozen token map {k} mismatch')
 chat=cfg['model'].get('chat_template',True);layers=sorted({int(x) for x in a.layers.split(',')});device=next(model.parameters()).device
 by={}
 for r in rows:
  if r.get('pair_direction')==0 and r.get('edited_field')==f'proposed_{r["focal_variable"]}':by.setdefault((r['history_id'],r['query_role'],r['focal_position']),{})[r['focal_valid']]=r
 history_ids=sorted({r['history_id'] for r in rows})
 if len(history_ids)!=96:raise ValueError('mechanistic discovery requires the 96-history confirmatory set')
 discovery,heldout=partition_history_ids(history_ids,24,a.seed);chosen=set(discovery)
 out=Path(a.output);manifest=Path(str(out)+'.run.json');complete=Path(str(out)+'.complete.jsonl');out.parent.mkdir(parents=True,exist_ok=True)
 # Do all token-position alignment checks before the first model forward.
 aligned={};encoded={}
 for key,pair in by.items():
  if key[0] not in chosen:continue
  yes,no=pair[True],pair[False]
  report=audit_validity_prompt_alignment(yes,no,tok,chat);aligned[key]=report
  for valid,r in ((True,yes),(False,no)):
   text=report['applied_prompt'] if valid else report['ignored_prompt'];batch=tok(text,return_tensors='pt',add_special_tokens=False);batch={k:v.to(device) for k,v in batch.items()}
   for value in (r['semantic_values']['proposed_'+r['focal_variable']],r['semantic_values']['initial_'+r['focal_variable']]):
    if continuation_token_id(tok,text,' '+value)!=token_ids[value]:raise ValueError(f'focal decision value {value!r} is not a stable one-token answer')
   encoded[(key,valid)]=(text,batch)
 sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
 try:commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
 except Exception:commit=None
 fingerprint={'dataset_sha256':sha(a.dataset),'token_map_sha256':sha(a.token_ids),'config_sha256':sha(a.config),'gate_sha256':sha(gate_path),
  'template_sha256':focal_template_hash(),'model_id':cfg['model']['id'],'model_revision':cfg['resolved_model_revision'],'tokenizer_revision':cfg['resolved_tokenizer_revision'],
  'stage':'discovery','history_ids':discovery,'heldout_history_ids':heldout,'seed':a.seed,'layers':layers,'positions':a.positions,'position_batch_size':a.position_batch_size,'chat_template':chat,'git_commit':commit}
 fingerprint['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()+Path(__import__('src.experiments.patching',fromlist=['__file__']).__file__).read_bytes()).hexdigest()
 if a.resume:
  if not manifest.exists() or json.loads(manifest.read_text())!=fingerprint:raise ValueError('resume fingerprint mismatch; preserve checkpoint and choose new output')
  done=recover_patch_checkpoint(out,complete)
 else:
  if out.exists() or manifest.exists() or complete.exists():raise FileExistsError('patch output/checkpoint exists; use --resume or choose a fresh path')
  out.touch();complete.touch();manifest.write_text(json.dumps(fingerprint,indent=2)+'\n');done=set()
 candidate_values=rows[0]['candidate_values']
 def logits_map(x):return {v:float(x[token_ids[v]]) for v in candidate_values}
 try:
  for key in tqdm(sorted(k for k in by if k[0] in chosen),desc='status_focal discovery cells'):
   hid,qrole,fpos=key;cell_id=f'{hid}:{qrole}:{fpos}'
   if cell_id in done:continue
   yes,no=by[key][True],by[key][False];dirrows=[]
   for direction,donor_state,recipient_state in (('APPLIED_to_IGNORED',True,False),('IGNORED_to_APPLIED',False,True)):
    donor=encoded[(key,donor_state)];recipient=encoded[(key,recipient_state)]
    with torch.inference_mode():
     donor_logits,donor_acts=capture_run(model,donor[1]);recipient_logits,recipient_acts=capture_run(model,recipient[1])
    ap=aligned[key]['status_positions']
    def query_positions(text,row):
     char=text.index('What is '+row['literal_names'][row['query']]+'?');end=char+len('What is '+row['literal_names'][row['query']])
     off=tok(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'];return [i for i,(x,y) in enumerate(off) if x<end and y>char]
    qp=query_positions(recipient[0],no if recipient_state is False else yes)
    if a.positions=='focal_status':positions=ap
    elif a.positions=='status_and_query':positions=sorted(set(ap+qp))
    elif a.positions=='answer':positions=[recipient[1]['input_ids'].shape[1]-1]
    else:positions=list(range(recipient[1]['input_ids'].shape[1]))
    focal=yes['focal_variable'];proposed=yes['semantic_values']['proposed_'+focal];initial=yes['semantic_values']['initial_'+focal]
    donor_applied=donor_state is True
    for layer in layers:
     if layer<0 or layer>=len(donor_acts):raise ValueError(f'layer {layer} outside model')
     for start in range(0,len(positions),a.position_batch_size):
      pp=positions[start:start+a.position_batch_size]
      with torch.inference_mode():patched=patched_logits_positions(model,recipient[1],donor_acts[layer],layer,pp)
      for j,position in enumerate(pp):
       delta=focal_status_patch_delta(recipient_logits,patched[j],proposed,initial,token_ids,donor_applied)
       dirrows.append({'pair_id':cell_id,'history_id':hid,'query_role':qrole,'focal_position':fpos,'layer':layer,'position':position,'direction':direction,
        'focal_variable':focal,'proposed_value':proposed,'initial_value':initial,**delta,
        'donor_logits':logits_map(donor_logits),'recipient_logits':logits_map(recipient_logits),'patched_logits':logits_map(patched[j]),
        'donor_prompt':donor[0],'recipient_prompt':recipient[0],'status_positions':ap})
   commit_patch_pair(out,complete,cell_id,dirrows);done.add(cell_id)
 except BaseException:raise
 side=Path(str(out)+'.provenance.json')
 side.write_text(json.dumps({'purpose':'exploratory_validity_mechanistic_discovery','stage':'discovery','dataset_sha256':fingerprint['dataset_sha256'],'token_map_sha256':fingerprint['token_map_sha256'],'config_sha256':fingerprint['config_sha256'],'gate_sha256':fingerprint['gate_sha256'],'template_sha256':fingerprint['template_sha256'],'model_revision':fingerprint['model_revision'],'tokenizer_revision':fingerprint['tokenizer_revision'],'discovery_history_ids':discovery,'heldout_history_ids':heldout,'reserve_histories_used':False,'layers':layers,'positions':a.positions,'position_batch_size':a.position_batch_size,'directions':['APPLIED_to_IGNORED','IGNORED_to_APPLIED'],'metric':'S = logit(proposed_focal) - logit(initial_focal)','raw_candidate_logits_preserved':True},indent=2)+'\n')
if __name__=='__main__':main()
