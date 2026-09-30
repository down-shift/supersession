#!/usr/bin/env python3
"""Residual localization runner for matched APPLIED/IGNORED prompt pairs."""
import argparse,json,hashlib
from pathlib import Path
import torch
from src.data.io import read_jsonl
from src.data.status_focal import audit
from src.data.supersession_behavior import render_behavior_example
from src.experiments.patching import capture_run,patched_logits,validity_patch_delta
from src.models.loader import load_model
from src.models.hooks import ResidualHooks
from src.utils import load_config

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/four_query_288.yaml');p.add_argument('--dataset',required=True);p.add_argument('--token-ids',required=True);p.add_argument('--output',required=True);p.add_argument('--layers',required=True);p.add_argument('--positions',default='final_preanswer');p.add_argument('--chat',action='store_true');a=p.parse_args()
 rows=read_jsonl(a.dataset);audit(rows,'confirmatory');out=Path(a.output)
 if out.exists() or Path(str(out)+'.provenance.json').exists():raise FileExistsError(f'{out} exists')
 cfg=load_config(a.config);model,tok=load_model(cfg);device=next(model.parameters()).device;tokens=json.loads(Path(a.token_ids).read_text())['token_ids'];layers=[int(x) for x in a.layers.split(',')]; result=[]
 groups={}
 for r in rows:groups.setdefault((r['history_id'],r['query_role'],r['focal_position']),{})[r['focal_valid']]=r
 for (hid,q,pos),pair in groups.items():
  yes,no=pair[True],pair[False]
  focal=yes['focal_variable'];valid=yes['semantic_values'][f'proposed_{focal}'];obsolete=yes['semantic_values'][f'initial_{focal}']; encoded={}
  for valid_state,r in ((True,yes),(False,no)):
   text=render_behavior_example(r,tok,a.chat);e=tok(text,return_tensors='pt',add_special_tokens=False);e={k:v.to(device) for k,v in e.items()};logits,captures=capture_run(model,e);encoded[valid_state]=(e,logits,captures,text)
  for direction,donor_state,recipient_state in (('APPLIED_to_IGNORED',True,False),('IGNORED_to_APPLIED',False,True)):
   donor=encoded[donor_state];recipient=encoded[recipient_state];di=donor[0]['input_ids'][0].tolist();ri=recipient[0]['input_ids'][0].tolist()
   if len(di)!=len(ri):raise ValueError('matched validity prompts differ in tokenized length')
   if a.positions=='final_preanswer':positions=[len(ri)-1]
   elif a.positions=='all':positions=list(range(len(ri)))
   else:positions=[int(x) for x in a.positions.split(',')]
   for layer in layers:
    source=donor[2][layer]
    # Positions are run individually to keep donor and recipient sequence axes explicit.
    for position in positions:
     patched=patched_logits(model,recipient[0],source,layer,position)[0]
     vec=lambda z:{v:float(z[tokens[v]]) for v in tokens}
     d=validity_patch_delta(recipient[1],patched,valid,obsolete,tokens)
     result.append({'history_id':hid,'query_role':q,'focal_position':pos,'layer':layer,'position':position,'direction':direction,
       'valid_value':valid,'obsolete_value':obsolete,**d,'recipient_logits':vec(recipient[1]),'donor_logits':vec(donor[1]),'patched_logits':vec(patched),
       'donor_prompt':donor[3],'recipient_prompt':recipient[3],'token_ids':ri})
 out.write_text(''.join(json.dumps(r)+'\n' for r in result))
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 Path(str(out)+'.provenance.json').write_text(json.dumps({'experiment_kind':'status_focal_validity_patch','dataset':str(Path(a.dataset).resolve()),'dataset_sha256':sha(a.dataset),'token_map_sha256':sha(a.token_ids),'config_sha256':sha(a.config),'layers':layers,'positions':a.positions,'directions':['APPLIED_to_IGNORED','IGNORED_to_APPLIED'],'raw_candidate_logits_preserved':True,'head_qkv_scanning':False,'model':cfg['model']},indent=2)+'\n')
if __name__=='__main__':main()
