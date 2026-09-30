#!/usr/bin/env python3
"""Resumable competence-only status_focal scoring with frozen provenance."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np,torch
from tqdm.auto import tqdm
from src.data.io import read_jsonl,sha256_file
from src.data.progress import prepare_jsonl_progress,append_jsonl_record
from src.data.status_focal import audit,render_prompt,audit_validity_prompt_alignment,focal_template_hash
from src.data.token_validation import continuation_token_id
from src.models.loader import load_model
from src.utils import load_config,provenance,save_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/four_query_288.yaml');p.add_argument('--dataset',required=True);p.add_argument('--token-ids',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 rows=read_jsonl(a.dataset);audit(rows)
 if not rows or rows[0]['stage'] not in ('development','frozen_gate'):raise ValueError('competence runner only accepts development/frozen_gate data')
 token_doc=json.loads(Path(a.token_ids).read_text());token_ids=token_doc['token_ids'];config=load_config(a.config);model,tok=load_model(config)
 for key in ('model_revision','tokenizer_revision'):
  requested=config['model'].get(key) or config['model'].get('revision')
  if not config.get('resolved_'+key) and requested and re.fullmatch(r'[0-9a-f]{40}',requested):config['resolved_'+key]=requested
  if not re.fullmatch(r'[0-9a-f]{40}',str(config.get('resolved_'+key))):raise ValueError(f'exact {key} is required')
  if token_doc.get(key) and token_doc[key]!=config.get('resolved_'+key):raise ValueError(f'frozen token map {key} differs from loaded model')
 chat=config['model'].get('chat_template',True);device=next(model.parameters()).device;candidate_values=rows[0]['candidate_values']
 if any(r['candidate_values']!=candidate_values for r in rows):raise ValueError('candidate map changes across prompts')
 candidate_ids=np.asarray([token_ids[v] for v in candidate_values],dtype=np.int64);alignment_by_cell={};groups={}
 for r in rows:groups.setdefault((r['history_id'],r['prompt_variant'],r['query_role'],r['focal_position']),{})[r['focal_valid']]=r
 for key,pair in groups.items():
  try:audit_validity_prompt_alignment(pair[True],pair[False],tok,chat);alignment_by_cell[key]=(True,None)
  except (ValueError,RuntimeError) as exc:alignment_by_cell[key]=(False,str(exc))
 for r in rows:
  prompt=render_prompt(r,tok,chat)
  if r['answer'] not in token_ids:raise ValueError('answer absent from frozen token map')
  for value in candidate_values:
   if continuation_token_id(tok,prompt,' '+value)!=token_ids[value]:raise ValueError(f'{value!r} is not a stable one-token candidate at {r["example_id"]}')
 code=hashlib.sha256(Path(__file__).read_bytes()+Path('src/data/status_focal.py').read_bytes()+Path('src/data/supersession_behavior.py').read_bytes()).hexdigest()
 fingerprint={**config,'status_focal_gate':{'code_sha256':code,'config_sha256':sha256_file(a.config),'template_sha256':focal_template_hash(),'chat':chat}}
 done=prepare_jsonl_progress(a.output,a.dataset,a.token_ids,fingerprint,rows,resume=a.resume)
 side=Path(a.output+'.provenance.json')
 if a.resume and done and not side.exists():raise ValueError('checkpoint provenance missing; use a fresh path')
 if not side.exists():
  save_json({'purpose':'competence_gate_only','experiment_kind':'status_focal','stage':rows[0]['stage'],**provenance(config,a.dataset),
   'dataset_path':str(Path(a.dataset).resolve()),'dataset_sha256':sha256_file(a.dataset),'config_path':str(Path(a.config).resolve()),'config_sha256':sha256_file(a.config),
   'token_map_path':str(Path(a.token_ids).resolve()),'token_map_sha256':sha256_file(a.token_ids),'template_sha256':focal_template_hash(),
   'resolved_model_revision':config['resolved_model_revision'],'resolved_tokenizer_revision':config['resolved_tokenizer_revision'],
   'code_sha256':code,'chat_template':chat,'candidate_values':candidate_values,
   'outputs':['full_vocab_next_token_accuracy','target_rank','candidate_accuracy','candidate_target_rank'],'causal_effects_computed':False},side)
 saved={r['example_id']:r for r in read_jsonl(a.output)} if Path(a.output).exists() else {}
 for r in tqdm(rows,desc='status_focal competence'):
  if r['example_id'] in done:
   if any(saved[r['example_id']].get(k)!=v for k,v in r.items()):raise ValueError('saved gate metadata differs from dataset')
   continue
  prompt=render_prompt(r,tok,chat);batch=tok(prompt,return_tensors='pt',add_special_tokens=False);batch={k:v.to(device) for k,v in batch.items()}
  with torch.inference_mode():logits=model(**batch,use_cache=False).logits[0,-1].float()
  target=token_ids[r['answer']];target_logit=logits[target];candidate=torch.as_tensor(candidate_ids,device=logits.device)
  record={**r,'prompt':prompt,'query_position':int(batch['input_ids'].shape[1]-1),
   'full_vocab_next_token_accuracy':int(logits.argmax().item()==target),'target_rank':1+int((logits>target_logit).sum().item()),
   'candidate_accuracy':int(candidate[int(logits[candidate].argmax().item())].item()==target),
   'candidate_target_rank':1+int((logits[candidate]>target_logit).sum().item()),
   'validity_alignment_passed':alignment_by_cell[(r['history_id'],r['prompt_variant'],r['query_role'],r['focal_position'])][0],
   'validity_alignment_error':alignment_by_cell[(r['history_id'],r['prompt_variant'],r['query_role'],r['focal_position'])][1]}
  append_jsonl_record(a.output,record)
 print(f'scored/resumed {len(rows)} focal competence prompts; {len(done)} loaded from checkpoint')
if __name__=='__main__':main()
