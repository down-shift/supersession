#!/usr/bin/env python3
"""Score matched focal value edits and retain raw logits plus competence diagnostics."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np,torch
from tqdm.auto import tqdm
from src.data.io import read_jsonl,sha256_file
from src.data.progress import prepare_jsonl_progress,append_jsonl_record
from src.data.status_focal import (audit,render_prompt,audit_validity_prompt_alignment,
    audit_value_edit_alignment,focal_template_hash,verify_competence_artifact,focal_history_signatures)
from src.data.token_validation import continuation_token_id
from src.models.loader import load_model
from src.utils import load_config

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/four_query_288.yaml');p.add_argument('--dataset',required=True);p.add_argument('--token-ids',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 rows=read_jsonl(a.dataset);audit(rows,'confirmatory');data_prov=json.loads(Path(a.dataset+'.provenance.json').read_text())
 if data_prov.get('dataset_sha256')!=sha256_file(a.dataset) or data_prov.get('template_sha256')!=focal_template_hash():raise ValueError('confirmatory dataset provenance mismatch')
 gate_path=data_prov.get('frozen_gate_path')
 if not gate_path:raise ValueError('confirmatory data lacks frozen gate provenance')
 gate,gate_rows=verify_competence_artifact(gate_path,'frozen_gate',a.config,a.token_ids)
 if data_prov.get('frozen_gate_sha256')!=sha256_file(gate_path) or rows[0]['prompt_variant']!=gate['selected_variant'] or rows[0]['candidate_values']!=gate['candidate_values']:raise ValueError('confirmatory wording/candidates are not bound to the passing gate')
 gen_gate=json.loads(Path(gate['dataset_path']+'.provenance.json').read_text());selection_path=gen_gate.get('selection_path')
 if not selection_path:raise ValueError('frozen gate lacks development selection provenance')
 selection,dev_rows=verify_competence_artifact(selection_path,'development',a.config,a.token_ids)
 all_old=focal_history_signatures(gate_rows)|focal_history_signatures(dev_rows)
 if focal_history_signatures(rows)&all_old:raise ValueError('confirmatory histories overlap prompt development or frozen gate')
 if data_prov.get('token_map_sha256')!=sha256_file(a.token_ids) or data_prov.get('config_sha256')!=sha256_file(a.config):raise ValueError('confirmatory config/token map differs from generation')
 cfg=load_config(a.config);token_doc=json.loads(Path(a.token_ids).read_text());token_ids=token_doc['token_ids'];model,tok=load_model(cfg)
 for k in ('model_revision','tokenizer_revision'):
  req=cfg['model'].get(k) or cfg['model'].get('revision')
  if not cfg.get('resolved_'+k) and req and re.fullmatch(r'[0-9a-f]{40}',req):cfg['resolved_'+k]=req
  if not re.fullmatch(r'[0-9a-f]{40}',str(cfg.get('resolved_'+k))):raise ValueError(f'exact {k} required')
  if cfg['resolved_'+k]!=gate['scoring_provenance'].get('resolved_'+k):raise ValueError(f'loaded {k} differs from frozen gate')
  if token_doc.get(k) and token_doc[k]!=cfg.get('resolved_'+k):raise ValueError(f'frozen token map {k} mismatch')
 chat=cfg['model'].get('chat_template',True);device=next(model.parameters()).device
 # Pair index. Every prompt edit is one token; every matched validity pair has aligned status spans.
 pairs={}
 for r in rows:pairs.setdefault(r['pair_id'],{})[r['pair_direction']]=r
 if any(set(m)!={0,1} for m in pairs.values()):raise ValueError('incomplete confirmatory edit pair')
 by_cell={}
 for r in rows:by_cell.setdefault((r['history_id'],r['query_role'],r['focal_position'],r['edited_field'],r['pair_direction']),{})[r['focal_valid']]=r
 prompts={}; encodings={}
 for pair in pairs.values():
  base,edited=pair[0],pair[1]
  if base['answer'] not in token_ids or edited['answer'] not in token_ids:raise ValueError('answer missing from frozen token map')
  for r in (base,edited):
   if r['example_id'] not in prompts:
    text=render_prompt(r,tok,chat);enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    prompts[r['example_id']]=text;encodings[r['example_id']]=list(enc['input_ids'])
    for value in r['candidate_values']:
     if continuation_token_id(tok,text,' '+value)!=token_ids[value]:raise ValueError(f'{value!r} is not a stable one-token candidate at {r["example_id"]}')
  audit_value_edit_alignment(base,edited,tok,chat)
 for_key=sorted({(r['history_id'],r['query_role'],r['focal_position'],r['edited_field'],r['pair_direction']) for r in rows})
 for hid,q,pos,field,direction in for_key:
  yes=by_cell[(hid,q,pos,field,direction)][True];no=by_cell[(hid,q,pos,field,direction)][False]
  audit_validity_prompt_alignment(yes,no,tok,chat)
 out=Path(a.output);units=[]
 for pid,m in pairs.items():units.append({**m[0],'example_id':pid})
 code=hashlib.sha256(Path(__file__).read_bytes()+Path('src/data/status_focal.py').read_bytes()).hexdigest()
 fp={**cfg,'status_focal_behavior':{'dataset_sha256':sha256_file(a.dataset),'token_map_sha256':sha256_file(a.token_ids),'config_sha256':sha256_file(a.config),'template_sha256':focal_template_hash(),'code_sha256':code,'frozen_gate_sha256':sha256_file(gate_path)}}
 done=prepare_jsonl_progress(out,a.dataset,a.token_ids,fp,units,resume=a.resume)
 candidate_values=rows[0]['candidate_values'];cand_ids=np.asarray([token_ids[x] for x in candidate_values]);saved={r['example_id']:r for r in read_jsonl(out)} if out.exists() else {};units_by_id={u['example_id']:u for u in units}
 def competence(logits,answer):
  tid=token_ids[answer];target=logits[tid];ci=torch.as_tensor(cand_ids,device=logits.device)
  return {'answer':answer,'full_vocab_next_token_accuracy':int(logits.argmax().item()==tid),'target_rank':1+int((logits>target).sum().item()),'candidate_accuracy':int(ci[int(logits[ci].argmax().item())].item()==tid),'candidate_target_rank':1+int((logits[ci]>target).sum().item())}
 for pid in tqdm(sorted(pairs),desc='status_focal matched behavior'):
  if pid in done:
   stored=saved[pid];expected=units_by_id[pid]
   if any(stored.get(k)!=v for k,v in expected.items() if k!='pair_direction') or stored.get('pair_direction') is not None or 'identity_transfer_E' not in stored or 'baseline_candidate_logits' not in stored:raise ValueError('behavior checkpoint metadata/raw diagnostics differ')
   continue
  rec=[];members=pairs[pid];base=members[0];edited=members[1]
  for r in (base,edited):
   text=prompts[r['example_id']];batch=tok(text,return_tensors='pt',add_special_tokens=False);batch={k:v.to(device) for k,v in batch.items()}
   with torch.inference_mode():logits=model(**batch,use_cache=False).logits[0,-1].float()
   rec.append((logits,competence(logits,r['answer'])))
  b,e=rec[0][0],rec[1][0];source=token_ids[base['source_value']];replacement=token_ids[base['replacement_value']]
  effect=float((e[replacement]-e[source])-(b[replacement]-b[source]))
  raw=lambda x:{v:float(x[token_ids[v]]) for v in candidate_values}
  record={**base,'example_id':pid,'pair_direction':None,'identity_transfer_E':effect,
   'baseline_source_minus_replacement':float(b[source]-b[replacement]),'edited_source_minus_replacement':float(e[source]-e[replacement]),
   'baseline_candidate_logits':raw(b),'edited_candidate_logits':raw(e),
   'baseline_competence':rec[0][1],'edited_competence':rec[1][1],
   'baseline_prompt':prompts[base['example_id']],'edited_prompt':prompts[edited['example_id']]}
  append_jsonl_record(out,record)
 side=Path(str(out)+'.provenance.json')
 if not side.exists():
  side.write_text(json.dumps({'purpose':'confirmatory_behavioral_matched_edit','experiment_kind':'status_focal','dataset_path':str(Path(a.dataset).resolve()),'dataset_sha256':sha256_file(a.dataset),'token_map_sha256':sha256_file(a.token_ids),'config_sha256':sha256_file(a.config),'template_sha256':focal_template_hash(),'code_sha256':code,'chat_template':chat,'resolved_model_revision':cfg['resolved_model_revision'],'resolved_tokenizer_revision':cfg['resolved_tokenizer_revision'],'gate_artifact_sha256':sha256_file(gate_path),'raw_candidate_logits_preserved':True},indent=2)+'\n')
if __name__=='__main__':main()
