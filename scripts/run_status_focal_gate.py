#!/usr/bin/env python3
"""Score status_focal prompts for competence only; never computes causal R."""
import argparse,json
from pathlib import Path
import torch
from tqdm.auto import tqdm
from src.data.status_focal import audit
from src.data.supersession_behavior import render_behavior_example
from src.data.io import read_jsonl
from src.models.loader import load_model
from src.utils import load_config

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/four_query_288.yaml');p.add_argument('--dataset',required=True);p.add_argument('--token-ids',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 rows=read_jsonl(a.dataset);audit(rows);out=Path(a.output)
 if out.exists():raise FileExistsError(f'{out} exists')
 ids=json.loads(Path(a.token_ids).read_text())['token_ids'];model,tok=load_model(load_config(a.config));device=next(model.parameters()).device
 with out.open('w') as f:
  for r in tqdm(rows,desc='status_focal competence'):
   prompt=render_behavior_example(r,tok,chat=True);batch=tok(prompt,return_tensors='pt',add_special_tokens=False);batch={k:v.to(device) for k,v in batch.items()}
   with torch.inference_mode(): logits=model(**batch,use_cache=False).logits[0,-1].float()
   target=ids[r['answer']];cand=torch.tensor([ids[x] for x in r['candidate_values']],device=logits.device);tl=logits[target]
   d={**r,'prompt':prompt,'full_vocab_next_token_accuracy':int(logits.argmax().item()==target),'target_rank':1+int((logits>tl).sum().item()),'candidate_accuracy':int(cand[logits[cand].argmax()].item()==target),'candidate_target_rank':1+int((logits[cand]>tl).sum().item())}
   f.write(json.dumps(d)+'\n')
if __name__=='__main__':main()
