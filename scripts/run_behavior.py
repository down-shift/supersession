#!/usr/bin/env python3
import argparse
from tqdm.auto import tqdm
from src.utils import load_config,provenance,save_json
from src.data.io import read_jsonl
from src.data.progress import prepare_jsonl_progress,append_jsonl_record
from src.experiments.behavior import score_example,competence_gate
from src.models.loader import load_model

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/pilot.yaml"); p.add_argument("--dataset",default="outputs/pilot/dataset.jsonl"); p.add_argument("--output",default="outputs/pilot/behavior.jsonl"); p.add_argument("--token-ids",required=True,help="JSON object mapping candidate values to tokenizer token IDs"); p.add_argument("--resume",action="store_true",help="continue a matching interrupted run from its last complete record"); a=p.parse_args()
c=load_config(a.config); rows=read_jsonl(a.dataset); ids=__import__("json").load(open(a.token_ids))["token_ids"]
completed=prepare_jsonl_progress(a.output,a.dataset,a.token_ids,c,rows,resume=a.resume)
m,t=load_model(c); todo=[r for r in rows if str(r["example_id"]) not in completed]
for row in tqdm(todo,desc=f"Behavior scoring ({len(completed)} already saved)"):
    result=score_example(m,t,row,ids,chat=c["model"].get("chat_template",True))
    append_jsonl_record(a.output,result)
run=provenance(c,a.dataset); run["candidate_token_ids"]=ids; save_json(run,a.output+".provenance.json"); print(f"scored {len(rows)} examples; {len(completed)} loaded from checkpoint")
