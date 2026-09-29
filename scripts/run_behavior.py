#!/usr/bin/env python3
import argparse
from tqdm.auto import tqdm
from src.utils import load_config,provenance,save_json
from src.data.io import read_jsonl,write_jsonl
from src.experiments.behavior import score_example,competence_gate
from src.models.loader import load_model

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/pilot.yaml"); p.add_argument("--dataset",default="outputs/pilot/dataset.jsonl"); p.add_argument("--output",default="outputs/pilot/behavior.jsonl"); p.add_argument("--token-ids",required=True,help="JSON object mapping candidate values to tokenizer token IDs"); a=p.parse_args()
c=load_config(a.config); m,t=load_model(c); rows=read_jsonl(a.dataset); ids=__import__("json").load(open(a.token_ids))["token_ids"]; results=[score_example(m,t,r,ids,chat=c["model"].get("chat_template",True)) for r in tqdm(rows,desc="Behavior scoring")]; write_jsonl(results,a.output); run=provenance(c,a.dataset); run["candidate_token_ids"]=ids; save_json(run,a.output+".provenance.json"); print(f"scored {len(results)} examples")
