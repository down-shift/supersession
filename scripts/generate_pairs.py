#!/usr/bin/env python3
import argparse
from src.utils import load_config,provenance,save_json
from src.data.generate import make_contexts,counterfactual_pair
from src.data.splits import grouped_split
from src.data.io import write_jsonl
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/pilot.yaml"); p.add_argument("--output",default="outputs/pilot/pairs.jsonl"); p.add_argument("--token-ids",default=None); p.add_argument("--family",choices=["symbolic","natural"],default="symbolic"); a=p.parse_args(); c=load_config(a.config); d=c["dataset"]
if a.token_ids:
 import json
 d["values"]=list(json.load(open(a.token_ids))["token_ids"])
contexts=make_contexts(d["n_contexts"],c["seed"],d["values"],d["variables"],a.family); contexts=grouped_split(contexts,d["splits"],c["seed"]); pairs=[]
for ctx in contexts:
 for role in ("O_q","C_q","O_d","C_d"):
  alternate=next(v for v in d["values"] if v not in ctx["roles"].values())
  pair=counterfactual_pair(ctx,role,ctx["roles"][role],alternate)
  for direction,ex in enumerate(pair): ex.update({"pair_id":f"{ctx['example_id']}:{role}","intervention_role":role,"pair_direction":direction,"example_id":f"{ctx['example_id']}:{role}:{direction}"}); pairs.append(ex)
write_jsonl(pairs,a.output); save_json(provenance(c,a.output),a.output+".provenance.json"); print(f"wrote {len(pairs)} paired intervention members")
