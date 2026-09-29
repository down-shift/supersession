#!/usr/bin/env python3
import argparse
from src.utils import load_config,provenance,save_json
from src.data.generate import make_contexts
from src.data.splits import grouped_split
from src.data.io import write_jsonl

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/pilot.yaml"); p.add_argument("--output",default="outputs/pilot/dataset.jsonl"); p.add_argument("--family",choices=["symbolic","natural"],default="symbolic"); p.add_argument("--token-ids",default=None); a=p.parse_args()
c=load_config(a.config); d=c["dataset"]
if a.token_ids:
 import json
 d["values"]=list(json.load(open(a.token_ids))["token_ids"])
rows=make_contexts(d["n_contexts"],c["seed"],d["values"],d["variables"],a.family); rows=grouped_split(rows,d["splits"],c["seed"]); write_jsonl(rows,a.output); save_json(provenance(c,a.output),a.output+".provenance.json"); print(f"wrote {len(rows)} examples to {a.output}")
