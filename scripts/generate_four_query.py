#!/usr/bin/env python3
"""Generate balanced history-level four-query and matched-edit datasets."""
import argparse, json
from src.utils import load_config,provenance,save_json
from src.data.generate import make_histories,expand_history_queries,matched_history_pairs,PROMPT_VARIANTS
from src.data.splits import grouped_split
from src.data.io import write_jsonl

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/four_query_288.yaml"); p.add_argument("--output",default="outputs/four_query/histories.jsonl")
p.add_argument("--kind",choices=("histories","queries","pairs"),default="queries"); p.add_argument("--token-ids",default=None); p.add_argument("--partition",choices=("prompt_dev","gate","confirmatory"),default="confirmatory"); p.add_argument("--prompt-variant",choices=PROMPT_VARIANTS,default="first_latest"); p.add_argument("--all-prompt-variants",action="store_true"); p.add_argument("--calibration",action="store_true",help="legacy alias for an independent gate dataset"); a=p.parse_args()
c=load_config(a.config); d=c["dataset"]; values=d["values"]
if not a.token_ids: raise ValueError("pass the frozen --token-ids.json from four-query token validation")
values=list(json.load(open(a.token_ids))["token_ids"])
if len(values)!=int(d.get("candidate_count",12)): raise ValueError("token_ids.json must contain exactly dataset.candidate_count experimental values")
partition="gate" if a.calibration else a.partition
count=d.get({"prompt_dev":"prompt_dev_histories","gate":"gate_histories","confirmatory":"n_histories"}[partition],192)
seed=c["seed"]+{"prompt_dev":1000003,"gate":2000003,"confirmatory":0}[partition]
histories=make_histories(count,seed,values,d["variables"],partition=partition)
for h in histories: h["candidate_values"]=values
histories=grouped_split(histories,d.get("splits",[.7,.15,.15]),seed)
variants=PROMPT_VARIANTS if a.all_prompt_variants else (a.prompt_variant,)
histories=[{**h,"prompt_variant":variant} for h in histories for variant in variants]
if a.kind=="histories": rows=histories
elif a.kind=="queries": rows=[q|{"split":h["split"]} for h in histories for q in expand_history_queries(h)]
else: rows=[r|{"split":h["split"]} for h in histories for r in matched_history_pairs(h,d.get("replacement_offset"))]
write_jsonl(rows,a.output); save_json(provenance(c,a.output),a.output+".provenance.json")
print(f"wrote {len(rows)} {a.kind} from {len(histories)} histories to {a.output}")
