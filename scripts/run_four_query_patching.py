#!/usr/bin/env python3
"""Targeted residual patching of a matched old-x edit across all four queries."""
import argparse,collections,json
import numpy as np
from tqdm.auto import tqdm
from src.data.io import read_jsonl,write_jsonl
from src.data.generate import render_example
from src.experiments.patching import patch_sweep
from src.models.loader import load_model
from src.utils import load_config,provenance,save_json

p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/four_query_pilot.yaml"); p.add_argument("--pairs",required=True); p.add_argument("--token-ids",required=True); p.add_argument("--output",required=True); p.add_argument("--stage",choices=("discovery","heldout"),default="discovery"); p.add_argument("--n-histories",type=int,default=None); p.add_argument("--discovery-histories",type=int,default=24); p.add_argument("--seed",type=int,default=20260929); p.add_argument("--layers",default=None,help="comma-separated frozen layer indices; required for heldout stage"); p.add_argument("--all-positions",action="store_true",help="exhaustive sweep, discovery stage only, at most 24 histories"); a=p.parse_args()
layers=[int(x) for x in a.layers.split(",")] if a.layers else None
if a.stage=="heldout" and layers is None: raise ValueError("heldout patching requires --layers frozen from discovery")
if a.all_positions and a.stage!="discovery": raise ValueError("exhaustive position sweeps are restricted to discovery")
c=load_config(a.config); model,tok=load_model(c); token_ids=json.load(open(a.token_ids))["token_ids"]
pairs=collections.defaultdict(dict)
for ex in read_jsonl(a.pairs):
    if ex["edited_binding"] in ("old_x","current_x"): pairs[ex["pair_id"]][int(ex["pair_direction"])]=ex
selected=collections.defaultdict(dict)
for pair,members in pairs.items():
    if set(members)=={0,1}: selected[members[0]["history_id"]][(members[0]["edited_binding"],members[0]["query_id"])]=members
output=[]
history_ids=sorted(selected); np.random.default_rng(a.seed).shuffle(history_ids)
cut=a.discovery_histories; available=history_ids[:cut] if a.stage=="discovery" else history_ids[cut:]
limit=a.n_histories if a.n_histories is not None else (min(24,len(available)) if a.stage=="discovery" else min(96,len(available)))
if a.all_positions and limit>24: raise ValueError("exhaustive position sweeps are limited to 24 discovery histories")
for hid in tqdm(available[:limit],desc=f"Query-conditioned {a.stage} patches"):
    queries=selected[hid]
    for (binding,query_id),members in queries.items():
        for direction,(donor,recipient) in enumerate(((members[0],members[1]),(members[1],members[0]))):
            text=render_example(donor,tok,chat=c["model"].get("chat_template",True)); enc=tok(text,add_special_tokens=False,return_offsets_mapping=True); offsets=enc["offset_mapping"]
            def span_positions(start,end): return [i for i,(lo,hi) in enumerate(offsets) if lo<end and hi>start]
            old=donor["old_x"]; old_start=text.index(old); current=donor["current_x"]; curr_start=text.index(current)
            if binding=="old_x": sites={"edited_old_x_value":span_positions(old_start,old_start+len(old)),"current_x_value":span_positions(curr_start,curr_start+len(current))}
            else: sites={"old_x_value":span_positions(old_start,old_start+len(old)),"edited_current_x_value":span_positions(curr_start,curr_start+len(current))}
            qvar=donor["variables"][0]; qtext=("initial value of "+qvar if donor["query_time"]=="initial" else "current value of "+qvar); qstart=text.rfind(qtext)
            if qstart<0: raise ValueError(f"cannot locate query variable for {query_id}")
            vstart=qstart+len(qtext)-len(qvar); sites["query_variable"]=span_positions(vstart,vstart+len(qvar)); sites["final_preanswer"]=[len(enc["input_ids"])-1]
            positions=sorted(set(p for group in sites.values() for p in group))
            table=patch_sweep(model,tok,donor,recipient,token_ids,chat=c["model"].get("chat_template",True),positions=None if a.all_positions else positions,layers=layers)
            name_by_position={p:name for name,ps in sites.items() for p in ps}
            for row in table:
                row.update({"history_id":hid,"query_id":query_id,"edited_binding":binding,"direction":"donor_to_recipient" if direction==0 else "recipient_to_donor","site":name_by_position.get(row["position"],"other"),"donor_value":donor[binding],"recipient_value":recipient[binding],"metric_orientation":"toward activation donor"}); output.append(row)
write_jsonl(output,a.output); save_json({"provenance":provenance(c,a.pairs),"design":"old-x historical edit and current-x positive-control donor patches compared across matched query variants","stage":a.stage,"all_positions":a.all_positions,"frozen_layers":layers,"discovery_histories":cut,"n_histories":len({r["history_id"] for r in output}),"sites":["edited assignment value","other x assignment value","query_variable","final_preanswer"]},a.output+".provenance.json")
print(f"saved {len(output)} targeted patch records")
