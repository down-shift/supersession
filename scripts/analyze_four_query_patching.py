#!/usr/bin/env python3
"""History-clustered, donor-oriented analysis of four-query patch records."""
import argparse,collections
import numpy as np
import pandas as pd
from src.data.io import read_jsonl
from src.utils import save_json
from src.analysis.metrics import trimmed_mean

def summary(values,seed=0,n_boot=10000):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return {"n_histories":0}
    rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(x,(n_boot,len(x)),replace=True),axis=1)
    return {"n_histories":len(x),"mean":float(x.mean()),"median":float(np.median(x)),"trimmed_mean_10pct":trimmed_mean(x,.1),"fraction_positive":float(np.mean(x>0)),"ci95_cluster_bootstrap":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}

p=argparse.ArgumentParser(); p.add_argument("--patches",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--seed",type=int,default=20260929); a=p.parse_args()
from pathlib import Path
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=read_jsonl(a.patches)
required={"history_id","query_id","edited_binding","direction","site","layer","patch_delta_toward_donor"}
if rows and not required.issubset(rows[0]): raise ValueError(f"patch records missing fields: {sorted(required-set(rows[0]))}")
# Average multi-token sites within direction first; directions already use each
# donor's orientation, so they can then be averaged without sign reversal.
directional=collections.defaultdict(list)
for r in rows: directional[(r["history_id"],r["edited_binding"],r["query_id"],r["site"],r["layer"],r["direction"])].append(float(r["patch_delta_toward_donor"]))
direct_mean={k:float(np.mean(v)) for k,v in directional.items()}
paired=collections.defaultdict(dict)
for (hid,binding,query,site,layer,direction),value in direct_mean.items(): paired[(hid,binding,query,site,layer)][direction]=value
effects=[]
for (hid,binding,query,site,layer),dirs in paired.items():
    if set(dirs)!={"donor_to_recipient","recipient_to_donor"}: raise ValueError(f"missing a patch direction for {(hid,binding,query,site,layer)}")
    effects.append({"history_id":hid,"edited_binding":binding,"query_id":query,"site":site,"layer":layer,"mean_donor_oriented_patch_delta":float(np.mean(list(dirs.values())))})
ef=pd.DataFrame(effects); ef.to_csv(out/"patch_effects_by_history.csv",index=False)
results={}
for (site,layer),frame in ef.groupby(["site","layer"]):
    cell={}
    for binding in ("old_x","current_x","old_z","current_z"):
        for query in ("current_x","initial_x","current_z","initial_z"):
            part=frame[(frame.edited_binding==binding)&(frame.query_id==query)]
            cell[(binding,query)]=part.groupby("history_id").mean_donor_oriented_patch_delta.mean().to_dict()
    common=set.intersection(*(set(v) for v in cell.values()))
    ids=sorted(common)
    sx=[.5*((cell[("old_x","initial_x")][h]-cell[("old_x","current_x")][h])+(cell[("current_x","current_x")][h]-cell[("current_x","initial_x")][h])) for h in ids]
    sz=[.5*((cell[("old_z","initial_z")][h]-cell[("old_z","current_z")][h])+(cell[("current_z","current_z")][h]-cell[("current_z","initial_z")][h])) for h in ids]
    results[f"{site}/layer_{layer}"]={"histories_complete":len(ids),"matrix":{b:{q:summary(list(cell[(b,q)].values()),a.seed) for q in ("current_x","initial_x","current_z","initial_z")} for b in ("old_x","current_x","old_z","current_z")},"temporal_double_dissociation":{"S_x":summary(sx,a.seed),"S_z":summary(sz,a.seed),"mean_S_x_S_z":summary([.5*(x+z) for x,z in zip(sx,sz)],a.seed)}}
save_json({"results":results,"orientation":"both directions are converted to patch_delta_toward_donor before history-level averaging","bootstrap_unit":"history_id","normalized_recovery":"secondary; raw donor-oriented effects are primary"},out/"four_query_patch_summary.json")
print(f"wrote donor-oriented patch summaries for {len(results)} site/layer cells to {out}")
