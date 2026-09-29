#!/usr/bin/env python3
"""Analyze discovery-x or held-out-x/z donor-oriented patch records."""
import argparse,collections
import numpy as np
import pandas as pd
from src.data.io import read_jsonl
from src.utils import save_json
from src.analysis.metrics import trimmed_mean

def summary(values,seed=0,n_boot=10000):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return {"n_histories":0,"mean":None,"ci95_cluster_bootstrap":None}
    rng=np.random.default_rng(seed); boot=np.mean(rng.choice(x,(n_boot,len(x)),replace=True),axis=1)
    return {"n_histories":len(x),"mean":float(x.mean()),"median":float(np.median(x)),"trimmed_mean_10pct":trimmed_mean(x,.1),
            "fraction_positive":float(np.mean(x>0)),"ci95_cluster_bootstrap":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}

def cells_for(frame,bindings):
    return {(b,q):frame[(frame.edited_binding==b)&(frame.query_id==q)].groupby("history_id").effect.mean().to_dict()
            for b in bindings for q in ("current_x","initial_x","current_z","initial_z")}

def temporal_contrasts(cell,bindings):
    ids_x=sorted(set.intersection(*(set(cell[k]) for k in (("old_x","initial_x"),("old_x","current_x"),("current_x","current_x"),("current_x","initial_x"))))) if {"old_x","current_x"}.issubset(bindings) else []
    sx=[.5*((cell[("old_x","initial_x")][h]-cell[("old_x","current_x")][h])+(cell[("current_x","current_x")][h]-cell[("current_x","initial_x")][h])) for h in ids_x]
    result={"S_x":summary(sx),"S_x_complete_histories":len(ids_x)}
    if {"old_z","current_z"}.issubset(bindings):
        ids_z=sorted(set.intersection(*(set(cell[k]) for k in (("old_z","initial_z"),("old_z","current_z"),("current_z","current_z"),("current_z","initial_z")))))
        sz=[.5*((cell[("old_z","initial_z")][h]-cell[("old_z","current_z")][h])+(cell[("current_z","current_z")][h]-cell[("current_z","initial_z")][h])) for h in ids_z]
        ids_both=sorted(set(ids_x)&set(ids_z)); both=[.5*(
            .5*((cell[("old_x","initial_x")][h]-cell[("old_x","current_x")][h])+(cell[("current_x","current_x")][h]-cell[("current_x","initial_x")][h]))+
            .5*((cell[("old_z","initial_z")][h]-cell[("old_z","current_z")][h])+(cell[("current_z","current_z")][h]-cell[("current_z","initial_z")][h])) ) for h in ids_both]
        ids_rel=sorted(set.intersection(*(set(cell[k]) for k in (("old_x","current_x"),("old_x","current_z"),("old_z","current_z"),("old_z","current_x")))))
        relevance=[.5*((cell[("old_x","current_x")][h]-cell[("old_x","current_z")][h])+(cell[("old_z","current_z")][h]-cell[("old_z","current_x")][h])) for h in ids_rel]
        result.update({"S_z":summary(sz),"S_z_complete_histories":len(ids_z),"mean_S_x_S_z":summary(both),"both_complete_histories":len(ids_both),
                       "symmetric_obsolete_query_relevance":summary(relevance),"relevance_complete_histories":len(ids_rel)})
    return result

p=argparse.ArgumentParser(); p.add_argument("--patches",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--stage",choices=("discovery","heldout"),required=True); p.add_argument("--seed",type=int,default=20260929); p.add_argument("--selection-site",default="final_preanswer"); p.add_argument("--region-width",type=int,choices=(3,),default=3); a=p.parse_args()
from pathlib import Path
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=read_jsonl(a.patches)
required={"history_id","query_id","edited_binding","direction","site","layer","patch_delta_toward_donor"}
if rows and not required.issubset(rows[0]): raise ValueError(f"patch records missing fields: {sorted(required-set(rows[0]))}")
present={r["edited_binding"] for r in rows}; xbindings={"old_x","current_x"}; allbindings=xbindings|{"old_z","current_z"}
if a.stage=="discovery" and not xbindings.issubset(present): raise ValueError("discovery patch data must include old_x and current_x")
if a.stage=="heldout" and not allbindings.issubset(present): raise ValueError("heldout patch data must include old_x/current_x and old_z/current_z")
bindings=sorted(xbindings if a.stage=="discovery" else allbindings)
# Average multi-token sites within each direction. Both directions already use
# donor-relative margins, so only then average them within history.
directional=collections.defaultdict(list)
for r in rows: directional[(r["history_id"],r["edited_binding"],r["query_id"],r["site"],r["layer"],r["direction"])].append(float(r["patch_delta_toward_donor"]))
direct_mean={k:float(np.mean(v)) for k,v in directional.items()}; paired=collections.defaultdict(dict)
for (hid,binding,query,site,layer,direction),value in direct_mean.items(): paired[(hid,binding,query,site,layer)][direction]=value
effects=[]
for (hid,binding,query,site,layer),dirs in paired.items():
    if set(dirs)!={"donor_to_recipient","recipient_to_donor"}: raise ValueError(f"missing patch direction for {(hid,binding,query,site,layer)}")
    effects.append({"history_id":hid,"edited_binding":binding,"query_id":query,"site":site,"layer":layer,"mean_donor_oriented_patch_delta":float(np.mean(list(dirs.values())))})
ef=pd.DataFrame(effects).rename(columns={"mean_donor_oriented_patch_delta":"effect"}); ef.to_csv(out/"patch_effects_by_history.csv",index=False)
results={}
for (site,layer),frame in ef.groupby(["site","layer"]):
    cell=cells_for(frame,bindings)
    results[f"{site}/layer_{layer}"]={"histories_by_cell":{f"{b}/{q}":len(cell[(b,q)]) for b in bindings for q in ("current_x","initial_x","current_z","initial_z")},
        "matrix":{b:{q:summary(list(cell[(b,q)].values()),a.seed) for q in ("current_x","initial_x","current_z","initial_z")} for b in bindings},
        "temporal_contrasts":temporal_contrasts(cell,bindings)}

selection=None
if a.stage=="discovery":
    # Prespecified selection: at final pre-answer residual, choose the contiguous
    # three-layer window with the largest mean history-level S_x; lowest start wins ties.
    frame=ef[ef.site==a.selection_site]; layers=sorted(frame.layer.unique())
    if len(layers)<a.region_width: raise ValueError(f"selection site {a.selection_site!r} has fewer than {a.region_width} layers")
    layer_scores=[]
    for layer in layers:
        cell=cells_for(frame[frame.layer==layer],xbindings)
        stats=temporal_contrasts(cell,xbindings)["S_x"]
        layer_scores.append((int(layer),stats["mean"],stats["n_histories"]))
    by_layer={x[0]:x[1] for x in layer_scores}; windows=[]
    for start_index in range(len(layers)-a.region_width+1):
        window=layers[start_index:start_index+a.region_width]
        if window==list(range(window[0],window[0]+a.region_width)):
            windows.append((float(np.mean([by_layer[int(k)] for k in window])),int(window[0]),[int(k) for k in window]))
    if not windows: raise ValueError("no contiguous three-layer region at the selection site")
    best=max(windows,key=lambda x:(x[0],-x[1]))
    selection={"site":a.selection_site,"statistic":"mean history-level S_x","fixed_region_width":a.region_width,"per_layer_S_x":layer_scores,"selected_layers":best[2],"selected_region_mean_S_x":best[0],"tie_break":"lowest starting layer"}

save_json({"stage":a.stage,"bindings":bindings,"results":results,"prespecified_discovery_layer_selection":selection,
           "orientation":"both directions are converted to patch_delta_toward_donor before history-level averaging","bootstrap_unit":"history_id","normalized_recovery":"secondary; raw donor-oriented effects are primary"},out/"four_query_patch_summary.json")
print(f"wrote {a.stage} donor-oriented patch summaries for {len(results)} site/layer cells to {out}")
