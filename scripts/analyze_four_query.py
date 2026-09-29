#!/usr/bin/env python3
"""History-clustered analysis for lexically matched four-query experiments."""
import argparse,collections
import numpy as np
import pandas as pd
from src.data.io import read_jsonl
from src.utils import save_json

def summarize(values,seed=0,n_boot=10000):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return {"n_histories":0}
    rng=np.random.default_rng(seed); boots=np.mean(rng.choice(x,(n_boot,len(x)),replace=True),axis=1)
    return {"n_histories":int(len(x)),"mean":float(x.mean()),"median":float(np.median(x)),
            "trimmed_mean_10pct":float(x[int(.1*len(x)):len(x)-int(.1*len(x))].mean()) if len(x)>=10 else float(x.mean()),
            "fraction_positive":float(np.mean(x>0)),"ci95_cluster_bootstrap":[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
            "sign_flip_permutation_p_two_sided":float((np.sum(np.abs(np.mean(rng.choice([-1,1],(n_boot,len(x)))*x,axis=1))>=abs(x.mean()))+1)/(n_boot+1))}

p=argparse.ArgumentParser(); p.add_argument("--behavior",required=True,help="unmodified four-query behavior JSONL"); p.add_argument("--pairs",required=True,help="matched baseline/edit JSONL"); p.add_argument("--output-dir",required=True); p.add_argument("--seed",type=int,default=20260929); a=p.parse_args()
from pathlib import Path
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
behavior=read_jsonl(a.behavior); pair_rows=read_jsonl(a.pairs)
# Query-only obsolete identity activation contrast, one value per history.
qmap={(r["history_id"],r["query_id"]):r for r in behavior}
activation=[]
for hid in sorted({r["history_id"] for r in behavior}):
    x=qmap[(hid,"current_x")]["candidate_logits"]; z=qmap[(hid,"current_z")]["candidate_logits"]
    activation.append({"history_id":hid,"obsolete_identity_query_contrast":(x[qmap[(hid,"current_x")]["roles"]["old_x"]]-x[qmap[(hid,"current_x")]["roles"]["old_z"]])-(z[qmap[(hid,"current_z")]["roles"]["old_x"]]-z[qmap[(hid,"current_z")]["roles"]["old_z"]])})
pd.DataFrame(activation).to_csv(out/"query_identity_contrast.csv",index=False)

grouped=collections.defaultdict(dict)
for r in pair_rows: grouped[r["pair_id"]][int(r["pair_direction"])]=r
effects=[]
for pair_id,m in grouped.items():
    if set(m)!={0,1}: raise ValueError(f"pair {pair_id} must contain baseline and edited records")
    base,edit=m[0],m[1]; source=base["source_value"]; target=base["replacement_value"]
    lb=base["candidate_logits"]; le=edit["candidate_logits"]
    identity=(le[target]-le[source])-(lb[target]-lb[source])
    answer=edit["roles"]["target"]
    effects.append({"history_id":base["history_id"],"pair_id":pair_id,"edited_binding":base["edited_binding"],"query_id":base["query_id"],
       "identity_transfer":float(identity),"target_logit_change":float(le[answer]-lb[answer]),
       "current_answer_vs_old_margin_change":float((le[answer]-le[source])-(lb[answer]-lb[source])),
       "old_source_logit_change":float(le[source]-lb[source]),"replacement_logit_change":float(le[target]-lb[target]),
       "correct_candidate_probability_change":float(edit["candidate_probabilities"][answer]-base["candidate_probabilities"][base["roles"]["target"]]),
       "correct_full_vocab_rank_before":int(base["full_vocab_rank"]),"correct_full_vocab_rank_after":int(edit["full_vocab_rank"]),"correct_full_vocab_rank_change":int(edit["full_vocab_rank"]-base["full_vocab_rank"]),
       "greedy_correct_before":int(base["full_vocab_accuracy"]),"greedy_correct_after":int(edit["full_vocab_accuracy"])})
ef=pd.DataFrame(effects); ef.to_csv(out/"matched_edit_effects.csv",index=False)

def per_history(frame,metric,query=None,binding=None):
    f=frame
    if query is not None: f=f[f.query_id==query]
    if binding is not None: f=f[f.edited_binding==binding]
    return f.groupby("history_id")[metric].mean().to_dict()
results={"unmodified_query_identity_current_x_minus_current_z":summarize([r["obsolete_identity_query_contrast"] for r in activation],a.seed),"matched_edits":{}}
for binding in ("old_x","old_z","current_x","current_z"):
    results["matched_edits"][binding]={}
    for query in ("current_x","initial_x","current_z","initial_z"):
        values=per_history(ef,"identity_transfer",query,binding)
        results["matched_edits"][binding][query]=summarize(list(values.values()),a.seed)
for binding in ("old_x","old_z"):
    x=per_history(ef,"identity_transfer","current_x",binding); z=per_history(ef,"identity_transfer","current_z",binding)
    common=sorted(set(x)&set(z)); diffs=[x[h]-z[h] for h in common]
    results["matched_edits"][binding]["current_x_minus_current_z"]={**summarize(diffs,a.seed),"contrast":"same source token, replacement, assignment position, and history; query changes only"}
    if binding=="old_x":
        now=per_history(ef,"target_logit_change","current_x",binding)
        results["matched_edits"][binding]["current_x_correct_answer_logit_change"]=summarize(list(now.values()),a.seed)
        margin=per_history(ef,"current_answer_vs_old_margin_change","current_x",binding)
        results["matched_edits"][binding]["current_x_answer_vs_old_margin_change"]=summarize(list(margin.values()),a.seed)
        other=per_history(ef,"target_logit_change","current_z",binding); common=sorted(set(now)&set(other))
        results["matched_edits"][binding]["correct_answer_logit_change_current_x_minus_current_z"]=summarize([now[h]-other[h] for h in common],a.seed)
        rank=per_history(ef,"correct_full_vocab_rank_change","current_x",binding)
        results["matched_edits"][binding]["current_x_correct_full_vocab_rank_change"]=summarize(list(rank.values()),a.seed)
save_json(results,out/"four_query_summary.json")
print(f"wrote history-level summaries for {len(activation)} histories to {out}")
