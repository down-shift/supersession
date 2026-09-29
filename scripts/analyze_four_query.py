#!/usr/bin/env python3
"""History-clustered analysis for lexically matched four-query experiments."""
import argparse,collections
import numpy as np
import pandas as pd
from src.data.io import read_jsonl
from src.utils import save_json
from src.analysis.metrics import trimmed_mean

def summarize(values,seed=0,n_boot=10000):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return {"n_histories":0}
    rng=np.random.default_rng(seed); boots=np.mean(rng.choice(x,(n_boot,len(x)),replace=True),axis=1)
    return {"n_histories":int(len(x)),"mean":float(x.mean()),"median":float(np.median(x)),
            "trimmed_mean_10pct":trimmed_mean(x,.1),
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
    roles_x=qmap[(hid,"current_x")]["roles"]; roles_z=qmap[(hid,"current_z")]["roles"]
    activation.append({"history_id":hid,"obsolete_identity_query_contrast_x":x[roles_x["old_x"]]-x[roles_x["old_z"]],"obsolete_identity_query_contrast_z":z[roles_z["old_z"]]-z[roles_z["old_x"]]})
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
       "greedy_correct_before":int(base["full_vocab_accuracy"]),"greedy_correct_after":int(edit["full_vocab_accuracy"]),"pair_both_greedy_correct":int(base["full_vocab_accuracy"] and edit["full_vocab_accuracy"])})
ef=pd.DataFrame(effects); ef.to_csv(out/"matched_edit_effects.csv",index=False)

def per_history(frame,metric,query=None,binding=None):
    f=frame
    if query is not None: f=f[f.query_id==query]
    if binding is not None: f=f[f.edited_binding==binding]
    return f.groupby("history_id")[metric].mean().to_dict()
results={"matched_edits":{}}
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

# Full temporal double-dissociation matrix, paired within history.
matrix={}; cell={}
for binding in ("old_x","current_x","old_z","current_z"):
    matrix[binding]={};
    for query in ("current_x","initial_x","current_z","initial_z"):
        values=per_history(ef,"identity_transfer",query,binding); cell[(binding,query)]=values
        matrix[binding][query]=summarize(list(values.values()),a.seed)
results["primary_temporal_identity_transfer_matrix"]=matrix
ids=sorted(set.intersection(*(set(cell[k]) for k in cell)))
sx=[.5*((cell[("old_x","initial_x")][h]-cell[("old_x","current_x")][h])+(cell[("current_x","current_x")][h]-cell[("current_x","initial_x")][h])) for h in ids]
sz=[.5*((cell[("old_z","initial_z")][h]-cell[("old_z","current_z")][h])+(cell[("current_z","current_z")][h]-cell[("current_z","initial_z")][h])) for h in ids]
results["temporal_double_dissociation"]={"S_x":summarize(sx,a.seed),"S_z":summarize(sz,a.seed),"mean_S_x_S_z":summarize([.5*(x+z) for x,z in zip(sx,sz)],a.seed),"definition":"S_v = 0.5[(E(old_v,initial_v)-E(old_v,current_v))+(E(current_v,current_v)-E(current_v,initial_v))]"}
relevance=[.5*((cell[("old_x","current_x")][h]-cell[("old_x","current_z")][h])+(cell[("old_z","current_z")][h]-cell[("old_z","current_x")][h])) for h in ids]
results["symmetric_obsolete_query_relevance"]=summarize(relevance,a.seed)
results["unmodified_obsolete_identity_query_contrast"]={"x_relevant_minus_z_relevant":summarize([r["obsolete_identity_query_contrast_x"] for r in activation],a.seed),"z_relevant_minus_x_relevant":summarize([r["obsolete_identity_query_contrast_z"] for r in activation],a.seed),"symmetric":summarize([.5*(r["obsolete_identity_query_contrast_x"]+r["obsolete_identity_query_contrast_z"]) for r in activation],a.seed)}

# Confirmatory competence is reported by query and design factors; no correct-only filtering.
def competence_table(rows,keys):
    buckets=collections.defaultdict(list)
    for r in rows: buckets[tuple(tuple(r[k]) if k=="order" else tuple(r[k]) if k=="variable_pair" else r[k] for k in keys)].append(r)
    return [{"group":dict(zip(keys,key)),"n":len(v),"candidate_accuracy":float(np.mean([x["accuracy"] for x in v])),"unrestricted_greedy_accuracy":float(np.mean([x["full_vocab_accuracy"] for x in v]))} for key,v in sorted(buckets.items(),key=lambda x:str(x[0]))]
results["confirmatory_competence"]={"by_query":competence_table(behavior,["query_id"]),"by_query_order_pair_orientation":competence_table(behavior,["query_id","order","variable_pair","orientation"])}
pair_comp=collections.defaultdict(dict)
for r in pair_rows: pair_comp[r["pair_id"]][int(r["pair_direction"])]=r
complete=[(m[0],m[1]) for m in pair_comp.values() if set(m)=={0,1}]
both_rows=[]
for base,edit in complete:
    both_rows.append({"query_id":base["query_id"],"edited_binding":base["edited_binding"],"accuracy":int(base["full_vocab_accuracy"] and edit["full_vocab_accuracy"]),"full_vocab_accuracy":int(base["full_vocab_accuracy"] and edit["full_vocab_accuracy"])})
results["confirmatory_pair_competence"]={"baseline_edit_by_query_and_binding":competence_table(pair_rows,["query_id","edited_binding","pair_direction"]),"both_members_correct_by_query_and_binding":competence_table(both_rows,["query_id","edited_binding"]),"both_members_greedy_correct_rate":float(np.mean([x["full_vocab_accuracy"] and y["full_vocab_accuracy"] for x,y in complete])) if complete else None,"n_pairs":len(complete)}
correct_only=[r for r in effects if r["greedy_correct_before"] and r["greedy_correct_after"]]
results["correct_only_sensitivity"]={"n_pair_effects":len(correct_only),"mean_identity_transfer":float(np.mean([r["identity_transfer"] for r in correct_only])) if correct_only else None,"interpretation":"sensitivity only; primary analyses include all histories"}
save_json(results,out/"four_query_summary.json")
print(f"wrote history-level summaries for {len(activation)} histories to {out}")
