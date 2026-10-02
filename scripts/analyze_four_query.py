#!/usr/bin/env python3
"""History-clustered analysis for lexically matched four-query experiments."""
import argparse,collections
import numpy as np
import pandas as pd
from src.data.io import read_jsonl
from src.utils import save_json
from src.analysis.metrics import trimmed_mean

def summarize(values,seed=0,n_boot=10000,total_histories=None):
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).all(): raise ValueError("nonfinite history statistic; refusing to drop histories")
    if n_boot<1: raise ValueError("bootstrap draws must be positive")
    if not len(x): return {"n_histories":0,"fraction_of_confirmatory_histories":0.0 if total_histories else None}
    rng=np.random.default_rng(seed); boots=np.mean(rng.choice(x,(n_boot,len(x)),replace=True),axis=1)
    return {"n_histories":int(len(x)),"fraction_of_confirmatory_histories":float(len(x)/total_histories) if total_histories else None,
            "mean":float(x.mean()),"median":float(np.median(x)),"trimmed_mean_10pct":trimmed_mean(x,.1),
            "fraction_positive":float(np.mean(x>0)),"ci95_cluster_bootstrap":[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],
            "sign_flip_permutation_p_two_sided":float((np.sum(np.abs(np.mean(rng.choice([-1,1],(n_boot,len(x)))*x,axis=1))>=abs(x.mean()))+1)/(n_boot+1))}

def per_history(frame,metric,query=None,binding=None):
    f=frame
    if query is not None: f=f[f.query_id==query]
    if binding is not None: f=f[f.edited_binding==binding]
    dimensions=[c for c in ("edited_binding","query_id") if c in f.columns]
    if dimensions and f.duplicated(["history_id",*dimensions]).any():
        raise ValueError("duplicate history-level binding/query cell")
    return f.groupby("history_id")[metric].mean().to_dict()

def temporal_results(frame,total_histories,seed):
    """Return edit matrix, version-selection controls, and relevance contrasts."""
    bindings=("old_x","current_x","old_z","current_z")
    queries=("current_x","initial_x","current_z","initial_z")
    cell={(b,q):per_history(frame,"identity_transfer",q,b) for b in bindings for q in queries}
    matrix={b:{q:summarize(list(cell[(b,q)].values()),seed,total_histories=total_histories) for q in queries} for b in bindings}
    def contrast(keys,fn):
        ids=sorted(set.intersection(*(set(cell[k]) for k in keys)))
        vals=[fn(*(cell[k][h] for k in keys)) for h in ids]
        return summarize(vals,seed,total_histories=total_histories)
    sx=contrast((("old_x","initial_x"),("old_x","current_x"),("current_x","current_x"),("current_x","initial_x")),lambda oi,oc,cc,ci:.5*((oi-oc)+(cc-ci)))
    sz=contrast((("old_z","initial_z"),("old_z","current_z"),("current_z","current_z"),("current_z","initial_z")),lambda oi,oc,cc,ci:.5*((oi-oc)+(cc-ci)))
    relevance=contrast((("old_x","current_x"),("old_x","current_z"),("old_z","current_z"),("old_z","current_x")),lambda xc,xz,zc,zx:.5*((xc-xz)+(zc-zx)))
    return {"matrix":matrix,"S_x":sx,"S_z":sz,"mean_S_x_S_z":summarize(
        _paired_values(frame) ,seed,total_histories=total_histories),
        "symmetric_obsolete_query_relevance":relevance}

def _paired_values(frame):
    # Recompute paired history-level scores; marginal summaries cannot be averaged
    # to obtain a paired estimate or its uncertainty.
    vals={}
    for h in frame.history_id.unique():
        rows=frame[frame.history_id==h]
        e={(r.edited_binding,r.query_id):float(r.identity_transfer) for r in rows.itertuples()}
        needed=(("old_x","initial_x"),("old_x","current_x"),("current_x","current_x"),("current_x","initial_x"),
                ("old_z","initial_z"),("old_z","current_z"),("current_z","current_z"),("current_z","initial_z"))
        if all(k in e for k in needed):
            vals[h]=.5*(.5*((e[("old_x","initial_x")]-e[("old_x","current_x")])+(e[("current_x","current_x")]-e[("current_x","initial_x")]))+
                         .5*((e[("old_z","initial_z")]-e[("old_z","current_z")])+(e[("current_z","current_z")]-e[("current_z","initial_z")])) )
    return list(vals.values())

def competence_table(rows,keys):
    buckets=collections.defaultdict(list)
    for r in rows:
        buckets[tuple(tuple(r[k]) if k in ("order","variable_pair") else r[k] for k in keys)].append(r)
    return [{"group":dict(zip(keys,key)),"n":len(v),"candidate_accuracy":float(np.mean([x["accuracy"] for x in v])),
             "full_vocab_next_token_accuracy":float(np.mean([x["full_vocab_next_token_accuracy"] for x in v]))}
            for key,v in sorted(buckets.items(),key=lambda x:str(x[0]))]

p=argparse.ArgumentParser(); p.add_argument("--behavior",required=True,help="unmodified four-query behavior JSONL"); p.add_argument("--pairs",required=True,help="matched baseline/edit JSONL"); p.add_argument("--output-dir",required=True); p.add_argument("--seed",type=int,default=20260929); a=p.parse_args()
from pathlib import Path
out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
behavior=read_jsonl(a.behavior); pair_rows=read_jsonl(a.pairs)
history_ids={r["history_id"] for r in behavior}; total_histories=len(history_ids)
if len(behavior)!=4*total_histories or any(sum(r["history_id"]==h for r in behavior)!=4 for h in history_ids):
    raise ValueError("confirmatory behavior must contain exactly four queries per history")
if len({(r["history_id"],r["query_id"]) for r in behavior})!=len(behavior):
    raise ValueError("duplicate history/query behavior record")
for hid in history_ids:
    if {r["query_id"] for r in behavior if r["history_id"]==hid}!={"current_x","current_z","initial_x","initial_z"}:
        raise ValueError(f"history {hid} has incomplete four-query behavior")

# Unedited identity activity is secondary; the matched edit contrast below is primary.
qmap={(r["history_id"],r["query_id"]):r for r in behavior}; activation=[]
for hid in sorted(history_ids):
    x=qmap[(hid,"current_x")]; z=qmap[(hid,"current_z")]
    activation.append({"history_id":hid,"x_relevant_minus_irrelevant":x["candidate_logits"][x["roles"]["old_x"]]-x["candidate_logits"][x["roles"]["old_z"]],
                       "z_relevant_minus_irrelevant":z["candidate_logits"][z["roles"]["old_z"]]-z["candidate_logits"][z["roles"]["old_x"]]})
pd.DataFrame(activation).to_csv(out/"query_identity_contrast.csv",index=False)

grouped=collections.defaultdict(dict)
for r in pair_rows:
    direction=r.get("pair_direction")
    if direction not in (0,1) or direction in grouped[r["pair_id"]]:
        raise ValueError(f"invalid or duplicate pair direction: {r.get('pair_id')}")
    grouped[r["pair_id"]][direction]=r
effects=[]
for pair_id,m in grouped.items():
    if set(m)!={0,1}: raise ValueError(f"pair {pair_id} must contain baseline and edited records")
    base,edit=m[0],m[1]; source=base["source_value"]; target=base["replacement_value"]
    for key in ("history_id","edited_binding","query_id","source_value","replacement_value","matching_values","variables","orientation"):
        if base.get(key)!=edit.get(key): raise ValueError(f"pair {pair_id} metadata differs: {key}")
    if source==target: raise ValueError(f"pair {pair_id} source equals replacement")
    if set(base["candidate_logits"])!=set(edit["candidate_logits"]): raise ValueError(f"pair {pair_id} candidate sets differ")
    if not np.isfinite(list(base["candidate_logits"].values())+list(edit["candidate_logits"].values())).all():
        raise ValueError(f"pair {pair_id} has nonfinite candidate logits")
    lb=base["candidate_logits"]; le=edit["candidate_logits"]; answer=edit["roles"]["target"]
    effects.append({"history_id":base["history_id"],"pair_id":pair_id,"edited_binding":base["edited_binding"],"query_id":base["query_id"],
       "identity_transfer":float((le[target]-le[source])-(lb[target]-lb[source])),"target_logit_change":float(le[answer]-lb[answer]),
       "current_answer_vs_old_margin_change":float((le[answer]-le[source])-(lb[answer]-lb[source])),
       "old_source_logit_change":float(le[source]-lb[source]),"replacement_logit_change":float(le[target]-lb[target]),
       "correct_candidate_probability_change":float(edit["candidate_probabilities"][answer]-base["candidate_probabilities"][base["roles"]["target"]]),
       "correct_full_vocab_rank_before":int(base["full_vocab_rank"]),"correct_full_vocab_rank_after":int(edit["full_vocab_rank"]),
       "correct_full_vocab_rank_change":int(edit["full_vocab_rank"]-base["full_vocab_rank"]),
       "full_vocab_next_token_correct_before":int(base["full_vocab_next_token_accuracy"]),"full_vocab_next_token_correct_after":int(edit["full_vocab_next_token_accuracy"]),
       "pair_both_full_vocab_next_token_correct":int(base["full_vocab_next_token_accuracy"] and edit["full_vocab_next_token_accuracy"])})
required={(h,b,q) for h in history_ids for b in ("old_x","old_z","current_x","current_z") for q in ("current_x","initial_x","current_z","initial_z")}
observed={(r["history_id"],r["edited_binding"],r["query_id"]) for r in effects}
if observed!=required or len(effects)!=len(required): raise ValueError("matched edit effects do not form the complete unique history/binding/query grid")
ef=pd.DataFrame(effects); ef.to_csv(out/"matched_edit_effects.csv",index=False)

all_temporal=temporal_results(ef,total_histories,a.seed)
results={"primary_supersession_residual":all_temporal["symmetric_obsolete_query_relevance"],
         "retention_positive_control":all_temporal["matrix"]["old_x"]["initial_x"],
         "current_state_positive_control":all_temporal["matrix"]["current_x"]["current_x"],
         "irrelevant_history_baseline":{"old_x_edit_while_asking_current_z":all_temporal["matrix"]["old_x"]["current_z"],"old_z_edit_while_asking_current_x":all_temporal["matrix"]["old_z"]["current_x"]},
         "temporal_double_dissociation_sanity_control":all_temporal,"matched_edits":{}}
for binding in ("old_x","old_z","current_x","current_z"):
    results["matched_edits"][binding]={query:all_temporal["matrix"][binding][query] for query in ("current_x","initial_x","current_z","initial_z")}
for binding in ("old_x","old_z"):
    x=per_history(ef,"identity_transfer","current_x",binding); z=per_history(ef,"identity_transfer","current_z",binding)
    common=sorted(set(x)&set(z)); results["matched_edits"][binding]["relevant_minus_irrelevant"]={**summarize([x[h]-z[h] if binding=="old_x" else z[h]-x[h] for h in common],a.seed,total_histories=total_histories),"contrast":"orientation is relevant query minus irrelevant query"}
results["unmodified_obsolete_identity_query_contrast"]={"x_relevant_minus_irrelevant":summarize([r["x_relevant_minus_irrelevant"] for r in activation],a.seed,total_histories=total_histories),"z_relevant_minus_irrelevant":summarize([r["z_relevant_minus_irrelevant"] for r in activation],a.seed,total_histories=total_histories),"symmetric":summarize([.5*(r["x_relevant_minus_irrelevant"]+r["z_relevant_minus_irrelevant"]) for r in activation],a.seed,total_histories=total_histories)}

# The same primary contrasts are recomputed after filtering pairs, only as sensitivity.
correct=ef[ef.pair_both_full_vocab_next_token_correct==1].copy()
correct_temporal=temporal_results(correct,total_histories,a.seed)
results["correct_only_sensitivity"]={**correct_temporal,"n_pair_effects":len(correct),"fraction_pair_effects":float(len(correct)/len(ef)) if len(ef) else 0.0,
                                     "interpretation":"mirrors the full primary analyses; sensitivity only, never the primary filtering rule"}

results["confirmatory_competence"]={"by_query":competence_table(behavior,["query_id"]),
    "by_order":competence_table(behavior,["order"]),"by_variable_pair":competence_table(behavior,["variable_pair"]),
    "by_orientation":competence_table(behavior,["orientation"]),
    "query_by_order_by_pair_by_orientation":competence_table(behavior,["query_id","order","variable_pair","orientation"])}
pair_comp=collections.defaultdict(dict)
for r in pair_rows: pair_comp[r["pair_id"]][int(r["pair_direction"])]=r
complete=[(m[0],m[1]) for m in pair_comp.values() if set(m)=={0,1}]
both_rows=[{"query_id":base["query_id"],"edited_binding":base["edited_binding"],"accuracy":int(base["full_vocab_next_token_accuracy"] and edit["full_vocab_next_token_accuracy"]),"full_vocab_next_token_accuracy":int(base["full_vocab_next_token_accuracy"] and edit["full_vocab_next_token_accuracy"])} for base,edit in complete]
results["confirmatory_pair_competence"]={"baseline_edit_by_query_and_binding":competence_table(pair_rows,["query_id","edited_binding","pair_direction"]),
    "both_members_correct_by_query_and_binding":competence_table(both_rows,["query_id","edited_binding"]),
    "both_members_full_vocab_next_token_correct_rate":float(np.mean([x["full_vocab_next_token_accuracy"] for x in both_rows])) if both_rows else None,"n_pairs":len(complete)}
save_json(results,out/"four_query_summary.json")
print(f"wrote history-level summaries for {total_histories} histories to {out}")
