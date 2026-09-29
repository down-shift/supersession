#!/usr/bin/env python3
"""Aggregate raw behavior into tables, paired effects, and basic figures."""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.io import read_jsonl
from src.analysis.metrics import bootstrap_mean_ci,js_divergence
from src.utils import save_json
p=argparse.ArgumentParser(); p.add_argument("--behavior",default="outputs/pilot/behavior.jsonl"); p.add_argument("--pairs",default=None); p.add_argument("--output-dir",default="outputs/pilot/analysis"); p.add_argument("--seed",type=int,default=20260929); a=p.parse_args(); import pathlib
out=pathlib.Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); rows=read_jsonl(a.behavior); df=pd.DataFrame(rows)
summary={k:bootstrap_mean_ci(df[k].to_numpy(),seed=a.seed) for k in ("accuracy","B","R","M") if k in df}
df.to_csv(out/"behavior_raw.csv",index=False); pd.DataFrame([{"metric":k,**v} for k,v in summary.items()]).to_csv(out/"behavior_summary.csv",index=False); save_json(summary,out/"behavior_summary.json")
for key,title in (("B","Binding-specific obsolete residue B"),("R","Current-binding control R")):
 if key in df:
  fig,ax=plt.subplots(figsize=(5,3.3)); ax.hist(df[key],bins=30,color="#4267ac",alpha=.85); ax.axvline(0,color="black",linewidth=1); ax.set(xlabel="Logit contrast",ylabel="Examples",title=title); fig.tight_layout(); fig.savefig(out/f"{key.lower()}_distribution.png",dpi=180); plt.close(fig)
if a.pairs:
 pr=read_jsonl(a.pairs); grouped={}
 for r in pr: grouped.setdefault((r["pair_id"],r["intervention_role"]),{})[r["pair_direction"]]=r
 effects={"O_q":[],"O_d":[]}; obsolete=[]
 for (pair_id,role),members in grouped.items():
  if len(members)!=2: continue
  x,y=members[0],members[1]; px=x["candidate_probabilities"]; py=y["candidate_probabilities"]
  js=js_divergence([px[k] for k in px],[py[k] for k in px]); source_value=x.get("roles",{}).get(role); target_value=y.get("roles",{}).get(role); lx=x.get("candidate_logits",{}); ly=y.get("candidate_logits",{})
  contrast_x=(lx.get(source_value,0)-lx.get(target_value,0)) if source_value and target_value else None; contrast_y=(ly.get(source_value,0)-ly.get(target_value,0)) if source_value and target_value else None
  effects.setdefault(role,[]).append({"pair_id":pair_id,"js":js,"delta_B":y.get("B",0)-x.get("B",0),"delta_Cq":y["C_q"]-x["C_q"],"source_value":source_value,"target_value":target_value,"source_value_logit_change":ly.get(source_value,0)-lx.get(source_value,0) if source_value else None,"target_value_logit_change":ly.get(target_value,0)-lx.get(target_value,0) if target_value else None,"obsolete_value_contrast_change":contrast_y-contrast_x if contrast_x is not None else None})
  if role in ("O_q","O_d"): obsolete.append((role,pair_id,y.get("B",0)-x.get("B",0),js))
 q={p:(b,j) for r,p,b,j in obsolete if r=="O_q"}; d={p:(b,j) for r,p,b,j in obsolete if r=="O_d"}
 # Effects use same base contexts. Pair IDs include intervention role; recover base context.
 q_by={p.split(":O_q")[0]:v for p,v in q.items()}; d_by={p.split(":O_d")[0]:v for p,v in d.items()}; common=set(q_by)&set(d_by); did=np.array([q_by[k][0]-d_by[k][0] for k in sorted(common)]); js_diff=np.array([q_by[k][1]-d_by[k][1] for k in sorted(common)])
 cf={role:{"mean_js":float(np.mean([r["js"] for r in vals])) if vals else None,"n_pairs":len(vals)} for role,vals in effects.items()}
 if len(did): cf["obsolete_effect_difference_in_differences"]={**bootstrap_mean_ci(did,groups=np.array(sorted(common)),seed=a.seed),"interpretation":"input counterfactual effect; not an internal causal variable"}
 if len(js_diff): cf["obsolete_query_minus_distractor_js"]={**bootstrap_mean_ci(js_diff,groups=np.array(sorted(common)),seed=a.seed),"interpretation":"paired candidate-restricted distributional input counterfactual difference"}
 save_json(cf,out/"counterfactual_summary.json"); pd.DataFrame([{"role":role,**entry} for role,vals in effects.items() for entry in vals]).to_csv(out/"counterfactual_raw.csv",index=False)
 if len(js_diff):
  fig,ax=plt.subplots(figsize=(5,3.5)); qjs=np.array([q_by[k][1] for k in sorted(common)]); djs=np.array([d_by[k][1] for k in sorted(common)]); means=[qjs.mean(),djs.mean()]; ax.bar([0,1],means,color=["#b95b50","#4267ac"],width=.6); ax.scatter(np.zeros(len(qjs)),qjs,color="black",s=8,alpha=.35); ax.scatter(np.ones(len(djs)),djs,color="black",s=8,alpha=.35); ax.set_xticks([0,1],["Change O_q","Change O_d"]); ax.set_ylabel("Candidate restricted JS divergence"); ax.set_title("Matched input counterfactual effects"); fig.tight_layout(); fig.savefig(out/"obsolete_counterfactual_js.png",dpi=180); plt.close(fig)
print(f"wrote aggregate tables and figures to {out}")
