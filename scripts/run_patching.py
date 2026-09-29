#!/usr/bin/env python3
"""Current-binding positive control and obsolete-value residual patch sweeps."""
import argparse,json
from tqdm.auto import tqdm
from src.utils import load_config,save_json,provenance
from src.data.io import read_jsonl,write_jsonl
from src.data.generate import counterfactual_pair
from src.models.loader import load_model
from src.experiments.patching import patch_sweep
p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/pilot.yaml"); p.add_argument("--dataset",default="outputs/pilot/dataset.jsonl"); p.add_argument("--token-ids",default="outputs/pilot/token_ids.json"); p.add_argument("--output",default="outputs/pilot/patching.jsonl"); p.add_argument("--n-pairs",type=int,default=2); a=p.parse_args()
c=load_config(a.config); model,tok=load_model(c); rows=read_jsonl(a.dataset); values=json.load(open(a.token_ids))["token_ids"]; output=[]
for ex in tqdm(rows[:a.n_pairs],desc="Patching contexts"):
 for role in ("C_q","O_q","O_d"):
  alt=next(v for v in values if v not in ex["roles"].values()); source,target=counterfactual_pair(ex,role,ex["roles"][role],alt)
  for direction,s,t in tqdm((("source_to_target",source,target),("target_to_source",target,source)),desc=f"{role} directions",leave=False):
   table=patch_sweep(model,tok,s,t,values,chat=c["model"].get("chat_template",True))
   for row in table:
    row.update({"example_id":ex["example_id"],"intervention_role":role,"source_value":s["roles"][role],"target_value":t["roles"][role],"positive_control":role=="C_q","direction":direction,"metric_orientation":"each direction is oriented toward its activation donor; patch_delta_toward_donor = patched margin - recipient margin"}); output.append(row)
write_jsonl(output,a.output); run=provenance(c,a.dataset); run["candidate_token_ids"]=values; save_json(run,a.output+".provenance.json")
import numpy as np,matplotlib.pyplot as plt
positive=[r for r in output if r["positive_control"]]
if positive:
 layers=sorted({r["layer"] for r in positive}); positions=sorted({r["position"] for r in positive}); matrix=np.array([[np.mean([r["patched_source_minus_target"] for r in positive if r["layer"]==ly and r["position"]==po]) for po in positions] for ly in layers]); fig,ax=plt.subplots(figsize=(11,5)); im=ax.imshow(matrix,aspect="auto",origin="lower",interpolation="nearest"); ax.set(xlabel="Patched token position",ylabel="Block output layer",title="Current-binding patching positive control: source − target answer logit"); fig.colorbar(im,ax=ax,label="logit difference"); fig.tight_layout(); fig.savefig(a.output+".current_positive_control.png",dpi=180); plt.close(fig)
obsolete=[r for r in output if r["intervention_role"] in ("O_q","O_d")]; summary={}
for role in ("O_q","O_d"):
 subset=[r for r in obsolete if r["intervention_role"]==role]; summary[role]={"n_records":len(subset),"mean_output_directed_patch_change":float(np.mean([r["patched_source_minus_target"]-r["target_source_minus_target"] for r in subset])) if subset else None}
save_json({"raw_metric":"donor-oriented logit margin; patch_delta_toward_donor = patched margin - recipient margin; normalized recovery omitted when endpoint separation is near zero","summary":summary,"interpretation":"Residual intervention effect; not complete circuit identification"},a.output+".obsolete_summary.json")
print(f"saved {len(output)} raw patch records")
