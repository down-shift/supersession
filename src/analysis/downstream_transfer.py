"""History-level derived-code transfer analysis."""
from collections import defaultdict
import numpy as np
from src.analysis.metrics import trimmed_mean

def summarize_histories(values,seed=20261033,n_boot=10000):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if not len(x): return {"n_histories":0,"mean":None,"median":None,"trimmed_mean_10pct":None,"fraction_positive":None,"ci95_cluster_bootstrap":None,"sign_flip_permutation_p_two_sided":None}
    rng=np.random.default_rng(seed); boot=x[rng.integers(0,len(x),(n_boot,len(x)))].mean(1)
    signs=rng.choice([-1.,1.],(n_boot,len(x))); p=(np.sum(np.abs((signs*x).mean(1))>=abs(x.mean()))+1)/(n_boot+1)
    return {"n_histories":len(x),"mean":float(x.mean()),"median":float(np.median(x)),"trimmed_mean_10pct":trimmed_mean(x,.1),"fraction_positive":float((x>0).mean()),"ci95_cluster_bootstrap":[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],"sign_flip_permutation_p_two_sided":float(p)}

def sequence_logprob(model,tokenizer,prompt,candidate,chat=False):
    """Teacher-force every token in a complete candidate continuation."""
    import torch
    prefix=tokenizer(prompt,add_special_tokens=False)["input_ids"]
    full=tokenizer(prompt+candidate,add_special_tokens=False)["input_ids"]
    if len(full)<=len(prefix) or full[:len(prefix)]!=prefix: raise ValueError("candidate is not a stable continuation at prompt boundary")
    device=next(model.parameters()).device
    ids=torch.tensor([full],device=device)
    with torch.inference_mode(): logits=model(input_ids=ids,use_cache=False).logits[0].float()
    logp=torch.log_softmax(logits,dim=-1)
    return float(sum(logp[i-1,full[i]].item() for i in range(len(prefix),len(full))))

def score_protocol_record(row,scorer,cache,stage):
    scored=scorer(row)
    if stage!="confirmatory":
        for k in ("candidate_logprobs","derived_transfer","candidate_accuracy"): scored.pop(k,None)
        scored["causal_effects_computed"]=False; return scored
    if row["pair_direction"]==0: cache[row["pair_id"]]=scored; return scored
    base=cache.get(row["pair_id"])
    if base is None: raise ValueError("edited pair member lacks baseline")
    if base["codebook"]!=row["codebook"] or base["query_id"]!=row["query_id"]: raise ValueError("pair codebook/query mismatch")
    cs=base["codebook"][row["source_value"]]; cr=base["codebook"][row["replacement_value"]]
    lb,le=base["candidate_logprobs"],scored["candidate_logprobs"]
    scored["derived_transfer"]=(le[cr]-le[cs])-(lb[cr]-lb[cs]); scored["causal_effects_computed"]=True
    return scored

def derive_cells(rows):
    pairs=defaultdict(dict)
    for r in rows: pairs[r["pair_id"]][int(r["pair_direction"])]=r
    cells=defaultdict(dict)
    for pid,m in pairs.items():
        if set(m)!={0,1}: raise ValueError(f"incomplete pair {pid}")
        b,e=m[0],m[1]
        for k in ("history_id","edited_binding","query_id","source_value","replacement_value","codebook"):
            if b[k]!=e[k]: raise ValueError("matched pair metadata differs")
        cs=b["codebook"][b["source_value"]]; cr=b["codebook"][b["replacement_value"]]
        val=(e["candidate_logprobs"][cr]-e["candidate_logprobs"][cs])-(b["candidate_logprobs"][cr]-b["candidate_logprobs"][cs])
        key=(b["edited_binding"],b["query_id"])
        if key in cells[b["history_id"]]: raise ValueError("duplicate history cell")
        cells[b["history_id"]][key]=float(val)
    return cells

def history_contrasts(rows):
    cells=derive_cells(rows); out=[]
    for hid,e in sorted(cells.items()):
        required={(b,q) for b in ("old_x","old_z","current_x","current_z") for q in ("current_x","current_z")}
        if set(e)!=required: raise ValueError("history missing binding/query cells")
        stale=.5*((e["old_x","current_x"]-e["old_x","current_z"])+(e["old_z","current_z"]-e["old_z","current_x"]))
        live=.5*((e["current_x","current_x"]-e["current_x","current_z"])+(e["current_z","current_z"]-e["current_z","current_x"]))
        out.append({"history_id":hid,"R_stale_derived":stale,"R_live_derived":live})
    return out

def evaluate_competence_gate(rows,scores,threshold=.97):
    exp={r["example_id"]:r for r in rows}; got={r["example_id"]:r for r in scores}
    if set(exp)!=set(got) or len(got)!=len(scores): raise ValueError("gate scores must exactly cover dataset")
    groups=defaultdict(list); diag=defaultdict(list)
    for eid,r in exp.items():
        s=got[eid]
        if any(s.get(k)!=v for k,v in r.items()): raise ValueError("score metadata differs")
        if s.get("causal_effects_computed") is not False or "candidate_logprobs" in s or "derived_transfer" in s: raise ValueError("competence score exposes causal output")
        acc=s.get("current_code_accuracy")
        if acc not in (0,1): raise ValueError("invalid current code accuracy")
        groups["current"].append(acc); diag[(r["query_id"],r["orientation"])].append(acc)
    return {"pass":len(groups["current"])>0 and float(np.mean(groups["current"]))>=threshold,"threshold":threshold,
      "n":len(groups["current"]),"current_derived_code_accuracy":float(np.mean(groups["current"])),
      "query_orientation_diagnostics":{"|".join(map(str,k)):{"n":len(v),"accuracy":float(np.mean(v))} for k,v in sorted(diag.items())},
      "causal_effects_computed":False}

def current_answer_stability(rows,scores):
    # Uses baseline/edit stale cells only; all observations remain included.
    byid={r["example_id"]:r for r in scores}; pairs=defaultdict(dict)
    for r in rows:
        if r.get("edited_binding") not in ("old_x","old_z"): continue
        pairs[r["pair_id"]][r["pair_direction"]]=byid[r["example_id"]]
    out=[]
    for pid,m in pairs.items():
        b,e=m[0],m[1]; code=b["answer_code"]
        for name,s in (("baseline",b),("edited",e)):
            lp=s["candidate_logprobs"]; other=max(v for k,v in lp.items() if k!=code)
            s[name+"_correct_lp"]=lp[code]; s[name+"_margin"]=lp[code]-other
        out.append({"history_id":b["history_id"],"baseline_accuracy":b["current_code_accuracy"],"edited_accuracy":e["current_code_accuracy"],
          "baseline_correct_logprob":b["baseline_correct_lp"],"edited_correct_logprob":e["edited_correct_lp"],
          "baseline_margin":b["baseline_margin"],"edited_margin":e["edited_margin"],
          "delta_correct_logprob":e["edited_correct_lp"]-b["baseline_correct_lp"],"delta_margin":e["edited_margin"]-b["baseline_margin"]})
    return out
