"""Candidate-restricted behavioral metrics."""
import numpy as np
from tqdm.auto import tqdm

def role_metrics(logits, token_ids, roles):
    candidate_values=list(token_ids)
    candidate_logits=np.asarray([logits[token_ids[v]] for v in candidate_values],dtype=float)
    cp=np.exp(candidate_logits-candidate_logits.max()); cp/=cp.sum()
    if "target" in roles:
        answer=roles["target"]; target_logit=float(logits[token_ids[answer]])
        rank=1+int(np.sum(np.asarray(logits)>target_logit))
        return {"target":target_logit,"accuracy":int(candidate_values[int(np.argmax(candidate_logits))]==answer),
                "candidate_rank":int(1+np.sum(candidate_logits>target_logit)),"full_vocab_rank":rank,
                "candidate_probabilities":dict(zip(candidate_values,cp.tolist()))}
    if "O_q" not in roles:
        q=float(logits[token_ids[roles["C_q"]]]); d=float(logits[token_ids[roles["C_d"]]])
        vals=np.array([q,d]); p=np.exp(vals-vals.max()); p/=p.sum()
        return {"C_q":q,"C_d":d,"R":q-d,"accuracy":int(candidate_values[int(np.argmax(candidate_logits))]==roles["C_q"]),"candidate_probabilities":dict(zip(candidate_values,cp.tolist())),"role_candidate_probabilities":{"C_q":float(p[0]),"C_d":float(p[1])}}
    z={r:float(logits[token_ids[roles[r]]]) for r in ("C_q","O_q","C_d","O_d")}
    vals=np.array([z[r] for r in ("C_q","O_q","C_d","O_d")]); p=np.exp(vals-vals.max()); p/=p.sum()
    return {**z,"B":z["O_q"]-z["O_d"],"R":z["C_q"]-z["C_d"],"M":z["C_q"]-max(z["O_q"],z["C_d"],z["O_d"]),"accuracy":int(candidate_values[int(np.argmax(candidate_logits))]==roles["C_q"]),"candidate_probabilities":dict(zip(candidate_values,cp.tolist())),"role_candidate_probabilities":dict(zip(("C_q","O_q","C_d","O_d"),p.tolist()))}

def js_divergence(p,q,eps=1e-12):
    p=np.asarray(p,dtype=float); q=np.asarray(q,dtype=float); p=p/p.sum(); q=q/q.sum(); m=(p+q)/2
    kl=lambda a,b: np.sum(a*np.log((a+eps)/(b+eps)))
    return float((kl(p,m)+kl(q,m))/2)

def bootstrap_mean_ci(values, groups=None, n_boot=2000, seed=0, alpha=.05):
    values=np.asarray(values,dtype=float)
    if groups is None: groups=np.arange(len(values))
    unique=np.unique(groups); rng=np.random.default_rng(seed); means=[]
    for _ in tqdm(range(n_boot),desc="Bootstrap CI",leave=False):
        sampled=rng.choice(unique,size=len(unique),replace=True); indexes=np.concatenate([np.flatnonzero(groups==g) for g in sampled]); means.append(values[indexes].mean())
    return {"mean":float(values.mean()),"ci_low":float(np.quantile(means,alpha/2)),"ci_high":float(np.quantile(means,1-alpha/2)),"n":len(values)}

def counterfactual_effect(logits_a,logits_b,token_ids,roles,source_role):
    candidates=[roles[x] for x in ("C_q","O_q","C_d","O_d")]
    pa=np.exp(logits_a[candidates]-np.max(logits_a[candidates])); pa/=pa.sum()
    pb=np.exp(logits_b[candidates]-np.max(logits_b[candidates])); pb/=pb.sum()
    return {"js":js_divergence(pa,pb),"delta_Cq":float(logits_b[token_ids[roles["C_q"]]]-logits_a[token_ids[roles["C_q"]]]),"delta_source_value":float(logits_b[token_ids[source_role]]-logits_a[token_ids[source_role]])}
