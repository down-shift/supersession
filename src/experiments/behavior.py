"""Behavioral logit scoring. Requires Transformers model setup."""
import numpy as np
from src.data.generate import render_example
from src.analysis.metrics import role_metrics

def score_example(model,tokenizer,ex,token_ids,device=None,chat=True):
    import torch
    prompt=render_example(ex,tokenizer,chat=chat); batch=tokenizer(prompt,return_tensors="pt",add_special_tokens=False)
    device=device or next(model.parameters()).device; batch={k:v.to(device) for k,v in batch.items()}
    with torch.inference_mode(): logits=model(**batch,use_cache=False).logits[0,-1].float().cpu().numpy()
    result=role_metrics(logits,token_ids,ex["roles"]); result.update({"example_id":ex["example_id"],"prompt":prompt,"query_position":int(batch["input_ids"].shape[1]-1)})
    result["roles"]={k:v for k,v in ex["roles"].items()}
    result["candidate_logits"]={value:float(logits[token_id]) for value,token_id in token_ids.items()}
    for key in ("pair_id","intervention_role","pair_direction","split","family"):
        if key in ex: result[key]=ex[key]
    return result

def competence_gate(direct,overwrite,thresholds):
    da=float(np.mean([x["accuracy"] for x in direct])); oa=float(np.mean([x["accuracy"] for x in overwrite]))
    return {"direct_accuracy":da,"overwrite_accuracy":oa,"pass":da>=thresholds["direct_accuracy"] and oa>=thresholds["overwrite_accuracy"]}
