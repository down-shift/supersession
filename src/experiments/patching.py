"""Residual output patching between aligned full-sequence runs."""
from src.models.hooks import ResidualHooks
from tqdm.auto import tqdm

def assert_aligned(source_ids,target_ids,expected_differences):
    if len(source_ids)!=len(target_ids): raise ValueError("paired tokenized prompts differ in length")
    actual=[i for i,(a,b) in enumerate(zip(source_ids,target_ids)) if a!=b]
    if actual!=list(expected_differences): raise ValueError(f"token differences {actual} != expected {list(expected_differences)}")

def patched_logits(model,target_inputs,source_activations,layer,position):
    import torch
    hook=ResidualHooks(model,capture_layers=[],patch={"layer":layer,"position":position,"source":source_activations})
    try:
        with torch.inference_mode(): return model(**target_inputs,use_cache=False).logits[:,-1].float()
    finally: hook.close()

def patched_logits_positions(model,target_inputs,source_activations,layer,positions):
    """One forward for a batch of position-specific patches at a selected layer."""
    import torch
    count=len(positions); expanded={k:v.expand(count,*v.shape[1:]) for k,v in target_inputs.items()}
    source=source_activations.expand(count,*source_activations.shape[1:])
    hook=ResidualHooks(model,capture_layers=[],patch={"layer":layer,"positions":positions,"source":source})
    try:
        with torch.inference_mode(): return model(**expanded,use_cache=False).logits[:,-1].float()
    finally: hook.close()

def capture_run(model,inputs):
    """Return logits and block-output residuals for one unpadded input sequence."""
    import torch
    hook=ResidualHooks(model)
    try:
        with torch.inference_mode(): out=model(**inputs,use_cache=False)
        return out.logits[0,-1].float().detach(),{i:x.detach() for i,x in hook.captures.items()}
    finally: hook.close()

def patch_sweep(model,tokenizer,source,target,metric_token_ids,chat=True,position_batch_size=16,positions=None,layers=None):
    """Patch selected block outputs (all positions by default); return raw logits."""
    import torch
    from src.data.generate import render_example
    if position_batch_size<1: raise ValueError("position_batch_size must be positive")
    device=next(model.parameters()).device
    def encode(ex): return tokenizer(render_example(ex,tokenizer,chat=chat),return_tensors="pt",add_special_tokens=False).to(device)
    sin,tin=encode(source),encode(target); si=sin["input_ids"][0].tolist(); ti=tin["input_ids"][0].tolist(); differences=[i for i,(a,b) in enumerate(zip(si,ti)) if a!=b]
    changed=[k for k in ("old_x","old_z","current_x","current_z") if source[k]!=target[k]]
    if len(changed)!=1: raise ValueError(f"patch pair must alter one semantic value, found {changed}")
    role_key=changed[0]; old_value=source[role_key]; new_value=target[role_key]
    source_text=render_example(source,tokenizer,chat=chat); target_text=render_example(target,tokenizer,chat=chat)
    try:
        def value_positions(text,value):
            char=text.index(value); enc=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
            return {i for i,(a,b) in enumerate(enc["offset_mapping"]) if a < char+len(value) and b > char}
        expected=value_positions(source_text,old_value)|value_positions(target_text,new_value)
    except (TypeError,NotImplementedError,KeyError) as e:
        raise RuntimeError("patch alignment audit requires a fast tokenizer with offset mappings") from e
    assert_aligned(si,ti,[i for i in range(min(len(si),len(ti))) if i in expected])
    if not differences or not set(differences).issubset(expected):
        raise ValueError(f"tokenized pair differs outside the intended {role_key} value span: {differences} vs {sorted(expected)}")
    if not differences: raise ValueError("patch pair has no differing input token")
    selected_positions=list(range(len(si))) if positions is None else sorted(set(int(p) for p in positions))
    if any(p<0 or p>=len(si) for p in selected_positions): raise ValueError("selected patch position is outside the input sequence")
    s_logits,s_resid=capture_run(model,sin); t_logits,_=capture_run(model,tin)
    selected_layers=sorted(s_resid) if layers is None else sorted(set(int(x) for x in layers))
    if not set(selected_layers).issubset(s_resid): raise ValueError(f"requested patch layers unavailable: {selected_layers}")
    rows=[]
    for layer in tqdm(selected_layers,desc="Patching layers"):
        acts=s_resid[layer]
        for start in tqdm(range(0,len(selected_positions),position_batch_size),desc=f"Layer {layer} position batches",leave=False):
            batch_positions=selected_positions[start:start+position_batch_size]; outs=patched_logits_positions(model,tin,acts,layer,batch_positions).detach().cpu()
            for position,out in zip(batch_positions,outs):
                donor_margin=float(s_logits[metric_token_ids[source[role_key]]]-s_logits[metric_token_ids[target[role_key]]])
                recipient_margin=float(t_logits[metric_token_ids[source[role_key]]]-t_logits[metric_token_ids[target[role_key]]])
                patched_margin=float(out[metric_token_ids[source[role_key]]]-out[metric_token_ids[target[role_key]]])
                denominator=donor_margin-recipient_margin
                rows.append({"layer":layer,"position":position,"patched_logits":{k:float(out[v]) for k,v in metric_token_ids.items()},"source_logits":{k:float(s_logits[v]) for k,v in metric_token_ids.items()},"target_logits":{k:float(t_logits[v]) for k,v in metric_token_ids.items()},"patched_source_minus_target":patched_margin,"source_source_minus_target":donor_margin,"target_source_minus_target":recipient_margin,"patch_delta_toward_donor":patched_margin-recipient_margin,"normalized_recovery":(patched_margin-recipient_margin)/denominator if abs(denominator)>1e-6 else None,"donor_value":source[role_key],"recipient_value":target[role_key],"input_difference_positions":differences,"semantic_role_key":role_key})
    return rows
