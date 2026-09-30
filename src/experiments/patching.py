"""Residual output patching between aligned full-sequence runs."""
from src.models.hooks import ResidualHooks
from tqdm.auto import tqdm

def partition_history_ids(history_ids, discovery_size=24, seed=20260929):
    """Deterministically partition unique history IDs into disjoint stages."""
    import numpy as np
    ids=sorted(set(history_ids))
    if discovery_size<1 or discovery_size>=len(ids): raise ValueError("discovery_size must leave at least one heldout history")
    np.random.default_rng(seed).shuffle(ids)
    return ids[:discovery_size],ids[discovery_size:]

def patch_effect_metrics(donor_margin, recipient_margin, patched_margin, epsilon=1e-6):
    """Compute raw donor-oriented patch change and optional normalized recovery."""
    denominator=float(donor_margin-recipient_margin)
    delta=float(patched_margin-recipient_margin)
    return {"patch_delta_toward_donor":delta,"normalized_recovery":delta/denominator if abs(denominator)>epsilon else None}

def version_selection_diagnostics(baseline_logits, patched_logits, current_value, obsolete_value, token_ids):
    """Raw correct-current vs recipient obsolete margin around an intervention."""
    old_id=token_ids[obsolete_value]; current_id=token_ids[current_value]
    before=float(baseline_logits[current_id]-baseline_logits[old_id]); after=float(patched_logits[current_id]-patched_logits[old_id])
    return {'decision_current_value':current_value,'decision_obsolete_value':obsolete_value,
      'baseline_current_minus_old':before,'patched_current_minus_old':after,'patch_delta_current_minus_old':after-before,
      'baseline_current_logit':float(baseline_logits[current_id]),'baseline_old_logit':float(baseline_logits[old_id]),
      'patched_current_logit':float(patched_logits[current_id]),'patched_old_logit':float(patched_logits[old_id])}

def identity_transfer_effect(baseline_logits, edited_logits, source_value, replacement_value, token_ids):
    """Counterfactual value identity transfer E, with fixed replacement-source orientation."""
    src=token_ids[source_value]; rep=token_ids[replacement_value]
    base=float(baseline_logits[rep]-baseline_logits[src]); edit=float(edited_logits[rep]-edited_logits[src])
    return {'identity_transfer_E':edit-base,'baseline_replacement_minus_source':base,'edited_replacement_minus_source':edit}

def relevant_minus_irrelevant(relevant_effect, irrelevant_effect):
    return float(relevant_effect)-float(irrelevant_effect)

def focal_cells(stage="discovery", all_positions=False, cell_set="focal"):
    """Predeclared edited-binding/query cells for four-query patching."""
    if cell_set == "full":
        bindings=("old_x","current_x","old_z","current_z")
        return [(b,q) for b in bindings for q in ("current_x","initial_x","current_z","initial_z")]
    if stage == "discovery":
        cells=[("old_x","current_x"),("old_x","current_z")]
        if not all_positions:
            cells += [("old_x","initial_x"),("current_x","current_x"),("current_x","initial_x")]
        return cells
    return [("old_x","current_x"),("old_x","current_z"),("old_z","current_z"),("old_z","current_x"),
            ("old_x","initial_x"),("old_z","initial_z"),("current_x","current_x"),("current_z","current_z"),
            ("current_x","initial_x"),("current_z","initial_z")]

def relevance_contrast(cell, binding, layer, site_role="final_preanswer"):
    """Per-history obsolete-value query relevance contrast."""
    relevant="current_"+binding[-1]
    other="current_"+("z" if binding.endswith("x") else "x")
    a=cell.get((binding,relevant,layer,site_role),{}); b=cell.get((binding,other,layer,site_role),{})
    ids=sorted(set(a)&set(b))
    return {h:a[h]-b[h] for h in ids}

def symmetric_relevance(cell, layer, site_role="final_preanswer"):
    x=relevance_contrast(cell,"old_x",layer,site_role); z=relevance_contrast(cell,"old_z",layer,site_role)
    ids=sorted(set(x)&set(z))
    return {h:.5*(x[h]+z[h]) for h in ids}

def canonical_site_role(binding, token_kind):
    """Map an edited binding and assignment/query token kind to stable roles."""
    if token_kind == "old_value": return "edited_binding_value" if binding.startswith("old_") else "same_variable_other_value"
    if token_kind == "current_value": return "edited_binding_value" if binding.startswith("current_") else "same_variable_other_value"
    return {"current_assignment_variable":"current_assignment_variable","query_variable":"query_variable","final_preanswer":"final_preanswer"}.get(token_kind,"other_position")

def r_x_patch(old_x_current_x, old_x_current_z):
    return float(old_x_current_x)-float(old_x_current_z)

def symmetric_r_patch(r_x, r_z):
    return .5*(float(r_x)+float(r_z))

def _read_complete_jsonl(path):
    """Read JSONL, safely dropping only an interrupted final partial line."""
    import json
    from pathlib import Path
    path=Path(path)
    if not path.exists(): return []
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        cut=raw.rfind(b'\n')+1
        raw=raw[:cut]; path.write_bytes(raw)
    result=[]
    for i,line in enumerate(raw.splitlines()):
        try: result.append(json.loads(line))
        except json.JSONDecodeError as exc: raise ValueError(f'invalid JSONL record at {path}:{i+1}') from exc
    return result

def recover_patch_checkpoint(output, completion_log):
    """Trust only explicit markers whose complete pair row count is present.

    Any output rows written before a marker are removed and their pair is
    returned as incomplete so the runner recomputes it.
    """
    import json, os
    from pathlib import Path
    output=Path(output); completion_log=Path(completion_log)
    rows=_read_complete_jsonl(output); markers=_read_complete_jsonl(completion_log)
    counts={}
    for row in rows: counts[row['pair_id']]=counts.get(row['pair_id'],0)+1
    complete={m['pair_id'] for m in markers if counts.get(m['pair_id'],0)==m.get('row_count')}
    keep=[r for r in rows if r['pair_id'] in complete]
    if len(keep)!=len(rows):
        tmp=Path(str(output)+'.repair')
        tmp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in keep),encoding='utf8')
        os.replace(tmp,output)
    good_markers=[m for m in markers if m['pair_id'] in complete]
    if len(good_markers)!=len(markers):
        tmp=Path(str(completion_log)+'.repair')
        tmp.write_text(''.join(json.dumps(m,sort_keys=True)+'\n' for m in good_markers),encoding='utf8')
        os.replace(tmp,completion_log)
    return complete

def commit_patch_pair(output, completion_log, pair_id, rows):
    """Durably append a whole pair, then its explicit completion marker."""
    import json
    from pathlib import Path
    output=Path(output); completion_log=Path(completion_log)
    with output.open('a',encoding='utf8') as f:
        for row in rows: f.write(json.dumps(row,sort_keys=True)+'\n')
        f.flush(); __import__('os').fsync(f.fileno())
    marker={'pair_id':pair_id,'row_count':len(rows)}
    with completion_log.open('a',encoding='utf8') as f:
        f.write(json.dumps(marker,sort_keys=True)+'\n')
        f.flush(); __import__('os').fsync(f.fileno())

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
                effect=patch_effect_metrics(donor_margin,recipient_margin,patched_margin)
                rows.append({"layer":layer,"position":position,"patched_logits":{k:float(out[v]) for k,v in metric_token_ids.items()},"source_logits":{k:float(s_logits[v]) for k,v in metric_token_ids.items()},"target_logits":{k:float(t_logits[v]) for k,v in metric_token_ids.items()},"patched_source_minus_target":patched_margin,"source_source_minus_target":donor_margin,"target_source_minus_target":recipient_margin,**effect,"donor_value":source[role_key],"recipient_value":target[role_key],"input_difference_positions":differences,"semantic_role_key":role_key})
    return rows
