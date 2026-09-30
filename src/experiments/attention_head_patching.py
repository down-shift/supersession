"""Per-query-head causal patches at the Qwen3 attention output-projection input.

For the pinned Transformers Qwen3 implementation, attention returns
``[batch, sequence, query_heads, head_dim]`` and Qwen3Attention.forward then
flattens that tensor and calls ``o_proj``. The ``o_proj`` pre-hook therefore
sees ``[batch, sequence, query_heads * head_dim]``. We reshape that input and
replace selected query-head slices before projection. This is a faithful
pre-o_proj intervention; no o_proj-output dimensions are treated as heads.
The tensor order was checked against Transformers 5.17.0's
``models/qwen3/modeling_qwen3.py`` (Qwen3Attention.forward and
eager_attention_forward).
"""

def validate_qwen3_attention_heads(model):
    from src.experiments.component_patching import validate_qwen3_blocks
    blocks=validate_qwen3_blocks(model)
    config=getattr(model.config,"text_config",model.config)
    qheads=int(getattr(config,"num_attention_heads",0) or 0)
    kvheads=int(getattr(config,"num_key_value_heads",0) or 0)
    if qheads<1 or kvheads<1 or qheads%kvheads:
        raise RuntimeError(f"Unsupported Qwen3 GQA configuration: query heads={qheads}, KV heads={kvheads}")
    dimensions=[]
    for i,block in enumerate(blocks):
        attn=block.self_attn
        head_dim=int(getattr(attn,"head_dim",0) or 0)
        qproj=getattr(attn,"q_proj",None); kproj=getattr(attn,"k_proj",None); oproj=getattr(attn,"o_proj",None)
        expected_q=qheads*head_dim; expected_kv=kvheads*head_dim
        if (head_dim<1 or qproj is None or kproj is None or oproj is None or
            getattr(qproj,"out_features",None)!=expected_q or
            getattr(kproj,"out_features",None)!=expected_kv or
            getattr(oproj,"in_features",None)!=expected_q or
            not callable(getattr(oproj,"register_forward_pre_hook",None))):
            raise RuntimeError(
                f"Unsupported Qwen3 attention block {i}: expected q_proj={qheads}x{head_dim}, "
                f"k_proj={kvheads}x{head_dim}, and o_proj input={expected_q}; "
                "cannot safely identify per-query-head slices before o_proj"
            )
        groups=getattr(attn,"num_key_value_groups",qheads//kvheads)
        if groups!=qheads//kvheads:
            raise RuntimeError(f"Unsupported Qwen3 GQA mapping at block {i}: num_key_value_groups={groups}")
        dimensions.append((qheads,kvheads,head_dim))
    return blocks,dimensions


def validate_head_sets(head_sets,dimensions):
    """Validate mapping {layer: iterable(query_head_index)} against model dims."""
    checked={}
    for layer,heads in head_sets.items():
        layer=int(layer)
        if layer<0 or layer>=len(dimensions): raise ValueError(f"head-patch layer {layer} is outside the model")
        qheads=dimensions[layer][0]; heads=tuple(sorted(set(map(int,heads))))
        if not heads: raise ValueError(f"head set for layer {layer} is empty")
        invalid=[h for h in heads if h<0 or h>=qheads]
        if invalid: raise ValueError(f"query-head indices {invalid} outside 0..{qheads-1} at layer {layer}")
        checked[layer]=heads
    if not checked: raise ValueError("at least one (layer, query-head) pair is required")
    return checked


def pair_direction_examples(members):
    """Return frozen baseline/edited examples by explicit pair-direction key."""
    if not isinstance(members,dict) or set(members)!={0,1}:
        raise ValueError(f"paired examples must be keyed by pair_direction 0 and 1, got {sorted(members) if isinstance(members,dict) else type(members).__name__}")
    return members[0],members[1]


class HeadInputHook:
    """Capture or replace one/more query-head results before Qwen3 ``o_proj``."""
    def __init__(self,attention,query_heads,head_dim,source=None,position=None,head_sets_per_row=None):
        self.query_heads=int(query_heads); self.head_dim=int(head_dim)
        self.source=source; self.position=position; self.head_sets_per_row=head_sets_per_row
        self.activation=None
        def hook(_module,args):
            if not args or not hasattr(args[0],"shape"):
                raise RuntimeError("Qwen3 o_proj pre-hook did not receive its concatenated head-result tensor")
            flat=args[0]
            if flat.ndim!=3 or flat.shape[-1]!=self.query_heads*self.head_dim:
                raise RuntimeError(
                    f"Unsupported o_proj input shape {tuple(flat.shape)}; expected [batch, sequence, "
                    f"{self.query_heads*self.head_dim}] from {self.query_heads} query heads x {self.head_dim}"
                )
            view=flat.reshape(flat.shape[0],flat.shape[1],self.query_heads,self.head_dim)
            if self.source is None:
                self.activation=view.detach().clone()
                return None
            if self.position is None or not 0<=self.position<view.shape[1]:
                raise ValueError(f"head patch position {self.position} outside sequence length {view.shape[1]}")
            if self.source.ndim!=4 or self.source.shape[1:]!=view.shape[1:]:
                raise ValueError(f"donor head activation shape {tuple(self.source.shape)} incompatible with {tuple(view.shape)}")
            if self.head_sets_per_row is None:
                row_heads=[tuple(range(self.query_heads)) for _ in range(view.shape[0])]
            else:
                row_heads=self.head_sets_per_row
                if len(row_heads)!=view.shape[0]:
                    raise ValueError("one selected head set is required for each expanded batch row")
            if self.source.shape[0] not in (1,view.shape[0]):
                raise ValueError("donor head activation batch must be 1 or match the target batch")
            patched=view.clone()
            for row,heads in enumerate(row_heads):
                donor_row=row if self.source.shape[0]>1 else 0
                for head in heads:
                    if not 0<=int(head)<self.query_heads:
                        raise ValueError(f"query head {head} outside 0..{self.query_heads-1}")
                    patched[row,self.position,int(head),:]=self.source[donor_row,self.position,int(head),:].to(device=view.device,dtype=view.dtype)
            return (patched.reshape_as(flat),*args[1:])
        self.handle=attention.o_proj.register_forward_pre_hook(hook)

    def close(self): self.handle.remove()


def capture_head_layers(model,inputs,layers):
    import torch
    blocks,dimensions=validate_qwen3_attention_heads(model)
    hooks={}
    try:
        for layer in sorted(set(map(int,layers))):
            if layer<0 or layer>=len(blocks): raise ValueError(f"layer {layer} outside model")
            qheads,_,head_dim=dimensions[layer]
            hooks[layer]=HeadInputHook(blocks[layer].self_attn,qheads,head_dim)
        with torch.inference_mode(): output=model(**inputs,use_cache=False)
        if any(h.activation is None for h in hooks.values()): raise RuntimeError("one or more Qwen3 head hooks did not capture")
        return output.logits[0,-1].float().detach(),{layer:h.activation for layer,h in hooks.items()}
    finally:
        for hook in hooks.values(): hook.close()


def patch_head_batch_logits(model,target_inputs,layer,source_activation,head_ids,position):
    """Batch independent single-head interventions into one model forward."""
    import torch
    blocks,dimensions=validate_qwen3_attention_heads(model); qheads,_,head_dim=dimensions[layer]
    if any(not 0<=int(h)<qheads for h in head_ids): raise ValueError(f"head IDs must be in 0..{qheads-1}")
    n=len(head_ids)
    if n<1: raise ValueError("head_ids cannot be empty")
    expanded={k:v.expand(n,*v.shape[1:]) for k,v in target_inputs.items()}
    hook=HeadInputHook(blocks[layer].self_attn,qheads,head_dim,source=source_activation,position=position,head_sets_per_row=[(int(h),) for h in head_ids])
    try:
        with torch.inference_mode(): return model(**expanded,use_cache=False).logits[:,-1].float().detach()
    finally: hook.close()


def patch_head_set_logits(model,target_inputs,source_activations,head_sets,position):
    """Jointly patch manually selected heads, possibly across multiple layers."""
    import torch
    blocks,dimensions=validate_qwen3_attention_heads(model)
    selected=validate_head_sets(head_sets,dimensions); hooks={}
    try:
        for layer,heads in selected.items():
            qheads,_,head_dim=dimensions[layer]
            hooks[layer]=HeadInputHook(blocks[layer].self_attn,qheads,head_dim,source=source_activations[layer],position=position,head_sets_per_row=[heads])
        with torch.inference_mode(): return model(**target_inputs,use_cache=False).logits[0,-1].float().detach()
    finally:
        for hook in hooks.values(): hook.close()


def donor_oriented_margin_effect(donor_logits,recipient_logits,patched_logits,donor_value,recipient_value,token_ids):
    """Patch change in donor-value minus recipient-value margin."""
    from src.experiments.patching import patch_effect_metrics
    def margin(logits): return float(logits[token_ids[donor_value]]-logits[token_ids[recipient_value]])
    return patch_effect_metrics(margin(donor_logits),margin(recipient_logits),margin(patched_logits))
