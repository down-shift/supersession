"""Narrow Qwen3 Q/K/V capture and activation replacement utilities.

Q-head h maps to KV-head ``h // (num_query_heads // num_kv_heads)`` under
the contiguous repeat-interleave GQA layout. Query heads and KV heads are
distinct; K/V interventions therefore index KV heads. These hooks capture and
replace raw q_proj/k_proj/v_proj outputs before rotary embedding. Unsupported layouts fail closed.
"""
from __future__ import annotations
import torch
from src.experiments.attention_head_patching import validate_qwen3_attention_heads

def query_to_kv_head(query_head, query_heads, kv_heads):
    if query_heads<1 or kv_heads<1 or query_heads%kv_heads: raise ValueError('invalid GQA head counts')
    if not 0<=query_head<query_heads: raise ValueError('query head outside model range')
    return query_head//(query_heads//kv_heads)

class ProjectionHook:
    def __init__(self, projection, heads, head_dim, position, source=None, head_ids=None):
        self.heads=int(heads); self.head_dim=int(head_dim); self.position=int(position); self.source=source
        self.head_ids=tuple(range(heads)) if head_ids is None else tuple(map(int,head_ids)); self.activation=None
        if any(h<0 or h>=heads for h in self.head_ids): raise ValueError('projection head index out of range')
        def hook(_module,_inputs,output):
            if not hasattr(output,'shape') or output.ndim!=3 or output.shape[-1]!=self.heads*self.head_dim:
                raise RuntimeError('unsupported Qwen3 projection output layout')
            view=output.reshape(*output.shape[:2],self.heads,self.head_dim)
            if not 0<=self.position<view.shape[1]: raise ValueError('projection position outside sequence')
            if self.source is None:
                self.activation=view.detach().clone(); return output
            if self.source.ndim!=4 or self.source.shape[2:]!=view.shape[2:] or self.source.shape[1] not in (1,view.shape[1]): raise ValueError('source projection activation shape mismatch')
            patched=view.clone()
            if self.source.shape[0] not in (1,view.shape[0]): raise ValueError('source batch mismatch')
            for b in range(view.shape[0]):
                source_row=0 if self.source.shape[0]==1 else b
                source_pos=0 if self.source.shape[1]==1 else self.position
                for h in self.head_ids: patched[b,self.position,h]=self.source[source_row,source_pos,h].to(view)
            return patched.reshape_as(output)
        self.handle=projection.register_forward_hook(hook)
    def close(self): self.handle.remove()

def validate_qkv_model(model, layer):
    blocks,dims=validate_qwen3_attention_heads(model)
    if not 0<=layer<len(blocks): raise ValueError('layer outside model')
    attn=blocks[layer].self_attn; qh,kh,hd=dims[layer]
    for name,heads in (('q_proj',qh),('k_proj',kh),('v_proj',kh)):
        proj=getattr(attn,name,None)
        if proj is None or getattr(proj,'out_features',None)!=heads*hd or not callable(getattr(proj,'register_forward_hook',None)):
            raise RuntimeError(f'unsupported Qwen3 {name} layout; expected {heads} heads x {hd}')
    return attn,(qh,kh,hd)

def capture_qkv(model, inputs, layer, position):
    attn,(qh,kh,hd)=validate_qkv_model(model,layer); hooks={n:ProjectionHook(getattr(attn,n),heads,hd,position) for n,heads in (('q_proj',qh),('k_proj',kh),('v_proj',kh))}
    try:
        with torch.inference_mode(): out=model(**inputs,use_cache=False)
        return out.logits[0,-1].float().detach(),{n:h.activation[:,position:position+1].clone() for n,h in hooks.items()}
    finally:
        for h in hooks.values(): h.close()

def intervene_qkv_logits(model, inputs, layer, tensor, position, source, head_ids):
    if tensor not in ('q','k','v'): raise ValueError('tensor must be q, k, or v')
    attn,(qh,kh,hd)=validate_qkv_model(model,layer); heads=qh if tensor=='q' else kh; name={'q':'q_proj','k':'k_proj','v':'v_proj'}[tensor]
    hook=ProjectionHook(getattr(attn,name),heads,hd,position,source=source,head_ids=head_ids)
    try:
        with torch.inference_mode(): return model(**inputs,use_cache=False).logits[0,-1].float().detach()
    finally: hook.close()
