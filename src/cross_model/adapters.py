"""Explicit serial decoder adapters; attention contribution is post projection.

Each adapter names the corresponding decoder block, attention output, MLP output,
and concatenated query-head coordinates. Gemma 3 has additional feed-forward
normalization around its MLP; the hook is at the MLP module output, before its
post-feedforward norm. Block output is the residual entering the next block.
Query heads are concatenated at o_proj input (Phi3 uses fused qkv_proj).
"""
from contextlib import AbstractContextManager

from src.models.loader import decoder_blocks

FAMILIES = {'qwen3': ('q_proj', 'o_proj'), 'llama': ('q_proj', 'o_proj'),
            'mistral': ('q_proj', 'o_proj'), 'phi3': ('qkv_proj', 'o_proj'),
            'gemma3': ('q_proj', 'o_proj'), 'gemma3_text': ('q_proj', 'o_proj')}


def get_decoder_blocks(model):
    cfg = getattr(model.config, 'text_config', model.config)
    family = cfg.model_type
    if family not in FAMILIES: raise RuntimeError(f'unsupported adapter: {family}')
    blocks = decoder_blocks(model)
    if len(blocks) != cfg.num_hidden_layers: raise RuntimeError('layer count mismatch')
    for b in blocks:
        if not all(hasattr(b, x) for x in ('self_attn', 'mlp', 'input_layernorm', 'post_attention_layernorm')):
            raise RuntimeError('unsupported serial decoder layout')
    return blocks


def _block(model, layer):
    blocks = get_decoder_blocks(model)
    if not 0 <= layer < len(blocks): raise ValueError('layer outside model')
    return blocks[layer]


def normalized_depth(layer, count):
    if count < 2 or not 0 <= layer < count: raise ValueError('normalized depth requires >=2 layers')
    return layer / (count-1)


def get_final_residual_hook(model, layer): return _block(model, layer)
def get_attention_output_hook(model, layer): return _block(model, layer).self_attn
def get_mlp_output_hook(model, layer): return _block(model, layer).mlp


def head_dimensions(model, layer):
    cfg = getattr(model.config, 'text_config', model.config); b = _block(model, layer); a = b.self_attn
    heads, kv = cfg.num_attention_heads, cfg.num_key_value_heads
    dim = getattr(a, 'head_dim', getattr(cfg, 'head_dim', cfg.hidden_size // heads))
    projection_name, output_name = FAMILIES[cfg.model_type]
    p, out = getattr(a, projection_name), getattr(a, output_name)
    expected = (heads+2*kv)*dim if cfg.model_type == 'phi3' else heads*dim
    if heads % kv or p.out_features != expected or out.in_features != heads*dim:
        raise RuntimeError('head projection dimensions differ from adapter contract')
    return heads, kv, dim


def get_query_head_output_hook(model, layer, head):
    heads, _, _ = head_dimensions(model, layer)
    if not 0 <= head < heads: raise ValueError('query head outside model')
    return _block(model, layer).self_attn.o_proj


class ActivationHook(AbstractContextManager):
    """Capture or patch aligned prompt spans; casts donors to recipient activation dtype."""
    def __init__(self, model, layer, component, positions, source=None, head=None):
        if component == 'block_output': module = get_final_residual_hook(model, layer)
        elif component == 'attention_output': module = get_attention_output_hook(model, layer)
        elif component == 'mlp_output': module = get_mlp_output_hook(model, layer)
        elif component == 'query_head': module = get_query_head_output_hook(model, layer, head)
        else: raise ValueError('unknown component')
        self.activation = None
        self.handle = None
        positions = list(positions)
        if not positions or len(set(positions)) != len(positions): raise ValueError('invalid patch span')
        def apply(x):
            if x.ndim != 3 or min(positions) < 0 or max(positions) >= x.shape[1]:
                raise ValueError('activation shape or position mismatch')
            if component == 'query_head':
                heads, _, dim = head_dimensions(model, layer)
                if x.shape[-1] != heads*dim: raise RuntimeError('head concatenation shape mismatch')
                view = x.reshape(*x.shape[:2], heads, dim)
                selected = view[:, positions, head, :]
            else:
                view = x; selected = view[:, positions, :]
            if source is None:
                self.activation = selected.detach().clone()
                return x
            if source.shape != selected.shape: raise ValueError('donor shape mismatch')
            y = view.clone()
            donor = source.to(device=x.device, dtype=x.dtype)
            if component == 'query_head': y[:, positions, head, :] = donor
            else: y[:, positions, :] = donor
            return y.reshape_as(x)
        if component == 'query_head':
            def pre(_m, args): return (apply(args[0]), *args[1:])
            self.handle = module.register_forward_pre_hook(pre)
        else:
            def hook(_m, _i, output):
                x = output[0] if isinstance(output, tuple) else output
                y = apply(x)
                return (y, *output[1:]) if isinstance(output, tuple) else y
            self.handle = module.register_forward_hook(hook)
    def __exit__(self, *args):
        if self.handle is not None: self.handle.remove()
