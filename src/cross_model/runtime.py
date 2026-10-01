"""Pinned execution and cheap real-module no-op hook checks."""
import logging

from src.models.loader import load_model
from src.cross_model.progress import progress

logger = logging.getLogger(__name__)


def load_pinned_model(config):
    logger.info("Loading pinned model and tokenizer: %s", config['model']['id'])
    model,tokenizer = load_model(config)
    for key in ('model_revision','tokenizer_revision'):
        requested = config['model'].get(key) or config['model']['revision']
        actual = config.get('resolved_'+key) or requested
        if actual != requested: raise ValueError(f'loaded {key} differs from immutable pin')
        config['resolved_'+key] = actual
    return model,tokenizer


def hook_smoke(model,tokenizer,prompt):
    import torch
    from src.cross_model.adapters import ActivationHook, get_decoder_blocks
    from src.cross_model.tokens import encode
    blocks = get_decoder_blocks(model);position=len(encode(tokenizer,prompt))-1
    inputs={'input_ids':torch.tensor([encode(tokenizer,prompt)],device=model.get_input_embeddings().weight.device)}
    events=[];handles=[]
    modules=[('input_norm',blocks[0].input_layernorm),('attention',blocks[0].self_attn),
             ('post_attention_norm',blocks[0].post_attention_layernorm),('mlp',blocks[0].mlp),('block',blocks[0])]
    try:
        for label,module in modules:
            handles.append(module.register_forward_hook(lambda m,i,o,label=label:events.append(label)))
        with torch.inference_mode(): baseline=model(**inputs,use_cache=False).logits.detach().float()
    finally:
        for handle in handles:handle.remove()
    if events != [label for label,_ in modules]:raise RuntimeError('within-block ordering differs from adapter')
    dtypes={}
    for component in progress(('block_output','attention_output','mlp_output','query_head'),
                              desc='Checking intervention hooks', unit='hook'):
        kwargs={'head':0} if component=='query_head' else {}
        with ActivationHook(model,0,component,[position],**kwargs) as capture:
            with torch.inference_mode():unchanged=model(**inputs,use_cache=False).logits.detach().float()
        if capture.activation is None:raise RuntimeError('real hook did not capture')
        if not torch.allclose(baseline,unchanged,atol=1e-5,rtol=1e-5):raise RuntimeError('capture changed model output')
        with ActivationHook(model,0,component,[position],source=capture.activation,**kwargs):
            with torch.inference_mode(): patched=model(**inputs,use_cache=False).logits.detach().float()
        if not torch.isfinite(patched).all() or not torch.allclose(baseline,patched,atol=1e-5,rtol=1e-5):
            raise RuntimeError('self patch parity failed')
        dtypes[component]=str(capture.activation.dtype)
    return {'status':'passed','within_block_order':events,'activation_dtypes':dtypes,
            'model_class':model.__class__.__name__,'block_class':blocks[0].__class__.__name__,
            'attention_class':blocks[0].self_attn.__class__.__name__}
