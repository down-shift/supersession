"""Lazy Hugging Face loading for decoder-only models."""
import logging

logger = logging.getLogger(__name__)

def _prepare_remote_model_compat(config):
    """Bridge moved typing-only Transformers symbols used by pinned remote code."""
    model = config.get("model", {})
    if not model.get("trust_remote_code") or model.get("id") != "microsoft/Phi-4-mini-instruct":
        return
    import transformers.utils as transformer_utils
    if hasattr(transformer_utils, "LossKwargs"):
        return
    loss_kwargs = getattr(transformer_utils, "TransformersKwargs", None)
    shim = "transformers.utils.LossKwargs aliased to TransformersKwargs"
    if loss_kwargs is None:
        from typing import TypedDict
        loss_kwargs = TypedDict("LossKwargs", {"num_items_in_batch": int}, total=False)
        shim = "transformers.utils.LossKwargs supplied as a typing-only compatibility alias"
    transformer_utils.LossKwargs = loss_kwargs
    config.setdefault("transformers_compatibility_shims", []).append(shim)


def _load_phi4_mini(model_id, model_config, load_kwargs):
    """Adapt Phi-4-mini's legacy tied-weight declaration to mapping-based Transformers."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    revision = model_config.get("revision")
    remote_config = AutoConfig.from_pretrained(
        model_id, revision=revision, trust_remote_code=True)
    class_reference = remote_config.auto_map["AutoModelForCausalLM"]
    model_class = get_class_from_dynamic_module(
        class_reference, model_id, revision=revision, code_revision=revision)
    tied_keys = getattr(model_class, "_tied_weights_keys", None)
    if isinstance(tied_keys, (list, tuple)):
        if tied_keys != ["lm_head.weight"]:
            raise RuntimeError(f"unsupported Phi-4-mini tied-weight declaration: {tied_keys!r}")
        model_class._tied_weights_keys = {
            "lm_head.weight": "model.embed_tokens.weight"}
    kwargs = {key: value for key, value in load_kwargs.items()
              if key != "trust_remote_code"}
    model = model_class.from_pretrained(model_id, config=remote_config, **kwargs)
    _materialize_phi4_rope_buffers(model)
    return model


def _materialize_phi4_rope_buffers(model):
    """Recreate Phi-4's nonpersistent RoPE buffers after Accelerate meta dispatch."""
    embedding = model.get_input_embeddings()
    device = embedding.weight.device
    for module in model.modules():
        if module.__class__.__name__ != "Phi3RotaryEmbedding":
            continue
        original = getattr(module, "original_inv_freq", None)
        current = getattr(module, "inv_freq", None)
        if original is None or current is None or not (original.is_meta or current.is_meta):
            continue
        inv_freq, attention_scaling = module.rope_init_fn(module.config, device)
        module.register_buffer("inv_freq", inv_freq, persistent=False)
        module.original_inv_freq = inv_freq.clone()
        module.attention_scaling = attention_scaling


def load_model(config):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("Model execution requires optional torch and transformers dependencies; install with `uv sync --extra model` on a supported Python/GPU environment.") from e
    m=config["model"]; model_id=m["id"]; tok_id=m.get("tokenizer_id") or model_id
    remote_code=m.get("trust_remote_code",False)
    logger.info("Loading tokenizer %s at revision %s", tok_id, m.get("tokenizer_revision") or m.get("revision"))
    tok=AutoTokenizer.from_pretrained(tok_id,revision=m.get("tokenizer_revision") or m.get("revision"),
                                      trust_remote_code=remote_code)
    dtype=getattr(torch,m.get("dtype","float16"))
    quantization=m.get("quantization","none")
    load_kwargs={
        "revision":m.get("revision"),
        "torch_dtype":dtype,
        "attn_implementation":m.get("attn_implementation","eager"),
        "device_map":m.get("device_map","auto"),
        "trust_remote_code":remote_code,
    }
    if quantization == "int8":
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401 - ensure backend is installed before model dispatch
        except ImportError as e:
            raise RuntimeError("8-bit loading requires bitsandbytes; install with `uv sync --extra model --extra dev`.") from e
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization != "none":
        raise ValueError(f"unsupported quantization {quantization!r}; supported values: none, int8")
    _prepare_remote_model_compat(config)
    logger.info("Loading model %s revision=%s dtype=%s quantization=%s device_map=%s",
                model_id, m.get("revision"), m.get("dtype", "float16"), quantization,
                m.get("device_map", "auto"))
    if model_id == "microsoft/Phi-4-mini-instruct" and m.get("trust_remote_code"):
        model = _load_phi4_mini(model_id, m, load_kwargs)
        config.setdefault("transformers_compatibility_shims", []).append(
            "Phi3ForCausalLM._tied_weights_keys converted to target/source mapping")
        config.setdefault("transformers_compatibility_shims", []).append(
            "Phi3RotaryEmbedding nonpersistent RoPE buffers initialized after meta dispatch")
    else:
        model=AutoModelForCausalLM.from_pretrained(model_id,**load_kwargs)
    model.eval()
    config["resolved_model_revision"]=getattr(model.config,"_commit_hash",m.get("revision"))
    config["resolved_tokenizer_revision"]=getattr(tok,"_commit_hash",None) or getattr(tok,"init_kwargs",{}).get("_commit_hash",m.get("tokenizer_revision"))
    config["resolved_quantization"]=quantization
    logger.info("Model loaded: class=%s input_device=%s resolved_revision=%s",
                model.__class__.__name__, model.get_input_embeddings().weight.device,
                config["resolved_model_revision"])
    return model,tok

def decoder_blocks(model):
    cfg=getattr(model.config,"text_config",model.config); n=getattr(cfg,"num_hidden_layers",None)
    for root in (model,getattr(model,"model",None),getattr(model,"language_model",None)):
        blocks=getattr(root,"layers",None)
        if blocks is not None and (n is None or len(blocks)==n): return list(blocks)
    raise RuntimeError(f"Unsupported decoder architecture {model.__class__.__name__}: could not locate decoder blocks")
