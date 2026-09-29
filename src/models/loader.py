"""Lazy Hugging Face loading for decoder-only models."""
def load_model(config):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("Model execution requires optional torch and transformers dependencies; install with `uv sync --extra model` on a supported Python/GPU environment.") from e
    m=config["model"]; model_id=m["id"]; tok_id=m.get("tokenizer_id") or model_id
    tok=AutoTokenizer.from_pretrained(tok_id,revision=m.get("tokenizer_revision") or m.get("revision"))
    dtype=getattr(torch,m.get("dtype","float16"))
    quantization=m.get("quantization","none")
    load_kwargs={
        "revision":m.get("revision"),
        "torch_dtype":dtype,
        "attn_implementation":m.get("attn_implementation","eager"),
        "device_map":m.get("device_map","auto"),
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
    model=AutoModelForCausalLM.from_pretrained(model_id,**load_kwargs)
    model.eval()
    config["resolved_model_revision"]=getattr(model.config,"_commit_hash",m.get("revision"))
    config["resolved_tokenizer_revision"]=getattr(tok,"_commit_hash",None) or getattr(tok,"init_kwargs",{}).get("_commit_hash",m.get("tokenizer_revision"))
    config["resolved_quantization"]=quantization
    return model,tok

def decoder_blocks(model):
    cfg=getattr(model.config,"text_config",model.config); n=getattr(cfg,"num_hidden_layers",None)
    for root in (model,getattr(model,"model",None),getattr(model,"language_model",None)):
        blocks=getattr(root,"layers",None)
        if blocks is not None and (n is None or len(blocks)==n): return list(blocks)
    raise RuntimeError(f"Unsupported decoder architecture {model.__class__.__name__}: could not locate decoder blocks")
