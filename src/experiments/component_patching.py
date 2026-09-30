"""Block-component interventions for dense Hugging Face Qwen3 decoder blocks.

Qwen3 decoder order: input residual -> input norm -> self_attn contribution ->
residual add -> post-attention norm -> MLP contribution -> residual add.
"""

COMPONENTS = ("residual_input", "attention_output", "mlp_output", "block_output")


def validate_qwen3_blocks(model):
    """Fail closed unless this is the supported dense Qwen3 decoder layout."""
    from src.models.loader import decoder_blocks
    config = getattr(model.config, "text_config", model.config)
    if getattr(config, "model_type", None) != "qwen3":
        raise RuntimeError(
            f"Unsupported component-patching architecture: expected config.model_type='qwen3', "
            f"got {getattr(config, 'model_type', None)!r}"
        )
    blocks = decoder_blocks(model)
    for i, block in enumerate(blocks):
        if not all(hasattr(block, name) for name in ("self_attn", "mlp", "input_layernorm", "post_attention_layernorm")):
            raise RuntimeError(f"Unsupported Qwen3 block {i}: expected self_attn/mlp and both layer norms")
        if not callable(getattr(block.self_attn, "register_forward_hook", None)) or not callable(getattr(block.mlp, "register_forward_hook", None)):
            raise RuntimeError(f"Unsupported Qwen3 block {i}: attention and MLP must be hookable modules")
    return blocks


def _tensor_output(output):
    if isinstance(output, tuple):
        if not output or not hasattr(output[0], "shape"):
            raise RuntimeError("hooked module returned an unsupported output; expected tensor or tuple beginning with tensor")
        return output[0]
    if not hasattr(output, "shape"):
        raise RuntimeError("hooked module returned an unsupported output; expected tensor or tuple beginning with tensor")
    return output


def _replace_output(output, value):
    if isinstance(output, tuple):
        return (value, *output[1:])
    return value


class ComponentHook:
    """Capture or donor-patch one Qwen3 block tensor at one position."""
    def __init__(self, model, layer, component, source=None, position=None):
        if component not in COMPONENTS:
            raise ValueError(f"component must be one of {COMPONENTS}, got {component!r}")
        self.blocks = validate_qwen3_blocks(model)
        if layer < 0 or layer >= len(self.blocks):
            raise ValueError(f"layer {layer} outside 0..{len(self.blocks)-1}")
        self.layer, self.component, self.source, self.position = layer, component, source, position
        self.activation = None
        block = self.blocks[layer]
        if component == "block_output":
            module, kind = block, "output"
        elif component == "attention_output":
            module, kind = block.self_attn, "output"
        elif component == "mlp_output":
            module, kind = block.mlp, "output"
        else:
            module, kind = block, "input"

        if kind == "input":
            def hook(_module, args, kwargs=None):
                kwargs = kwargs or {}
                if args:
                    x = args[0]
                    is_kw = False
                elif "hidden_states" in kwargs:
                    x, is_kw = kwargs["hidden_states"], True
                else:
                    raise RuntimeError("Qwen3 decoder block pre-hook did not receive hidden_states")
                y = self._apply(x)
                if is_kw:
                    kw = dict(kwargs); kw["hidden_states"] = y
                    return (args, kw)
                return (y, *args[1:])
            try:
                self.handle = module.register_forward_pre_hook(hook, with_kwargs=True)
            except TypeError:
                # Older Torch supports positional Qwen3 hidden_states, as used by HF Qwen3Model.
                self.handle = module.register_forward_pre_hook(lambda m, a: hook(m, a))
        else:
            def hook(_module, _inputs, output):
                x = _tensor_output(output)
                y = self._apply(x)
                return _replace_output(output, y)
            self.handle = module.register_forward_hook(hook)

    def _apply(self, x):
        if x.ndim != 3:
            raise RuntimeError(f"expected [batch, sequence, hidden] at {self.component}, got {tuple(x.shape)}")
        if self.source is not None:
            if self.position is None or self.position >= x.shape[1] or self.position < 0:
                raise ValueError(f"patch position {self.position} outside sequence length {x.shape[1]}")
            if self.source.shape[-1] != x.shape[-1] or self.source.shape[1] <= self.position:
                raise ValueError("donor activation shape is incompatible with target hook tensor")
            y = x.clone()
            y[:, self.position, :] = self.source[:, self.position, :].to(device=x.device, dtype=x.dtype)
            return y
        self.activation = x.detach()
        return x

    def close(self):
        self.handle.remove()


def capture_component(model, inputs, layer, component):
    import torch
    hook = ComponentHook(model, layer, component)
    try:
        with torch.inference_mode():
            output = model(**inputs, use_cache=False)
        if hook.activation is None:
            raise RuntimeError(f"{component} hook did not capture an activation")
        return output.logits[0, -1].float().detach(), hook.activation
    finally:
        hook.close()


def patch_component_logits(model, inputs, layer, component, donor_activation, position):
    import torch
    hook = ComponentHook(model, layer, component, source=donor_activation, position=position)
    try:
        with torch.inference_mode():
            return model(**inputs, use_cache=False).logits[0, -1].float().detach()
    finally:
        hook.close()
