"""Query residual extraction and block-boundary residual patching."""
class ResidualHooks:
    """By convention, block output means the residual entering the next block."""
    def __init__(self, model, capture_layers=None, patch=None):
        self.model=model; self.blocks=__import__("src.models.loader",fromlist=["decoder_blocks"]).decoder_blocks(model)
        self.captures={}; self.handles=[]; wanted=set(capture_layers if capture_layers is not None else range(len(self.blocks)))
        for i,block in enumerate(self.blocks):
            if i not in wanted and not (patch and patch["layer"]==i): continue
            def hook(_module,_inputs,output,idx=i):
                x=output[0] if isinstance(output,tuple) else output
                if patch and idx==patch["layer"]:
                    source=patch["source"]
                    if "positions" in patch:
                        positions=patch["positions"]; rows=__import__("torch").arange(len(positions),device=x.device)
                        x=x.clone()
                        x[rows,positions,:]=source[rows,positions,:].to(device=x.device,dtype=x.dtype)
                    else:
                        source=patch["source"]; position=patch["position"]
                        x=x.clone(); x[:,position,:]=source[:,position,:].to(device=x.device,dtype=x.dtype)
                self.captures[idx]=x.detach()
                if isinstance(output,tuple): return (x,*output[1:])
                return x
            self.handles.append(block.register_forward_hook(hook))
    def close(self):
        for h in self.handles: h.remove()
