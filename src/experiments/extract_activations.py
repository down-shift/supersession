"""Stream query-position block outputs to NPZ chunks (layers × examples × hidden)."""
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from src.data.generate import render_example
from src.models.hooks import ResidualHooks

def extract(model,tokenizer,examples,output_dir,chunk_size=8,chat=True):
    import torch
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); meta=[]
    starts=range(0,len(examples),chunk_size)
    for start in tqdm(starts,desc="Activation chunks"):
        batch_examples=examples[start:start+chunk_size]; encoded=[]
        # no padding required: process individually to preserve query positions.
        all_layers=[]
        for ex in tqdm(batch_examples,desc="Extracting examples",leave=False):
            ids=tokenizer(render_example(ex,tokenizer,chat=chat),return_tensors="pt",add_special_tokens=False)["input_ids"].to(next(model.parameters()).device)
            hook=ResidualHooks(model)
            embedding={}
            embedding_handle=model.get_input_embeddings().register_forward_hook(lambda _m,_i,o: embedding.setdefault("x",o.detach()))
            try:
                with torch.inference_mode(): model(input_ids=ids,use_cache=False)
                pos=ids.shape[1]-1; all_layers.append(np.stack([embedding["x"][0,pos].float().cpu().numpy(),*[hook.captures[i][0,pos].float().cpu().numpy() for i in range(len(hook.blocks))]]))
            finally:
                hook.close(); embedding_handle.remove()
            meta.append({"example_id":ex["example_id"],"history_id":ex.get("history_id"),"query_id":ex.get("query_id"),"query_position":int(ids.shape[1]-1),"roles":ex["roles"],"split":ex.get("split")})
        np.savez_compressed(out/f"chunk_{start:06d}.npz",activations=np.asarray(all_layers,dtype=np.float16))
    return meta
