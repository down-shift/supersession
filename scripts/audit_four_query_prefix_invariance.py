#!/usr/bin/env python3
"""Audit that later query text cannot change old-x residual activations."""
import argparse, collections, json
from pathlib import Path
from src.data.io import read_jsonl
from src.data.generate import render_example
from src.experiments.patching import capture_run
from src.models.loader import load_model
from src.utils import load_config, save_json

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/four_query_288.yaml'); p.add_argument('--pairs',required=True); p.add_argument('--discovery-history-ids',required=True); p.add_argument('--output',required=True); p.add_argument('--n-histories',type=int,default=2); p.add_argument('--tolerance',type=float,default=1e-4); p.add_argument('--behavior-records',required=True)
a=p.parse_args()
history_ids=json.loads(Path(a.discovery_history_ids).read_text())[:a.n_histories]
by=collections.defaultdict(dict)
for ex in read_jsonl(a.pairs):
    if ex.get('edited_binding')=='old_x' and int(ex.get('pair_direction',-1))==0:
        by[ex['history_id']][ex['query_id']]=ex
queries=('current_x','current_z','initial_x','initial_z')
for hid in history_ids:
    if set(queries)-set(by[hid]): raise ValueError(f'missing old_x baseline query records for {hid}')
c=load_config(a.config); model,tok=load_model(c)
expected='b968826d9c46dd6066d109eabc6255188de91218'
if c.get('resolved_model_revision')!=expected or c.get('resolved_tokenizer_revision')!=expected:
    raise ValueError(f'model/tokenizer revision mismatch; expected {expected}, got {c.get("resolved_model_revision")}/{c.get("resolved_tokenizer_revision")}')
scored={}
if a.behavior_records:
    scored={r['example_id']:r for r in read_jsonl(a.behavior_records)}
device=next(model.parameters()).device; report=[]
import torch
for hid in history_ids:
    encoded_queries={}; positions={}; prompts={}
    for q in queries:
        ex=by[hid][q]; prompt=render_example(ex,tok,chat=c['model'].get('chat_template',True))
        if scored:
            previous=scored.get(ex['example_id'])
            if previous is None or previous.get('prompt')!=prompt: raise ValueError(f'frozen rendered prompt mismatch for {ex["example_id"]}')
        enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
        offsets=enc['offset_mapping']; start=prompt.index(ex['old_x'])
        pos=[i for i,(lo,hi) in enumerate(offsets) if lo<start+len(ex['old_x']) and hi>start]
        if not pos: raise ValueError(f'old-x token span not found for {hid}/{q}')
        positions[q]=pos; prompts[q]=prompt; encoded_queries[q]=enc['input_ids']
    reference=positions[queries[0]]
    if any(positions[q]!=reference for q in queries[1:]): raise ValueError(f'old-x token positions differ across queries for {hid}: {positions}')
    prefix_end=max(reference)+1
    prefix_ids=encoded_queries[queries[0]][:prefix_end]
    if any(encoded_queries[q][:prefix_end]!=prefix_ids for q in queries[1:]):
        raise ValueError(f'token IDs through old-x value differ across query variants for {hid}')
    width=max(len(encoded_queries[q]) for q in queries)
    pad_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None: raise ValueError('tokenizer needs a pad or EOS token for equal-shape prefix audit batching')
    input_ids=torch.full((len(queries),width),int(pad_id),dtype=torch.long)
    attention_mask=torch.zeros((len(queries),width),dtype=torch.long)
    for i,q in enumerate(queries):
        ids=torch.tensor(encoded_queries[q],dtype=torch.long); input_ids[i,:len(ids)]=ids; attention_mask[i,:len(ids)]=1
    _,captures=capture_run(model,{'input_ids':input_ids.to(device),'attention_mask':attention_mask.to(device)})
    for layer in sorted(captures):
        ref=captures[layer][0,reference,:]
        diffs={q:float((captures[layer][i,positions[q],:]-ref).abs().max().item()) for i,q in enumerate(queries[1:],start=1)}
        maximum=max(diffs.values())
        report.append({'history_id':hid,'layer':int(layer),'max_abs_difference':maximum,'max_by_query':diffs,'passed':maximum<=a.tolerance})
result={'histories':history_ids,'queries':list(queries),'tolerance':a.tolerance,'execution':'four prompts batched with right padding to equal sequence length','verified_identical_token_prefix_through_old_x':True,'per_layer':report,'passed':all(x['passed'] for x in report)}
save_json(result,a.output)
if not result['passed']: raise SystemExit(f'prefix-invariance audit exceeded tolerance {a.tolerance}; see {a.output}')
print(f'Prefix-invariance audit passed for {len(history_ids)} histories; wrote {a.output}')
