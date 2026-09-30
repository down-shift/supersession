#!/usr/bin/env python3
"""Resumable task-competence scoring only for the 24-history status_2x2 gate."""
import argparse, hashlib, json, re
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
from src.data.io import read_jsonl, sha256_file
from src.data.progress import prepare_jsonl_progress, append_jsonl_record
from src.data.supersession_behavior import (audit_status_2x2_gate_dataset, render_behavior_example)
from src.data.token_validation import continuation_token_id
from src.models.loader import load_model
from src.utils import load_config, provenance, save_json


def validate_gate_tokens(rows, tokenizer, token_ids, chat):
    if not token_ids or len(set(token_ids.values())) != len(token_ids):
        raise ValueError('validated candidate token map must be nonempty and unique')
    checked=set()
    for row in rows:
        if row['answer'] not in token_ids or not set(row['candidate_values']) <= token_ids.keys():
            raise ValueError(f"{row['example_id']}: target/candidate missing from validated token map")
        prompt=render_behavior_example(row,tokenizer,chat)
        if prompt in checked: continue
        checked.add(prompt)
        for value,expected in token_ids.items():
            if continuation_token_id(tokenizer,prompt,' '+value)!=expected:
                raise ValueError(f"{value!r}: unstable one-token continuation at {row['example_id']}")
    return {'unique_prompts_checked':len(checked),'candidate_count':len(token_ids),'status':'passed'}


def score_gate(model, tokenizer, rows, token_ids, output, completed, chat):
    import torch
    expected={r['example_id']:r for r in rows}
    saved=read_jsonl(output) if Path(output).exists() else []
    for score in saved:
        eid=score.get('example_id')
        if eid not in expected or any(score.get(k)!=v for k,v in expected[eid].items()):
            raise ValueError('gate checkpoint metadata differs from dataset')
        if any(k in score for k in ('candidate_logits','candidate_probabilities','identity_transfer','matched_edit_effect','source_logit_before')):
            raise ValueError('gate checkpoint contains prohibited causal/effect outputs')
    device=next(model.parameters()).device
    candidate_values=rows[0]['candidate_values']
    if any(r['candidate_values']!=candidate_values for r in rows): raise ValueError('gate candidate vocabulary changes across rows')
    candidate_ids=np.asarray([token_ids[v] for v in candidate_values],dtype=np.int64)
    for row in tqdm(rows,desc='Status 2x2 competence gate scoring'):
        if row['example_id'] in completed: continue
        prompt=render_behavior_example(row,tokenizer,chat)
        batch=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        batch={k:v.to(device) for k,v in batch.items()}
        with torch.inference_mode(): logits=model(**batch,use_cache=False).logits[0,-1].float()
        target_id=int(token_ids[row['answer']]); target=float(logits[target_id])
        vocab_argmax=int(logits.argmax().item())
        candidate_logits=logits[torch.as_tensor(candidate_ids,device=logits.device)]
        candidate_argmax=int(candidate_ids[int(candidate_logits.argmax().item())])
        result={**row,'prompt':prompt,'query_position':int(batch['input_ids'].shape[1]-1),
                'full_vocab_next_token_accuracy':int(vocab_argmax==target_id),
                'candidate_accuracy':int(candidate_argmax==target_id),
                'target_rank':1+int((logits>target).sum().item()),
                'candidate_target_rank':1+int((candidate_logits>target).sum().item())}
        append_jsonl_record(output,result)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/four_query_288.yaml'); p.add_argument('--dataset',required=True)
    p.add_argument('--token-ids',required=True); p.add_argument('--output',required=True); p.add_argument('--resume',action='store_true')
    a=p.parse_args(); config=load_config(a.config); rows=read_jsonl(a.dataset)
    audit_status_2x2_gate_dataset(rows)
    token_doc=json.loads(Path(a.token_ids).read_text()); token_ids=token_doc['token_ids']
    if not a.resume and any(Path(str(a.output)+suffix).exists() for suffix in ('','.run.json','.provenance.json')):
        raise FileExistsError(f'{a.output} exists; select a fresh output or use --resume')
    model,tokenizer=load_model(config)
    for key in ('model_revision','tokenizer_revision'):
        requested=config['model'].get(key) or config['model'].get('revision')
        if not config.get('resolved_'+key) and requested and re.fullmatch(r'[0-9a-f]{40}',requested): config['resolved_'+key]=requested
        if not re.fullmatch(r'[0-9a-f]{40}',str(config.get('resolved_'+key))):
            raise ValueError(f'cannot record exact {key}; pin an immutable revision')
        frozen=token_doc.get(key)
        if frozen and frozen!=config.get('resolved_'+key): raise ValueError(f'validated token-map {key} differs from loaded model')
    chat=config['model'].get('chat_template',True)
    token_audit=validate_gate_tokens(rows,tokenizer,token_ids,chat)
    code_paths=[Path(__file__),Path('src/data/supersession_behavior.py'),Path('src/data/progress.py'),Path('src/data/token_validation.py'),Path('src/models/loader.py')]
    code_sha=hashlib.sha256(b''.join(x.read_bytes() for x in code_paths)).hexdigest()
    fingerprint_config={**config,'status_2x2_gate':{'purpose':'task_accuracy_and_target_ranks_only','code_sha256':code_sha,
        'config_sha256':sha256_file(a.config),'chat':chat,'python':provenance(config,a.dataset)['python'],
        'packages':provenance(config,a.dataset)['packages']}}
    completed=prepare_jsonl_progress(a.output,a.dataset,a.token_ids,fingerprint_config,rows,resume=a.resume)
    sidecar=Path(a.output+'.provenance.json')
    if a.resume and completed and not sidecar.exists(): raise ValueError('gate checkpoint provenance missing; preserve it and use a fresh path')
    if not sidecar.exists():
        save_json({**provenance(config,a.dataset),'purpose':'competence_gate_only','experiment_kind':'status_2x2',
                   'dataset_sha256':sha256_file(a.dataset),'config_sha256':sha256_file(a.config),
                   'token_map_sha256':sha256_file(a.token_ids),'code_sha256':code_sha,
                   'token_alignment_audit':token_audit,'outputs':['full_vocab_next_token_accuracy','candidate_accuracy','target_rank','candidate_target_rank'],
                   'causal_effects_computed':False},sidecar)
    score_gate(model,tokenizer,rows,token_ids,a.output,completed,chat)
    print(f'scored/resumed {len(rows)} competence-gate prompts; {len(completed)} loaded from checkpoint')


if __name__=='__main__': main()
