#!/usr/bin/env python3
"""Resumable Qwen3 query-head patch scan and manually frozen reserve test."""
import argparse, collections, hashlib, json
from pathlib import Path
from tqdm.auto import tqdm
from src.data.io import read_jsonl
from src.data.generate import render_example
from src.experiments.attention_head_patching import (capture_head_layers, patch_head_batch_logits,
    patch_head_set_logits, validate_qwen3_attention_heads, validate_head_sets, donor_oriented_margin_effect,
    pair_direction_examples)
from src.experiments.patching import commit_patch_pair, recover_patch_checkpoint
from src.experiments.patching import version_selection_diagnostics
from src.models.loader import load_model
from src.utils import load_config, provenance, save_json

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/four_query_288.yaml'); p.add_argument('--pairs',required=True); p.add_argument('--token-ids',required=True)
p.add_argument('--partition-file',required=True); p.add_argument('--output',required=True)
p.add_argument('--reserve-subpartition',help='frozen JSON from freeze_head_reserve.py; required for reserve stage')
p.add_argument('--stage',choices=('discovery','reserve'),default='discovery')
p.add_argument('--layers',default='32-35',help='discovery layer list/range; default includes transition control layer 35')
p.add_argument('--heads',help='reserve only: manually frozen comma-separated LAYER:QUERY_HEAD pairs, e.g. 32:7,34:12')
p.add_argument('--head-batch-size',type=int,default=4); p.add_argument('--resume',action='store_true')
a=p.parse_args()

def parse_layers(value):
    result=set()
    for part in value.split(','):
        if '-' in part:
            lo,hi=map(int,part.split('-',1)); result.update(range(lo,hi+1))
        else: result.add(int(part))
    if not result or min(result)<0: raise ValueError('--layers must contain nonnegative layer IDs/ranges')
    return sorted(result)

def parse_head_pairs(value):
    if not value: raise ValueError('reserve confirmation requires --heads LAYER:HEAD[,LAYER:HEAD...]')
    pairs=[]
    for item in value.split(','):
        parts=item.split(':')
        if len(parts)!=2: raise ValueError(f'invalid frozen head {item!r}; expected LAYER:QUERY_HEAD')
        pairs.append((int(parts[0]),int(parts[1])))
    if len(set(pairs))!=len(pairs): raise ValueError('--heads contains duplicate (layer, head) pairs')
    return sorted(pairs)

if a.head_batch_size<1: raise ValueError('--head-batch-size must be positive')
if a.stage=='discovery':
    layers=parse_layers(a.layers); frozen_heads=None
    if not set(layers).issubset({32,33,34,35}): raise ValueError('discovery head scan is restricted to transition layers 32-35; do not scan all model layers')
    if a.heads: raise ValueError('--heads is reserved for manually frozen reserve confirmation')
else:
    frozen_heads=parse_head_pairs(a.heads); layers=sorted({layer for layer,_ in frozen_heads})

c=load_config(a.config); pairs=collections.defaultdict(dict)
for ex in read_jsonl(a.pairs): pairs[ex['pair_id']][int(ex['pair_direction'])]=ex
histories=collections.defaultdict(dict)
for members in pairs.values():
    if set(members)=={0,1}:
        row=members[0]; histories[row['history_id']][(row['edited_binding'],row['query_id'])]=members

partition_path=Path(a.partition_file)
if not partition_path.is_file():
    raise FileNotFoundError(f'frozen mechanistic partition file is missing: {partition_path}; run the existing partition-freeze step from frozen stage-1 discovery/held-out artifacts')
part=json.loads(partition_path.read_text()); part_ids=part['history_ids']
confirm=set(part_ids['confirmatory']); discovery=set(part_ids['stage1_discovery']); heldout=set(part_ids['stage1_heldout']); reserve=set(part_ids['unused_mechanistic_reserve'])
if discovery&heldout or discovery&reserve or heldout&reserve or discovery|heldout|reserve!=confirm:
    raise ValueError('mechanistic partition artifact is not a disjoint cover of confirmatory histories')
for name,ids in (('confirmatory',confirm),('stage1_discovery',discovery),('stage1_heldout',heldout),('unused_mechanistic_reserve',reserve)):
    if int(part.get('counts',{}).get(name,-1))!=len(ids): raise ValueError(f'partition artifact count mismatch for {name}')
if a.stage=='discovery': chosen=list(part_ids['stage1_discovery']); cells=(('old_x','current_x'),('old_x','current_z'),('old_x','initial_x'),('old_x','initial_z'),('current_x','current_x'),('current_x','current_z'),('old_z','current_z'),('old_z','current_x'),('old_z','initial_z'),('old_z','initial_x'),('current_z','current_z'),('current_z','current_x'))
else:
    if not a.reserve_subpartition: raise ValueError('reserve stage requires --reserve-subpartition so only head_confirmation is consumed')
    sub=json.loads(Path(a.reserve_subpartition).read_text())
    if sub.get('exact_cover_of')!='unused_mechanistic_reserve': raise ValueError('invalid reserve subpartition artifact')
    expected_parent_sha=hashlib.sha256(json.dumps(part,sort_keys=True).encode()).hexdigest()
    if sub.get('source_partition_sha256')!=expected_parent_sha: raise ValueError('reserve subpartition was frozen from a different parent partition artifact')
    subsets=sub.get('history_ids',{}); chosen=list(subsets.get('head_confirmation',[]))
    all_sub=[set(subsets.get(k,[])) for k in ('head_confirmation','path_confirmation','final_validation')]
    if any(all_sub[i]&all_sub[j] for i in range(3) for j in range(i+1,3)) or set.union(*all_sub)!=reserve:
        raise ValueError('reserve subpartition must be a disjoint exact cover of the previous unused reserve')
    for key,ids in zip(('head_confirmation','path_confirmation','final_validation'),all_sub):
        if sub.get('counts',{}).get(key)!=len(ids): raise ValueError(f'reserve subpartition count mismatch for {key}')
        expected_ids_sha=hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()
        if sub.get('sha256',{}).get(key)!=expected_ids_sha: raise ValueError(f'reserve subpartition ID digest mismatch for {key}')
    cells=(('old_x','current_x'),('old_x','current_z'),('old_z','current_z'),('old_z','current_x'))
if a.stage=='discovery' and len(chosen)!=24: raise ValueError(f'frozen discovery set must contain 24 histories, got {len(chosen)}')
if a.stage=='reserve' and set(chosen)&(discovery|heldout): raise ValueError('reserve overlaps already-used mechanistic histories')
if any(h not in histories for h in chosen): raise ValueError('partition history IDs do not match supplied pair records')
if any(cell not in histories[h] for h in chosen for cell in cells): raise ValueError('pair dataset lacks required binding/query cell')

out=Path(a.output); manifest=Path(str(out)+'.run.json'); completion=Path(str(out)+'.complete.jsonl'); out.parent.mkdir(parents=True,exist_ok=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
try: commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True,stderr=__import__('subprocess').DEVNULL).strip()
except Exception: commit=None
fingerprint={'pairs_sha256':sha(a.pairs),'token_map_sha256':sha(a.token_ids),'partition_sha256':sha(a.partition_file),'reserve_subpartition_sha256':sha(a.reserve_subpartition) if a.reserve_subpartition else None,'config_sha256':sha(a.config),'model_id':c['model']['id'],'stage':a.stage,'history_ids':chosen,'cells':cells,'layers':layers,'frozen_head_pairs':frozen_heads,'head_batch_size':a.head_batch_size,'git_commit':commit,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()+Path(__import__('src.experiments.attention_head_patching',fromlist=['__file__']).__file__).read_bytes()).hexdigest()}
if a.resume:
    if not manifest.exists(): raise ValueError('--resume requires an existing head-patching run manifest')
    saved=json.loads(manifest.read_text())
    if saved!=fingerprint: raise ValueError('cannot resume: attention-head checkpoint fingerprint differs; use a fresh output path')
    done=recover_patch_checkpoint(out,completion)
else:
    if out.exists() or manifest.exists(): raise FileExistsError(f'{out} exists; use --resume or choose a fresh output path')
    manifest.write_text(json.dumps(fingerprint,indent=2)+'\n'); out.touch(); completion.touch(); done=set()

token_ids=json.loads(Path(a.token_ids).read_text())['token_ids']; model,tok=load_model(c)
expected_revision='b968826d9c46dd6066d109eabc6255188de91218'
if c.get('resolved_model_revision')!=expected_revision or c.get('resolved_tokenizer_revision')!=expected_revision:
    raise ValueError(f'frozen model/tokenizer revision mismatch; expected {expected_revision}')
blocks,dimensions=validate_qwen3_attention_heads(model)
if max(layers)>=len(blocks): raise ValueError(f'requested layer {max(layers)} but model has {len(blocks)} layers')
if a.stage=='reserve':
    validate_head_sets({layer:[head for l,head in frozen_heads if l==layer] for layer in layers},dimensions)
behavior=Path(a.pairs).with_name('pair_behavior.jsonl')
if not behavior.exists(): raise FileNotFoundError(f'frozen rendered-prompt audit requires {behavior}')
scored={r['example_id']:r for r in read_jsonl(behavior)}
for h in chosen[:min(8,len(chosen))]:
    for cell in cells:
        ex=histories[h][cell][0]; record=scored.get(ex['example_id'])
        if record is None or record.get('prompt')!=render_example(ex,tok,chat=c['model'].get('chat_template',True)):
            raise ValueError(f'rendered prompt differs from frozen behavioral record for {ex["example_id"]}')
device=next(model.parameters()).device
def encode(ex): return tok(render_example(ex,tok,chat=c['model'].get('chat_template',True)),return_tensors='pt',add_special_tokens=False).to(device)

for hid in tqdm(chosen,desc=f'{a.stage} head-patching histories'):
  for binding,query in cells:
    base,edited=pair_direction_examples(histories[hid][(binding,query)])
    for layer in layers if a.stage=='discovery' else [None]:
        set_key='joint_'+','.join(f'L{l}H{h}' for l,h in frozen_heads) if a.stage=='reserve' else None
        pair_id=f'{base["pair_id"]}:heads:{set_key}' if set_key else f'{base["pair_id"]}:headscan:L{layer}'
        if pair_id in done: continue
        rows=[]
        for direction,donor,recipient,label in ((0,base,edited,'baseline_to_edited'),(1,edited,base,'edited_to_baseline')):
            din,rin=encode(donor),encode(recipient)
            if din['input_ids'].shape!=rin['input_ids'].shape: raise ValueError(f'paired input shapes differ for {base["pair_id"]}')
            donor_logits,donor_acts=capture_head_layers(model,din,layers)
            recipient_logits,_=capture_head_layers(model,rin,layers)
            position=rin['input_ids'].shape[1]-1
            if a.stage=='discovery':
                qheads,kvheads,head_dim=dimensions[layer]
                for start in range(0,qheads,a.head_batch_size):
                    head_ids=list(range(start,min(start+a.head_batch_size,qheads)))
                    patched=patch_head_batch_logits(model,rin,layer,donor_acts[layer],head_ids,position).cpu()
                    for row,head in enumerate(head_ids):
                        effect=donor_oriented_margin_effect(donor_logits,recipient_logits,patched[row],donor[binding],recipient[binding],token_ids)
                        row={'pair_id':pair_id,'history_id':hid,'edited_binding':binding,'query_id':query,'layer':layer,'head':head,'query_head_index':head,'num_attention_heads':qheads,'num_key_value_heads':kvheads,'head_dim':head_dim,'position':position,'site_role':'final_preanswer','direction':label,'donor_value':donor[binding],'recipient_value':recipient[binding],**effect}
                        if query.startswith('current_') and binding.startswith('old_'):
                            old_value=recipient[binding]; current_value=recipient['answer']
                            row.update(version_selection_diagnostics(recipient_logits,patched[row],current_value,old_value,token_ids))
                            row.update({'donor_old_value':donor[binding],'recipient_old_value_used':old_value})
                        rows.append(row)
            else:
                head_sets={l:[head for hl,head in frozen_heads if hl==l] for l in layers}
                patched=patch_head_set_logits(model,rin,donor_acts,head_sets,position)
                effect=donor_oriented_margin_effect(donor_logits,recipient_logits,patched,donor[binding],recipient[binding],token_ids)
                row={'pair_id':pair_id,'history_id':hid,'edited_binding':binding,'query_id':query,'layer':-1,'head':-1,'head_set':frozen_heads,'head_set_size':len(frozen_heads),'position':position,'site_role':'final_preanswer','direction':label,'donor_value':donor[binding],'recipient_value':recipient[binding],**effect}
                if query.startswith('current_') and binding.startswith('old_'):
                    old_value=recipient[binding]; current_value=recipient['answer']
                    row.update(version_selection_diagnostics(recipient_logits,patched,current_value,old_value,token_ids))
                    row.update({'donor_old_value':donor[binding],'recipient_old_value_used':old_value})
                rows.append(row)
        commit_patch_pair(out,completion,pair_id,rows); done.add(pair_id)

save_json({'provenance':provenance(c,a.pairs),'stage':a.stage,'analysis_label':'exploratory_individual_attention_head_scan' if a.stage=='discovery' else 'manually_frozen_reserve_head_set_confirmation','exploratory':a.stage=='discovery','history_ids':chosen,'layers':layers,'cells':cells,'frozen_head_pairs':frozen_heads,'head_indexing':'query-head index along the Qwen3 attention output [batch, sequence, query_heads, head_dim]; GQA repeats KV heads into query heads before attention output is formed','tensor_patched':'self_attn.o_proj pre-hook input reshaped from [batch, sequence, num_attention_heads*head_dim] to [batch, sequence, num_attention_heads, head_dim]; selected head slice replaced before o_proj','model_config':{'num_attention_heads':dimensions[0][0],'num_key_value_heads':dimensions[0][1],'head_dim':dimensions[0][2]},'attention_weights_used_as_evidence':False},str(out)+'.provenance.json')
print(f'saved/resumed {a.stage} attention-head patch records at {out}; histories={len(chosen)}, layers={layers}, head pairs={frozen_heads if frozen_heads else "all query heads"}')
