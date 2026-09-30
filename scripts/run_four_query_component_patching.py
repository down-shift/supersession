#!/usr/bin/env python3
"""Exploratory Qwen3 block-component patching; never selects a component automatically."""
import argparse, collections, hashlib, json
from pathlib import Path
from tqdm.auto import tqdm
from src.data.io import read_jsonl
from src.data.generate import render_example
from src.experiments.component_patching import COMPONENTS, capture_component, patch_component_logits, validate_qwen3_blocks
from src.experiments.patching import commit_patch_pair, recover_patch_checkpoint, patch_effect_metrics, version_selection_diagnostics
from src.models.loader import load_model
from src.utils import load_config, provenance, save_json

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/four_query_288.yaml'); p.add_argument('--pairs',required=True); p.add_argument('--token-ids',required=True)
p.add_argument('--partition-file',required=True); p.add_argument('--output',required=True)
p.add_argument('--discovery-ids',help='existing frozen discovery CSV/JSONL; defaults beside --pairs')
p.add_argument('--heldout-ids',help='existing frozen held-out CSV/JSONL; defaults beside --pairs')
p.add_argument('--stage',choices=('discovery','reserve'),default='discovery')
p.add_argument('--layers',default='25-35',help='discovery layer list/range, e.g. 25-35')
p.add_argument('--components',default=','.join(COMPONENTS)); p.add_argument('--component',choices=COMPONENTS); p.add_argument('--layer',type=int)
p.add_argument('--resume',action='store_true'); a=p.parse_args()
def parse_layers(s):
    ans=set()
    for item in s.split(','):
        if '-' in item:
            lo,hi=map(int,item.split('-',1)); ans.update(range(lo,hi+1))
        else: ans.add(int(item))
    if not ans or min(ans)<0: raise ValueError('layers must be nonnegative integers/ranges')
    return sorted(ans)
if a.stage=='discovery':
    layers=parse_layers(a.layers); components=tuple(x for x in a.components.split(',') if x)
    if a.component or a.layer is not None: raise ValueError('discovery uses --components and --layers')
    if not components or any(x not in COMPONENTS for x in components): raise ValueError(f'--components must be a comma-separated subset of {COMPONENTS}')
else:
    if a.component is None or a.layer is None: raise ValueError('reserve confirmation requires explicit --component and --layer; choices must be manually frozen')
    layers=[a.layer]; components=(a.component,)
c=load_config(a.config); pairs=collections.defaultdict(dict)
for ex in read_jsonl(a.pairs): pairs[ex['pair_id']][int(ex['pair_direction'])]=ex
histories=collections.defaultdict(dict)
for members in pairs.values():
    if set(members)=={0,1}:
        x=members[0]; histories[x['history_id']][(x['edited_binding'],x['query_id'])]=members
partition_path=Path(a.partition_file)
if not partition_path.exists():
    try:
        from scripts.freeze_mechanistic_partitions import ids_in, partition_sets
    except ModuleNotFoundError as exc:
        if exc.name != 'scripts': raise
        from freeze_mechanistic_partitions import ids_in, partition_sets
    out_root=Path(a.pairs).parent
    discovery_path=Path(a.discovery_ids) if a.discovery_ids else out_root/'mechanism/discovery/patch_Rx_by_layer_site.csv'
    heldout_path=Path(a.heldout_ids) if a.heldout_ids else out_root/'mechanism/heldout/heldout_R_by_history.csv'
    missing=[str(path) for path in (discovery_path,heldout_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'partition artifact is missing at {partition_path}, and cannot be reconstructed without existing frozen stage-1 ID sources: {missing}. Supply --discovery-ids/--heldout-ids or run the existing partition-freeze command first; no histories were repartitioned.')
    confirm_doc=read_jsonl(a.pairs)
    confirm=sorted(set(str(row['history_id']) for row in confirm_doc if row.get('partition','confirmatory')=='confirmatory'))
    discovery_ids=ids_in(discovery_path); heldout_ids=ids_in(heldout_path)
    discovery_set,heldout_set,reserve_set=partition_sets(confirm,discovery_ids,heldout_ids)
    reserve_ids=sorted(reserve_set)
    partition_doc={'source_artifacts':{'confirmatory_pairs':a.pairs,'stage1_discovery':str(discovery_path),'stage1_heldout':str(heldout_path)},'counts':{'confirmatory':len(confirm),'stage1_discovery':len(discovery_ids),'stage1_heldout':len(heldout_ids),'unused_mechanistic_reserve':len(reserve_ids)},'sha256':{'confirmatory':hashlib.sha256('\n'.join(confirm).encode()).hexdigest(),'stage1_discovery':hashlib.sha256('\n'.join(discovery_ids).encode()).hexdigest(),'stage1_heldout':hashlib.sha256('\n'.join(heldout_ids).encode()).hexdigest(),'unused_mechanistic_reserve':hashlib.sha256('\n'.join(reserve_ids).encode()).hexdigest()},'history_ids':{'confirmatory':confirm,'stage1_discovery':discovery_ids,'stage1_heldout':heldout_ids,'unused_mechanistic_reserve':reserve_ids}}
    partition_path.parent.mkdir(parents=True,exist_ok=True); partition_path.write_text(json.dumps(partition_doc,indent=2)+'\n')
part=json.loads(partition_path.read_text()); part_ids=part['history_ids']
confirm=set(part_ids['confirmatory']); discovery=set(part_ids['stage1_discovery']); heldout=set(part_ids['stage1_heldout']); reserve=set(part_ids['unused_mechanistic_reserve'])
if discovery&heldout or discovery&reserve or heldout&reserve or discovery|heldout|reserve!=confirm:
    raise ValueError('mechanistic partition artifact is not a disjoint cover of confirmatory histories')
for name,ids in (('confirmatory',confirm),('stage1_discovery',discovery),('stage1_heldout',heldout),('unused_mechanistic_reserve',reserve)):
    if int(part.get('counts',{}).get(name,-1))!=len(ids): raise ValueError(f'partition artifact count mismatch for {name}')
key='stage1_discovery' if a.stage=='discovery' else 'unused_mechanistic_reserve'
chosen=list(part_ids[key]);
expected_count=24 if a.stage=='discovery' else len(reserve)
if a.stage=='discovery' and len(chosen)!=expected_count: raise ValueError(f'frozen stage1_discovery must contain 24 IDs, got {len(chosen)}')
if a.stage=='reserve' and (len(chosen)!=expected_count or set(chosen)&(discovery|heldout)): raise ValueError('reserve IDs do not match the unused mechanistic reserve')
if any(h not in histories for h in chosen): raise ValueError('partition artifact IDs do not match supplied confirmatory pair records')
cells=(('old_x','current_x'),('old_x','current_z'))
if any(cell not in histories[h] for h in chosen for cell in cells): raise ValueError('pair dataset lacks an old_x/current query cell')
out=Path(a.output); manifest=Path(str(out)+'.run.json'); completion=Path(str(out)+'.complete.jsonl'); out.parent.mkdir(parents=True,exist_ok=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
try: commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True,stderr=__import__('subprocess').DEVNULL).strip()
except Exception: commit=None
fingerprint={'pairs_sha256':sha(a.pairs),'token_map_sha256':sha(a.token_ids),'partition_sha256':sha(a.partition_file),'config_sha256':sha(a.config),'model_id':c['model']['id'],'stage':a.stage,'history_ids':chosen,'layers':layers,'components':list(components),'cells':cells,'exploratory':a.stage=='discovery','manual_frozen_choice':{'component':a.component,'layer':a.layer} if a.stage=='reserve' else None,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()+Path(__import__('src.experiments.component_patching',fromlist=['__file__']).__file__).read_bytes()).hexdigest(),'git_commit':commit}
if a.resume:
    if not manifest.exists(): raise ValueError('--resume requires an existing run manifest')
    saved=json.loads(manifest.read_text())
    if saved!=fingerprint: raise ValueError('cannot resume: component patch checkpoint fingerprint differs; use a fresh output path')
    done=recover_patch_checkpoint(out,completion)
else:
    if out.exists() or manifest.exists(): raise FileExistsError(f'{out} exists; use --resume or choose a fresh path')
    manifest.write_text(json.dumps(fingerprint,indent=2)+'\n'); out.touch(); completion.touch(); done=set()
token_doc=json.loads(Path(a.token_ids).read_text()); token_ids=token_doc['token_ids']
model,tok=load_model(c)
expected_revision='b968826d9c46dd6066d109eabc6255188de91218'
if c.get('resolved_model_revision') != expected_revision or c.get('resolved_tokenizer_revision') != expected_revision:
    raise ValueError(f'frozen model/tokenizer revision mismatch: model={c.get("resolved_model_revision")}, tokenizer={c.get("resolved_tokenizer_revision")}; expected {expected_revision}')
blocks=validate_qwen3_blocks(model)
if max(layers)>=len(blocks): raise ValueError(f'requested layer {max(layers)} but model has {len(blocks)} layers')
behavior=Path(a.pairs).with_name('pair_behavior.jsonl')
if not behavior.exists(): raise ValueError(f'frozen rendered-prompt audit requires {behavior}')
scored={r['example_id']:r for r in read_jsonl(behavior)}
for h in chosen[:min(8,len(chosen))]:
    for cell in cells:
        ex=histories[h][cell][0]; record=scored.get(ex['example_id'])
        if record is None or record.get('prompt')!=render_example(ex,tok,chat=c['model'].get('chat_template',True)):
            raise ValueError(f'rendered prompt differs from frozen behavioral record for {ex["example_id"]}')
device=next(model.parameters()).device
def encode(ex):
    return tok(render_example(ex,tok,chat=c['model'].get('chat_template',True)),return_tensors='pt',add_special_tokens=False).to(device)
def metric(logits,donor_value,recipient_value):
    return float(logits[token_ids[donor_value]]-logits[token_ids[recipient_value]])
for hid in tqdm(chosen,desc=f'{a.stage} component histories'):
  for binding,query in cells:
    base,edited=histories[hid][(binding,query)][0],histories[hid][(binding,query)][1]
    for layer in layers:
      for component in components:
        pair_id=f'{base["pair_id"]}:{component}:L{layer}'
        if pair_id in done: continue
        rows=[]
        for direction,donor,recipient,label in ((0,base,edited,'baseline_to_edited'),(1,edited,base,'edited_to_baseline')):
            din,rin=encode(donor),encode(recipient)
            if din['input_ids'].shape!=rin['input_ids'].shape: raise ValueError(f'paired input shapes differ for {base["pair_id"]}; donor patch alignment is undefined')
            donor_logits,donor_act=capture_component(model,din,layer,component)
            recipient_logits,_=capture_component(model,rin,layer,component)
            pos=rin['input_ids'].shape[1]-1
            patched=patch_component_logits(model,rin,layer,component,donor_act,pos)
            old_key='old_x'; donor_value=donor[old_key]; recipient_value=recipient[old_key]
            dm=metric(donor_logits,donor_value,recipient_value); rm=metric(recipient_logits,donor_value,recipient_value); pm=metric(patched,donor_value,recipient_value)
            row={'pair_id':pair_id,'history_id':hid,'query_id':query,'edited_binding':binding,'layer':layer,'component':component,'site_role':'final_preanswer','position':pos,'direction':label,'donor_value':donor_value,'recipient_value':recipient_value,'donor_margin':dm,'recipient_margin':rm,'patched_margin':pm,**patch_effect_metrics(dm,rm,pm)}
            if query.startswith('current_') and binding.startswith('old_'):
                old_key=binding; current_value=recipient['answer']; old_value=recipient[old_key]
                row.update(version_selection_diagnostics(recipient_logits,patched,current_value,old_value,token_ids))
                row.update({'donor_old_value':donor[old_key],'recipient_old_value_used':old_value,
                    'baseline_current_logit':float(recipient_logits[token_ids[current_value]]),'patched_current_logit':float(patched[token_ids[current_value]]),
                    'baseline_old_logit':float(recipient_logits[token_ids[old_value]]),'patched_old_logit':float(patched[token_ids[old_value]])})
            rows.append(row)
        commit_patch_pair(out,completion,pair_id,rows); done.add(pair_id)
save_json({'provenance':provenance(c,a.pairs),'analysis_label':'exploratory_component_patching' if a.stage=='discovery' else 'manually_frozen_reserve_confirmation','exploratory':a.stage=='discovery','stage':a.stage,'history_ids':chosen,'layers':layers,'components':list(components),'cells':cells,'manual_frozen_choice':fingerprint['manual_frozen_choice'],'hook_semantics':{'residual_input':'decoder block hidden_states input before input_layernorm','attention_output':'self_attn first output tensor after output projection, before residual addition','mlp_output':'mlp output tensor after down projection, before residual addition','block_output':'decoder block first output tensor after both residual additions'}},str(out)+'.provenance.json')
print(f'saved/resumed {a.stage} component patch records at {out}; histories={len(chosen)}, layers={layers}, components={components}')
