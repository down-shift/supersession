#!/usr/bin/env python3
"""Resumable donor-oriented residual patching for the frozen four-query task."""
import argparse, collections, hashlib, json
from pathlib import Path
from tqdm.auto import tqdm
from src.data.io import read_jsonl
from src.data.generate import render_example, query_variable_span
from src.experiments.patching import (patch_sweep, partition_history_ids, focal_cells,
    canonical_site_role, recover_patch_checkpoint, commit_patch_pair)
from src.experiments.prefix_invariance import audit_prefix_invariance
from src.models.loader import load_model
from src.utils import load_config, provenance, save_json

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/four_query_288.yaml'); p.add_argument('--pairs',required=True); p.add_argument('--token-ids',required=True); p.add_argument('--output',required=True)
p.add_argument('--stage',choices=('discovery','heldout'),default='discovery'); p.add_argument('--n-histories',type=int); p.add_argument('--discovery-histories',type=int,default=24); p.add_argument('--seed',type=int,default=20260929)
p.add_argument('--layers'); p.add_argument('--all-positions',action='store_true'); p.add_argument('--cell-set',choices=('focal','full'),default='focal'); p.add_argument('--resume',action='store_true'); p.add_argument('--audit-prefix-invariance',type=int,default=0)
p.add_argument('--position-batch-size',type=int,default=16)
a=p.parse_args(); layers=[int(x) for x in a.layers.split(',')] if a.layers else None
if a.stage=='heldout' and layers is None: raise ValueError('heldout patching requires frozen --layers')
if a.all_positions and a.stage!='discovery': raise ValueError('--all-positions is discovery only')
c=load_config(a.config); token_doc=json.loads(Path(a.token_ids).read_text()); token_ids=token_doc['token_ids']
allpairs=collections.defaultdict(dict)
for ex in read_jsonl(a.pairs): allpairs[ex['pair_id']][int(ex['pair_direction'])]=ex
histories=collections.defaultdict(dict)
for pair,members in allpairs.items():
    if set(members)=={0,1}: histories[members[0]['history_id']][(members[0]['edited_binding'],members[0]['query_id'])]=members
ids,held=partition_history_ids(histories,a.discovery_histories,a.seed); chosen=ids if a.stage=='discovery' else held
limit=a.n_histories if a.n_histories is not None else min(24 if a.stage=='discovery' else 96,len(chosen)); chosen=chosen[:limit]
cells=focal_cells(a.stage,a.all_positions,a.cell_set)
if any(k not in histories[h] for h in chosen for k in cells): raise ValueError('pair dataset lacks one or more requested cells')
out=Path(a.output); manifest=Path(str(out)+'.run.json'); completion_log=Path(str(out)+'.complete.jsonl'); out.parent.mkdir(parents=True,exist_ok=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
try: commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True,stderr=__import__('subprocess').DEVNULL).strip()
except Exception: commit=None
fingerprint={'pairs_sha256':sha(a.pairs),'token_map_sha256':sha(a.token_ids),'config_sha256':sha(a.config),'model_id':c['model']['id'],'model_revision':c['model'].get('revision'),'prompt_variant':sorted({ex.get('prompt_variant') for h in histories.values() for pair in h.values() for ex in pair.values()}),'stage':a.stage,'cells':cells,'seed':a.seed,'discovery_history_count':a.discovery_histories,'history_ids':chosen,'n_histories':limit,'layers':layers,'all_positions':a.all_positions,'cell_set':a.cell_set,'position_batch_size':a.position_batch_size,'patch_code_version':commit}
fingerprint['patch_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()+Path(__import__('src.experiments.patching',fromlist=['__file__']).__file__).read_bytes()+Path(__import__('src.models.hooks',fromlist=['__file__']).__file__).read_bytes()).hexdigest()
if a.resume:
    if not manifest.exists(): raise ValueError('--resume requires patch run manifest')
    saved=json.loads(manifest.read_text())
    if saved!=fingerprint:
        changed=[]
        for key in sorted(set(saved)|set(fingerprint)):
            if saved.get(key)!=fingerprint.get(key):
                old,new=saved.get(key),fingerprint.get(key)
                if key=='history_ids':
                    import hashlib
                    old=json.dumps(old,sort_keys=True); new=json.dumps(new,sort_keys=True)
                    old=f'n={len(saved.get(key,[]))}, sha256={hashlib.sha256(old.encode()).hexdigest()[:12]}'
                    new=f'n={len(fingerprint.get(key,[]))}, sha256={hashlib.sha256(new.encode()).hexdigest()[:12]}'
                changed.append(f'{key}: saved={old!r}, current={new!r}')
        raise ValueError('cannot resume: patch checkpoint fingerprint differs; checkpoint was left untouched. '
                         'Use a fresh output path after reviewing changed fields:\n  '+'\n  '.join(changed))
    done=recover_patch_checkpoint(out,completion_log)
else:
    if out.exists() or manifest.exists(): raise FileExistsError(f'{out} exists; use --resume or a new path')
    manifest.write_text(json.dumps(fingerprint,indent=2)+'\n'); out.touch(); completion_log.touch(); done=set()
save_json(ids, out.parent/'discovery_history_ids.json'); save_json(held,out.parent/'heldout_history_ids.json')
model,tok=load_model(c)
expected_revision='b968826d9c46dd6066d109eabc6255188de91218'
if c.get('resolved_model_revision') != expected_revision or c.get('resolved_tokenizer_revision') != expected_revision:
    raise ValueError(f'frozen model/tokenizer revision mismatch: model={c.get("resolved_model_revision")}, tokenizer={c.get("resolved_tokenizer_revision")}; expected {expected_revision}')
behavior_log=Path(a.pairs).with_name('pair_behavior.jsonl')
if not behavior_log.exists(): raise ValueError(f'rendered-prompt audit requires frozen behavioral records: {behavior_log}')
scored={r['example_id']:r for r in read_jsonl(behavior_log)}
candidates=[members[0] for hid in chosen for members in histories[hid].values()]
for ex in candidates[:min(8,len(candidates))]:
    previous=scored.get(ex['example_id'])
    if previous is None: raise ValueError(f'frozen behavioral records are missing {ex["example_id"]}')
    if previous.get('prompt') != render_example(ex,tok,chat=c['model'].get('chat_template',True)):
        raise ValueError(f'rendered prompt differs from frozen behavioral run for {ex["example_id"]}; stop before patching')
if a.audit_prefix_invariance:
    audit=[]
    audit_ids=chosen[:a.audit_prefix_invariance]
    for hid in audit_ids:
        examples={}
        for query in ('current_x','current_z','initial_x','initial_z'):
            pair=allpairs.get(f'{hid}:old_x:{query}')
            if not pair: raise ValueError(f'prefix audit cannot find old_x/{query} for {hid}')
            examples[query]=pair[0]
        report=audit_prefix_invariance(model,tok,examples,('current_x','current_z','initial_x','initial_z'),chat=c['model'].get('chat_template',True),behavior_records=scored,tolerance=1e-4)
        audit.extend(dict(history_id=hid,**r) for r in report['per_layer'])
    save_json({'tolerance':1e-4,'histories':audit_ids,'execution':'shared one-batch right-padded helper','per_layer':audit,'passed':all(x['passed'] for x in audit)},str(out)+'.prefix_invariance.json')
    if any(not x['passed'] for x in audit): raise ValueError('causal-prefix invariance audit exceeded 1e-4')
try:
 for hid in tqdm(chosen,desc=f'{a.stage} patch histories'):
  for binding,query in cells:
   pair=histories[hid][(binding,query)]; base,edited=pair[0],pair[1]
   pair_id=base['pair_id']
   if pair_id in done: continue
   qrows=[]; direction_counts={}
   for direction,donor,recipient,label in ((0,base,edited,'baseline_to_edited'),(1,edited,base,'edited_to_baseline')):
    text=render_example(donor,tok,chat=c['model'].get('chat_template',True)); enc=tok(text,add_special_tokens=False,return_offsets_mapping=True); offsets=enc['offset_mapping']
    def positions(start,end): return [i for i,(lo,hi) in enumerate(offsets) if lo<end and hi>start]
    x= binding[-1]; old_key=f'old_{x}'; cur_key=f'current_{x}'
    old=donor[old_key]; os_=text.index(old); current=donor[cur_key]; cs=text.index(current)
    var=donor['variables'][0 if x=='x' else 1]
    oldrole=canonical_site_role(binding,'old_value')
    currole=canonical_site_role(binding,'current_value')
    sites={oldrole:positions(os_,os_+len(old)),currole:positions(cs,cs+len(current))}
    line_prefix='Update: '+var+' = '
    var_start=text.find(line_prefix)
    if var_start<0: raise ValueError(f'could not locate current assignment variable for {binding} under initial_update rendering')
    sites['current_assignment_variable']=positions(var_start+len('Update: '),var_start+len('Update: ')+len(var))
    if not sites['current_assignment_variable']: raise ValueError(f'current assignment variable tokenization is empty for {binding}')
    qstart,qend=query_variable_span(donor,text); sites['query_variable']=positions(qstart,qend); sites['final_preanswer']=[len(enc['input_ids'])-1]
    if a.all_positions:
     vars_=donor['variables']
     for key,assignment,var_idx,prefix in (
      ('old_x_value','Initial assignment:',0,'old_x'),('old_z_value','Initial assignment:',1,'old_z'),
      ('current_x_value','Update:',0,'current_x'),('current_z_value','Update:',1,'current_z')):
      value=donor[prefix]
      marker=assignment+' '+vars_[var_idx]+' = '
      value_start=text.find(marker)
      if value_start>=0:
       variable_start=value_start+len(assignment)+1
       value_start+=len(marker); sites[key]=positions(value_start,value_start+len(value))
       sites[key.replace('_value','_variable')]=positions(variable_start,variable_start+len(vars_[var_idx]))
    selected=sorted(set(range(len(enc['input_ids'])) if a.all_positions else (p for ps in sites.values() for p in ps)))
    table=patch_sweep(model,tok,donor,recipient,token_ids,chat=c['model'].get('chat_template',True),positions=selected,layers=layers,position_batch_size=a.position_batch_size)
    direction_counts[label]=len(table)
    role_at={pos:role for role,ps in sites.items() for pos in ps}
    for row in table:
     pos=row['position']; role=role_at.get(pos,'other_position'); ids_at=enc['input_ids']; tid=int(ids_at[pos]); token_text=tok.decode([tid],skip_special_tokens=False)
     correct=recipient['answer']; dk=token_ids[correct]; source_old=recipient[old_key]
     src_log=row['source_logits']; tgt_log=row['target_logits']; patched=row['patched_logits']
     # candidate logits are keyed by candidate strings in frozen token map
     correct_delta=patched[correct]-tgt_log[correct]
     rec_margin=tgt_log[correct]-tgt_log.get(source_old,0.0); pat_margin=patched[correct]-patched.get(source_old,0.0)
     row.update({'pair_id':pair_id,'history_id':hid,'query_id':query,'edited_binding':binding,'direction':label,'donor_pair_direction':label,'recipient_pair_direction':'edited_to_baseline' if label=='baseline_to_edited' else 'baseline_to_edited','donor_is_edited':direction==1,'donor_value':donor[binding],'recipient_value':recipient[binding],'site_role':role,'site_label':role,'site':role,'token_id':tid,'token_text':token_text,'correct_answer':correct,'recipient_correct_logit':tgt_log[correct],'donor_correct_logit':src_log[correct],'patched_correct_logit':patched[correct],'patch_correct_logit_delta':correct_delta,'recipient_correct_minus_old_source':rec_margin,'patched_correct_minus_old_source':pat_margin,'patch_correct_margin_delta':pat_margin-rec_margin if query.startswith('current_') else None})
     qrows.append(row)
   if len(direction_counts)!=2 or len(set(direction_counts.values()))!=1:
    raise RuntimeError(f'incomplete directions generated for {pair_id}: {direction_counts}')
   commit_patch_pair(out,completion_log,pair_id,qrows); done.add(pair_id)
except BaseException:
    raise
save_json({'provenance':provenance(c,a.pairs),'design':'query-conditioned obsolete-binding residual patching','stage':a.stage,'analysis_label':'exploratory_all_positions' if a.all_positions else 'targeted_patch','exploratory':bool(a.all_positions),'cells':cells,'cell_set':a.cell_set,'all_positions':a.all_positions,'frozen_layers':layers,'history_ids':chosen,'discovery_history_ids':ids,'heldout_history_ids':held,'prefix_audit_requested':a.audit_prefix_invariance},str(out)+'.provenance.json')
print(f'saved/resumed patch records at {out}; cells={cells}; histories={len(chosen)}')
