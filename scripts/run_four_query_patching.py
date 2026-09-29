#!/usr/bin/env python3
"""Resumable donor-oriented residual patching for the frozen four-query task."""
import argparse, collections, hashlib, json
from pathlib import Path
from tqdm.auto import tqdm
from src.data.io import read_jsonl
from src.data.generate import render_example, query_variable_span
from src.experiments.patching import patch_sweep, partition_history_ids, focal_cells, canonical_site_role, capture_run
from src.models.loader import load_model
from src.utils import load_config, provenance, save_json

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/four_query_288.yaml'); p.add_argument('--pairs',required=True); p.add_argument('--token-ids',required=True); p.add_argument('--output',required=True)
p.add_argument('--stage',choices=('discovery','heldout'),default='discovery'); p.add_argument('--n-histories',type=int); p.add_argument('--discovery-histories',type=int,default=24); p.add_argument('--seed',type=int,default=20260929)
p.add_argument('--layers'); p.add_argument('--all-positions',action='store_true'); p.add_argument('--cell-set',choices=('focal','full'),default='focal'); p.add_argument('--resume',action='store_true'); p.add_argument('--audit-prefix-invariance',type=int,default=0)
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
out=Path(a.output); manifest=Path(str(out)+'.run.json'); out.parent.mkdir(parents=True,exist_ok=True)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
try: commit=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True,stderr=__import__('subprocess').DEVNULL).strip()
except Exception: commit=None
fingerprint={'pairs_sha256':sha(a.pairs),'token_map_sha256':sha(a.token_ids),'config_sha256':sha(a.config),'model_id':c['model']['id'],'model_revision':c['model'].get('revision'),'prompt_variant':sorted({ex.get('prompt_variant') for h in histories.values() for pair in h.values() for ex in pair.values()}),'stage':a.stage,'cells':cells,'seed':a.seed,'discovery_history_count':a.discovery_histories,'history_ids':chosen,'n_histories':limit,'layers':layers,'all_positions':a.all_positions,'cell_set':a.cell_set,'patch_code_version':commit}
fingerprint['patch_code_sha256']=hashlib.sha256(Path(__file__).read_bytes()+Path(__import__('src.experiments.patching',fromlist=['__file__']).__file__).read_bytes()+Path(__import__('src.models.hooks',fromlist=['__file__']).__file__).read_bytes()).hexdigest()
if a.resume:
    if not manifest.exists(): raise ValueError('--resume requires patch run manifest')
    saved=json.loads(manifest.read_text())
    if saved!=fingerprint: raise ValueError('cannot resume: patch checkpoint fingerprint differs; use a fresh output path')
    raw=out.read_bytes() if out.exists() else b''
    if raw and not raw.endswith(b'\n'): raise ValueError('incomplete final JSONL line; repair/remove it before resuming')
    checkpoint=collections.defaultdict(set)
    for line in raw.splitlines():
        rec=json.loads(line); checkpoint[rec['pair_id']].add(rec['direction'])
    partial=[k for k,v in checkpoint.items() if v!={'baseline_to_edited','edited_to_baseline'}]
    if partial: raise ValueError(f'checkpoint has incomplete logical patch pairs ({len(partial)}); repair/remove those pair records before resuming')
    done=set(checkpoint)
else:
    if out.exists() or manifest.exists(): raise FileExistsError(f'{out} exists; use --resume or a new path')
    manifest.write_text(json.dumps(fingerprint,indent=2)+'\n'); out.touch(); done=set()
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
    audit=[]; device=next(model.parameters()).device
    audit_ids=chosen[:a.audit_prefix_invariance]
    for hid in audit_ids:
        acts_by_query={}
        for query in ('current_x','current_z','initial_x','initial_z'):
            pair=allpairs.get(f'{hid}:old_x:{query}')
            if not pair: raise ValueError(f'prefix audit cannot find old_x/{query} for {hid}')
            ex=pair[0]; rendered=render_example(ex,tok,chat=c['model'].get('chat_template',True)); encoded=tok(rendered,return_tensors='pt',add_special_tokens=False,return_offsets_mapping=True)
            offsets=encoded.pop('offset_mapping')[0].tolist(); start=rendered.index(ex['old_x']); spans=[i for i,(lo,hi) in enumerate(offsets) if lo<start+len(ex['old_x']) and hi>start]
            _,acts=capture_run(model,encoded.to(device)); acts_by_query[query]=(spans,acts)
        for layer in acts_by_query['current_x'][1]:
            reference=acts_by_query['current_x'][1][layer][0,acts_by_query['current_x'][0],:]
            maximum=0.0
            for query in ('current_z','initial_x','initial_z'):
                spans,acts=acts_by_query[query]; diff=(acts[layer][0,spans,:]-reference).abs().max().item(); maximum=max(maximum,diff)
            audit.append({'history_id':hid,'layer':int(layer),'max_abs_difference':maximum,'passed':maximum<=1e-4})
    save_json({'tolerance':1e-4,'histories':audit_ids,'per_layer':audit,'passed':all(x['passed'] for x in audit)},str(out)+'.prefix_invariance.json')
    if any(not x['passed'] for x in audit): raise ValueError('causal-prefix invariance audit exceeded 1e-4')
records=out.open('a',encoding='utf8')
try:
 for hid in tqdm(chosen,desc=f'{a.stage} patch histories'):
  for binding,query in cells:
   pair=histories[hid][(binding,query)]; base,edited=pair[0],pair[1]
   pair_id=base['pair_id']
   if pair_id in done: continue
   qrows=[]
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
    if var_start>=0: sites['current_assignment_variable']=positions(var_start+len('Update: '),var_start+len('Update: ')+len(var))
    qstart,qend=query_variable_span(donor,text); sites['query_variable']=positions(qstart,qend); sites['final_preanswer']=[len(enc['input_ids'])-1]
    selected=sorted(set(range(len(enc['input_ids'])) if a.all_positions else (p for ps in sites.values() for p in ps)))
    table=patch_sweep(model,tok,donor,recipient,token_ids,chat=c['model'].get('chat_template',True),positions=selected,layers=layers)
    role_at={pos:role for role,ps in sites.items() for pos in ps}
    for row in table:
     pos=row['position']; role=role_at.get(pos,'other_position'); ids_at=enc['input_ids']; tid=int(ids_at[pos]); token_text=tok.decode([tid],skip_special_tokens=False)
     correct=recipient['answer']; dk=token_ids[correct]; source_old=recipient[old_key]
     src_log=row['source_logits']; tgt_log=row['target_logits']; patched=row['patched_logits']
     # candidate logits are keyed by candidate strings in frozen token map
     correct_delta=patched[correct]-tgt_log[correct]
     rec_margin=tgt_log[correct]-tgt_log.get(source_old,0.0); pat_margin=patched[correct]-patched.get(source_old,0.0)
     row.update({'pair_id':pair_id,'history_id':hid,'query_id':query,'edited_binding':binding,'direction':label,'donor_pair_direction':label,'recipient_pair_direction':'edited_to_baseline' if label=='baseline_to_edited' else 'baseline_to_edited','donor_is_edited':direction==1,'donor_value':donor[binding],'recipient_value':recipient[binding],'site_role':role,'site_label':role,'site':role,'token_id':tid,'token_text':token_text,'correct_answer':correct,'recipient_correct_logit':tgt_log[correct],'donor_correct_logit':src_log[correct],'patched_correct_logit':patched[correct],'patch_correct_logit_delta':correct_delta,'recipient_correct_minus_old_source':rec_margin,'patched_correct_minus_old_source':pat_margin,'patch_correct_margin_delta':pat_margin-rec_margin if query.startswith('current_') else None})
     records.write(json.dumps(row,sort_keys=True)+'\n')
    records.flush(); done.add(pair_id)
finally: records.close()
save_json({'provenance':provenance(c,a.pairs),'design':'query-conditioned obsolete-binding residual patching','stage':a.stage,'cells':cells,'cell_set':a.cell_set,'all_positions':a.all_positions,'frozen_layers':layers,'history_ids':chosen,'discovery_history_ids':ids,'heldout_history_ids':held,'prefix_audit_requested':a.audit_prefix_invariance},str(out)+'.provenance.json')
print(f'saved/resumed patch records at {out}; cells={cells}; histories={len(chosen)}')
