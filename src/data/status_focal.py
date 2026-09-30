"""Validity-only focal update design and accuracy-only prompt gates."""
import copy, hashlib, itertools, json, random
from collections import defaultdict
from pathlib import Path

from src.data.io import read_jsonl, sha256_file
from src.data.status_prompt_gate import history_signatures as shared_history_signatures

VARIANTS = ('bracketed', 'active_words', 'accepted_words')
LABELS = {'bracketed': ('APPLIED', 'IGNORED'), 'active_words': ('ACTIVE', 'INACTIVE'),
          'accepted_words': ('YES', 'NO')}
PERSISTENCE_VARIANTS = ('persistence_rule', 'accept_reject_rule')
VARIANTS += PERSISTENCE_VARIANTS
LABELS.update({'persistence_rule': ('APPLIED', 'IGNORED'),
               'accept_reject_rule': ('ACCEPTED', 'REJECTED')})
DEVELOPMENT_MIN_ACCURACY = 0.98

def render(row):
    v=row['semantic_values']; names=row['literal_names']; focal=row['focal_variable']
    valid=row['focal_valid']; order=row['update_order']; variant=row['prompt_variant']
    label=LABELS[variant]
    lines=['Initial state:', f"{names['x']} = {v['initial_x']}", f"{names['z']} = {v['initial_z']}", '', 'Updates:']
    for role in order:
        status=(label[0] if (role != focal or valid) else label[1])
        # The nonfocal update is always valid.
        if variant=='bracketed': lines.append(f"{names[role]} -> {v['proposed_'+role]} [{status}]")
        elif variant=='active_words': lines.append(f"Update {names[role]} = {v['proposed_'+role]}; Status: {status}.")
        elif variant=='accepted_words': lines.append(f"{status} update: {names[role]} = {v['proposed_'+role]}")
        elif variant=='persistence_rule': lines.append(f"{names[role]} -> {v['proposed_'+role]} [{status}]")
        else: lines.append(f"{status}: {names[role]} -> {v['proposed_'+role]}.")
    if variant=='bracketed': apply='Use only APPLIED updates.'
    elif variant=='active_words': apply='Use only ACTIVE updates.'
    elif variant=='accepted_words': apply='Use only YES updates.'
    elif variant=='persistence_rule': apply='An APPLIED update replaces its variable value. An IGNORED update leaves its variable unchanged.'
    else: apply='An ACCEPTED update replaces its variable value. A REJECTED update leaves its variable unchanged.'
    query_role=row['query_role']; query=row['query']
    if variant=='persistence_rule': question=f"What is {names[query]} after processing these updates?"
    elif variant=='accept_reject_rule': question=f"What is the value of {names[query]} after processing these decisions?"
    else: question=f"What is {names[query]}?"
    return '\n'.join(lines+['',apply,question,'Respond with only the value, with no explanation.'])

def render_prompt(row,tokenizer,chat):
    from src.data.supersession_behavior import _answer_prefix
    return _answer_prefix(render(row),tokenizer,chat)

def focal_status_char_span(row,text):
    name=row['literal_names'][row['focal_variable']]; value=row['semantic_values']['proposed_'+row['focal_variable']]
    variant=row['prompt_variant']; label=LABELS[variant][0 if row['focal_valid'] else 1]
    if variant in ('bracketed','persistence_rule'): marker=f'{name} -> {value} [{label}]'; start=text.index(marker)+len(marker)-len(label)-1; return start,start+len(label)
    if variant=='active_words': marker=f'Update {name} = {value}; Status: {label}.'; start=text.index(marker)+len(marker)-len(label)-1; return start,start+len(label)
    if variant=='accepted_words': marker=f'{label} update: {name} = {value}'; start=text.index(marker); return start,start+len(label)
    marker=f'{label}: {name} -> {value}.'; start=text.index(marker); return start,start+len(label)

def audit_validity_prompt_alignment(yes,no,tokenizer,chat):
    """Require equal-length APPLIED/IGNORED token sequences with changes confined to their status span."""
    import numpy as np
    yt=render_prompt(yes,tokenizer,chat); nt=render_prompt(no,tokenizer,chat)
    try:
        ye=tokenizer(yt,add_special_tokens=False,return_offsets_mapping=True); ne=tokenizer(nt,add_special_tokens=False,return_offsets_mapping=True)
    except (TypeError,NotImplementedError,KeyError) as exc: raise RuntimeError('validity patch alignment requires fast-tokenizer offsets') from exc
    yi,ni=list(ye['input_ids']),list(ne['input_ids']); yo,noff=ye['offset_mapping'],ne['offset_mapping']
    if len(yi)!=len(ni): raise ValueError('matched APPLIED/IGNORED prompts have unequal token lengths')
    ys,ye_=focal_status_char_span(yes,yt); ns,ne_=focal_status_char_span(no,nt)
    yp=[i for i,(a,b) in enumerate(yo) if a<ye_ and b>ys]; np_=[i for i,(a,b) in enumerate(noff) if a<ne_ and b>ns]
    if not yp or len(yp)!=len(np_): raise ValueError('focal status spans have unequal token lengths')
    differences=[i for i,(a,b) in enumerate(zip(yi,ni)) if a!=b]
    if not differences or not set(differences)<=set(yp)|set(np_): raise ValueError('matched validity prompts differ outside the focal status span')
    if yp!=np_: raise ValueError('focal status spans are not position-aligned')
    return {'token_length':len(yi),'status_positions':yp,'differing_positions':differences,'applied_prompt':yt,'ignored_prompt':nt}

def audit_value_edit_alignment(base,edited,tokenizer,chat):
    """Require an edit to replace exactly one candidate token at the declared value span."""
    bt=render_prompt(base,tokenizer,chat);et=render_prompt(edited,tokenizer,chat)
    try:
        b=tokenizer(bt,add_special_tokens=False,return_offsets_mapping=True);e=tokenizer(et,add_special_tokens=False,return_offsets_mapping=True)
    except (TypeError,NotImplementedError,KeyError) as exc:raise RuntimeError('value-edit audit requires fast-tokenizer offsets') from exc
    bi,ei=list(b['input_ids']),list(e['input_ids']);bo,eo=b['offset_mapping'],e['offset_mapping']
    if len(bi)!=len(ei):raise ValueError('matched value-edit prompts differ in total token length')
    changed=[i for i,(x,y) in enumerate(zip(bi,ei)) if x!=y]
    if len(changed)!=1:raise ValueError('matched value edit must change exactly one input token')
    sv,rv=base['source_value'],base['replacement_value'];bs,es=bt.index(sv),et.index(rv)
    bp=[i for i,(x,y) in enumerate(bo) if x<bs+len(sv) and y>bs]
    ep=[i for i,(x,y) in enumerate(eo) if x<es+len(rv) and y>es]
    if len(bp)!=1 or len(ep)!=1 or bp!=ep or changed[0] not in bp:raise ValueError('edit token differs outside the declared value span')
    return {'position':changed[0],'source_value':sv,'replacement_value':rv,'prompt_length':len(bi)}

def focal_template_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+Path('src/data/supersession_behavior.py').read_bytes()+
                          Path('src/data/status_prompt_gate.py').read_bytes()).hexdigest()

def focal_history_signatures(rows):
    signatures=set()
    for r in rows:
        matching=r.get('matching_values',{})
        if set(('initial_x','initial_z','proposed_x','proposed_z'))<=set(matching):
            sig=tuple(matching[k] for k in ('initial_x','initial_z','proposed_x','proposed_z'))
            signatures.add(sig);signatures.add((sig[1],sig[0],sig[3],sig[2]))
    return signatures

def verify_competence_artifact(path,expected_stage,config_path,token_map_path):
    """Recompute a development/frozen artifact and verify its exact model inputs."""
    import re
    doc=json.loads(Path(path).read_text())
    if doc.get('stage')!=expected_stage or doc.get('template_sha256')!=focal_template_hash(): raise ValueError('gate stage/template mismatch')
    for file_key,hash_key in (('dataset_path','dataset_sha256'),('scores_path','scores_sha256')):
        if not Path(doc[file_key]).is_file() or sha256_file(doc[file_key])!=doc.get(hash_key): raise ValueError(f'gate {file_key} hash mismatch')
    sp=Path(doc.get('scores_provenance_path',''))
    if not sp.is_file() or sha256_file(sp)!=doc.get('scores_provenance_sha256'): raise ValueError('gate scoring provenance hash mismatch')
    ds=read_jsonl(doc['dataset_path']);scores=read_jsonl(doc['scores_path']);recomputed=gate_summary(ds,scores)
    if recomputed!=doc.get('competence_summary'): raise ValueError('gate artifact does not match recomputed competence')
    prov=doc.get('scoring_provenance',{})
    if (prov.get('purpose')!='competence_gate_only' or prov.get('experiment_kind')!='status_focal'
            or prov.get('stage')!=expected_stage or prov.get('causal_effects_computed') is not False):
        raise ValueError('gate scoring provenance is not a competence-only focal run')
    if prov.get('dataset_sha256')!=doc.get('dataset_sha256') or prov.get('scores_sha256') not in (None,doc.get('scores_sha256')):
        raise ValueError('gate scoring provenance input hashes disagree')
    if prov.get('candidate_values')!=doc.get('candidate_values'):raise ValueError('gate candidate vocabulary provenance mismatch')
    if prov.get('config_sha256')!=sha256_file(config_path) or prov.get('token_map_sha256')!=sha256_file(token_map_path): raise ValueError('gate config/token map differs')
    if prov.get('template_sha256')!=focal_template_hash(): raise ValueError('gate scoring template hash differs')
    for k in ('resolved_model_revision','resolved_tokenizer_revision'):
        if not re.fullmatch(r'[0-9a-f]{40}',str(prov.get(k))): raise ValueError(f'gate lacks exact {k}')
    token_doc=json.loads(Path(token_map_path).read_text())
    for k in ('model_revision','tokenizer_revision'):
        if token_doc.get(k) and token_doc[k]!=prov.get('resolved_'+k): raise ValueError(f'frozen token map {k} mismatch')
    if expected_stage=='frozen_gate':
        if not recomputed.get('passed'): raise ValueError('frozen gate competence does not pass on recomputation')
        if doc.get('passed')!=recomputed.get('passed'): raise ValueError('stored pass flag differs from recomputed competence')
    elif expected_stage=='development':
        qualified=development_variant_diagnostic(ds,scores,doc.get('selected_variant'))
        if not qualified['passed'] or doc.get('passed') is not True:
            raise ValueError('development selection did not pass every competence cell')
        if doc.get('development_qualification')!=qualified:
            raise ValueError('stored development qualification differs from recomputed competence')
    if doc.get('selected_variant') not in {r['prompt_variant'] for r in ds}: raise ValueError('selected wording is absent from gate data')
    if doc.get('selected_variant') not in doc.get('alignment_eligible_variants',[]):raise ValueError('selected wording failed focal status token alignment')
    return doc,ds

def generate(kind, n, values, seed, variants=('bracketed',), excluded_signatures=()):
    if kind not in ('development','frozen_gate','confirmatory'): raise ValueError('invalid focal stage')
    if kind=='frozen_gate' and n!=24 or kind=='confirmatory' and n!=96: raise ValueError('frozen gate requires 24 histories; confirmatory requires 96')
    if n<4 or n%4:raise ValueError('focal histories must be a positive multiple of four for independent focal/name counterbalance')
    variants=tuple(variants)
    if len(set(variants))!=len(variants) or not set(variants)<=set(VARIANTS): raise ValueError('invalid or duplicate prompt variants')
    if kind=='development' and not 2<=len(variants)<=3: raise ValueError('prompt development supports two or three explicit variants')
    if len(set(values))!=len(values) or len(values)<5: raise ValueError('at least five distinct values required')
    if kind!='development' and len(variants)!=1: raise ValueError('gate/final generation requires one prompt variant')
    rng=random.Random(seed); rows=[]; seen=set(excluded_signatures)
    for i in range(n):
        while True:
            vals=tuple(rng.sample(list(values),4))
            if vals not in seen: seen.add(vals); break
        semantic=dict(zip(('initial_x','initial_z','proposed_x','proposed_z'),vals))
        focal='x' if i%2==0 else 'z'; names={'x':'x','z':'z'} if (i//2)%2==0 else {'x':'z','z':'x'}
        spare=[v for v in values if v not in vals]; rng.shuffle(spare)
        replacements={f'proposed_{focal}':spare[0],f'initial_{focal}':spare[1%len(spare)]} if spare else {}
        for variant,valid,query_role,order in itertools.product(variants,(True,False),('focal','other'),('focal_first','focal_second')):
            role=focal if query_role=='focal' else ('z' if focal=='x' else 'x')
            seq=(focal, 'z' if focal=='x' else 'x') if order=='focal_first' else (('z' if focal=='x' else 'x'),focal)
            current={r:(f'proposed_{r}' if (r!=focal or valid) else f'initial_{r}') for r in ('x','z')}
            row={'schema':'status_focal_v1','experiment_kind':'status_focal','stage':kind,'history_id':f'focal_{kind}_{seed}_{i:06d}',
                 'history_index':i,'seed':seed,'matching_values':semantic.copy(),'semantic_values':semantic.copy(),
                 'literal_names':names.copy(),'orientation':(i//2)%2,'focal_variable':focal,'focal_valid':valid,
                 'query_role':query_role,'query':role,'update_order':seq,'focal_position':order,
                 'prompt_variant':variant,'current_fields':current,'answer':semantic[current[role]],'candidate_values':list(values)}
            suffix=f"{variant}:{int(valid)}:{query_role}:{order}"
            if kind=='confirmatory':
                for field in (f'proposed_{focal}',f'initial_{focal}'):
                    replacement=replacements[field]
                    pid=f"{row['history_id']}:{suffix}:{field}"
                    base=copy.deepcopy(row);base.update(pair_id=pid,pair_direction=0,edited_field=field,
                        source_value=semantic[field],replacement_value=replacement,example_id=pid+':0')
                    edit=copy.deepcopy(base);edit['pair_direction']=1;edit['example_id']=pid+':1'
                    edit['semantic_values'][field]=replacement
                    edit['answer']=edit['semantic_values'][edit['current_fields'][edit['query']]]
                    rows.extend((base,edit))
            else:
                row['example_id']=f"{row['history_id']}:{suffix}"
                rows.append(row)
    audit(rows, kind)
    return rows

def audit(rows, stage=None):
    if not rows: raise ValueError('empty status_focal data')
    actual={r['stage'] for r in rows}
    if len(actual)!=1 or (stage and actual!={stage}): raise ValueError('mixed focal stages')
    st=next(iter(actual)); groups=defaultdict(dict); ids=set()
    for r in rows:
        if r.get('experiment_kind')!='status_focal' or r.get('schema')!='status_focal_v1': raise ValueError('invalid focal record')
        if st in ('development','frozen_gate') and any(k in r for k in ('pair_id','pair_direction','identity_transfer','causal_effect')): raise ValueError('gate data contains causal fields')
        h=r['history_id']; cell=(r['prompt_variant'],r['focal_valid'],r['query_role'],r['focal_position'],r.get('edited_field'),r.get('pair_direction'))
        if r['example_id'] in ids or cell in groups[h]: raise ValueError('duplicate focal cell')
        ids.add(r['example_id']); groups[h][cell]=r
    variants={r['prompt_variant'] for r in rows}
    if (st=='development' and not 2<=len(variants)<=3) or (st!='development' and len(variants)!=1):raise ValueError('development needs two or three variants; gate/final isolates one')
    if st=='frozen_gate' and len(groups)!=24: raise ValueError('frozen gate requires 24 histories')
    if st=='confirmatory' and len(groups)!=96: raise ValueError('confirmatory focal data requires 96 histories')
    for hid,cells in groups.items():
        ref=next(iter(cells.values()))
        edits=(('proposed_'+ref['focal_variable'],'initial_'+ref['focal_variable']) if st=='confirmatory' else (None,))
        directions=(0,1) if st=='confirmatory' else (None,)
        required=set(itertools.product(variants,(True,False),('focal','other'),('focal_first','focal_second'),edits,directions))
        if set(cells)!=required: raise ValueError('incomplete focal validity/query/position cells')
        for r in cells.values():
            for k in ('matching_values','literal_names','focal_variable','orientation','candidate_values','seed'):
                if r[k]!=ref[k]: raise ValueError(f'history mismatch in {k}')
            if st!='confirmatory' and r['semantic_values']!=ref['matching_values']: raise ValueError('gate prompt changed its matched history')
            if st=='confirmatory' and r.get('pair_direction')==0 and r['semantic_values']!=ref['matching_values']: raise ValueError('baseline prompt changed its matched history')
            if r['current_fields']!={v:('proposed_'+v if v!=ref['focal_variable'] or r['focal_valid'] else 'initial_'+v) for v in ('x','z')}: raise ValueError('validity binding mismatch')
            if (len(r['matching_values'])!=4 or len(set(r['matching_values'].values()))!=4
                    or not set(r['matching_values'].values())<=set(r['candidate_values'])
                    or len(r['candidate_values'])!=len(set(r['candidate_values']))):raise ValueError('invalid matched history/candidate vocabulary')
            focal=r['focal_variable'];other='z' if focal=='x' else 'x'
            expected_order=(focal,other) if r['focal_position']=='focal_first' else (other,focal)
            expected_query=focal if r['query_role']=='focal' else other
            if tuple(r['update_order'])!=expected_order or r['query']!=expected_query:raise ValueError('semantic role/order/query metadata mismatch')
            if r['answer']!=r['semantic_values'][r['current_fields'][r['query']]]: raise ValueError('answer mismatch')
            if st=='confirmatory':
                field=r['edited_field']
                if field not in (f'proposed_{r["focal_variable"]}',f'initial_{r["focal_variable"]}'): raise ValueError('only focal proposed/initial values may be edited')
                if r['source_value']!=r['matching_values'][field] or r['replacement_value'] not in r['candidate_values'] or r['replacement_value'] in r['matching_values'].values(): raise ValueError('invalid matched value edit')
                if r['semantic_values'][field]!=(r['source_value'] if r['pair_direction']==0 else r['replacement_value']): raise ValueError('pair direction/value mismatch')
            if r.get('pair_direction')==1:
                base=next(x for x in rows if x['pair_id']==r['pair_id'] and x['pair_direction']==0)
                diffs=[k for k in base['semantic_values'] if base['semantic_values'][k]!=r['semantic_values'][k]]
                if diffs!=[r['edited_field']] or r['semantic_values'][r['edited_field']]!=r['replacement_value']: raise ValueError('edit must change exactly its declared value')
    if st=='frozen_gate' and len({r['seed'] for r in rows})!=1: raise ValueError('mixed frozen gate seeds')
    if st=='confirmatory':
        for cells in groups.values():
            by={(r['query_role'],r['focal_position'],r['focal_valid'],r['edited_field'],r['pair_direction']):r for r in cells.values()}
            for q,pos in itertools.product(('focal','other'),('focal_first','focal_second')):
                # Compare corresponding edit member as well as semantic prompt cells.
                foc=next(iter(cells.values()))['focal_variable']
                for field,direction in itertools.product((f'proposed_{foc}',f'initial_{foc}'),(0,1)):
                    yes,no=by[(q,pos,True,field,direction)],by[(q,pos,False,field,direction)]
                    if any(yes[k]!=no[k] for k in ('matching_values','literal_names','focal_variable','query','update_order','prompt_variant','edited_field','pair_direction','source_value','replacement_value')):
                        raise ValueError('validity conditions do not share exact matched history')
    if st in ('development','frozen_gate','confirmatory'):
        # Focal role and rendered literal-name orientation are crossed across histories.
        for focal in ('x','z'):
            if sum(next(iter(c.values()))['focal_variable']==focal for c in groups.values()) != len(groups)//2: raise ValueError('focal variable imbalance')
        combos=[(next(iter(c.values()))['focal_variable'],next(iter(c.values()))['orientation']) for c in groups.values()]
        if any(combos.count((f,o))!=len(groups)//4 for f in ('x','z') for o in (0,1)): raise ValueError('focal-variable/orientation crossing is unbalanced')
        for r in rows:
            expected_names={'x':'x','z':'z'} if r['orientation']==0 else {'x':'z','z':'x'}
            if r['literal_names']!=expected_names: raise ValueError('literal-name orientation metadata mismatch')
    return st

def gate_summary(dataset,scores):
    audit(dataset); expected={r['example_id']:r for r in dataset}
    if {r.get('example_id') for r in scores}!=set(expected): raise ValueError('scores do not exactly cover gate')
    cells=defaultdict(list)
    for s in scores:
        if any(k in s for k in ('identity_transfer','candidate_logits','causal_effect')): raise ValueError('causal output prohibited')
        r=expected[s['example_id']]
        if any(s.get(k)!=v for k,v in r.items()): raise ValueError('scored prompt metadata differs from gate dataset')
        if s.get('full_vocab_next_token_accuracy') not in (0,1) or type(s.get('target_rank')) is not int or s['target_rank']<1:
            raise ValueError('missing/invalid full-vocabulary competence metrics')
        if s.get('candidate_accuracy') not in (0,1) or type(s.get('candidate_target_rank')) is not int or s['candidate_target_rank']<1:
            raise ValueError('missing/invalid candidate competence metrics')
        if type(s.get('validity_alignment_passed')) is not bool:raise ValueError('missing focal status tokenizer alignment audit')
        cells[(r['prompt_variant'],r['focal_valid'],r['query_role'],r['focal_position'],r['focal_variable'],r['orientation'])].append(s)
    details=[]
    for key,ss in sorted(cells.items(),key=str):
        details.append({'cell':list(key),'n':len(ss),'full_vocab_accuracy':sum(x['full_vocab_next_token_accuracy'] for x in ss)/len(ss),
                        'target_rank_max':max(x['target_rank'] for x in ss),'candidate_accuracy':sum(x.get('candidate_accuracy',0) for x in ss)/len(ss),
                        'candidate_rank_max':max(x.get('candidate_target_rank',x['target_rank']) for x in ss)})
    expected={(v,valid,q,pos,focal,orientation) for v in {r['prompt_variant'] for r in dataset} for valid in (True,False) for q in ('focal','other') for pos in ('focal_first','focal_second') for focal in ('x','z') for orientation in (0,1)}
    complete=set(cells)==expected
    alignment_passed=all(s['validity_alignment_passed'] for s in scores)
    passed=complete and alignment_passed and all(x['full_vocab_accuracy']==1 and x['target_rank_max']==1 and
                            x['candidate_accuracy']==1 and x['candidate_rank_max']==1 for x in details)
    return {'stage':dataset[0]['stage'],'passed':passed,'complete_cells':complete,'alignment_passed':alignment_passed,'cells':details,'causal_effects_computed':False}

def development_variant_diagnostic(dataset,scores,variant,min_accuracy=DEVELOPMENT_MIN_ACCURACY):
    """Require near-perfect accuracy and rank-one answers in every competence cell."""
    if audit(dataset)!='development':raise ValueError('development qualification requires development data')
    summary=gate_summary(dataset,scores)
    cells=[c for c in summary['cells'] if c['cell'][0]==variant]
    expected=set(itertools.product((True,False),('focal','other'),('focal_first','focal_second'),('x','z'),(0,1)))
    observed={tuple(c['cell'][1:]) for c in cells}
    complete=observed==expected
    variant_scores=[s for s in scores if s['prompt_variant']==variant]
    alignment=bool(variant_scores) and all(s['validity_alignment_passed'] for s in variant_scores)
    failed=[c for c in cells if c['full_vocab_accuracy']<min_accuracy or c['candidate_accuracy']<min_accuracy
            or c['target_rank_max']!=1 or c['candidate_rank_max']!=1]
    return {'variant':variant,'passed':complete and alignment and not failed,
            'complete_cells':complete,'cell_count':len(cells),'expected_cell_count':len(expected),
            'alignment_passed':alignment,'minimum_cell_accuracy':min_accuracy,
            'failed_cells':failed}
