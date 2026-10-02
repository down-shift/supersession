"""Frozen fresh-history protocol for query_reactivation_v1, separate from cross_model_v1."""
import hashlib, itertools, random
import json
from collections import defaultdict
from pathlib import Path
from src.data.supersession_behavior import _answer_prefix

SCHEMA='query_reactivation_v1'
SEEDS={'development':20261020,'frozen_gate':20261021,'confirmatory':20261022}
COUNTS={'development':24,'frozen_gate':24,'confirmatory':96}
QUERIES=('current_x','initial_x','current_z','initial_z')
VALUES=('amber','birch','coral','denim','elm','frost','grape','hazel','indigo','jade','khaki','lilac')

def template_hash():
    return hashlib.sha256(Path(__file__).read_bytes()+Path('src/data/supersession_behavior.py').read_bytes()).hexdigest()

def validate_dataset_provenance(dataset_path, provenance_path):
    doc=json.loads(Path(provenance_path).read_text())
    actual=hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    if doc.get('protocol')!=SCHEMA or doc.get('dataset_sha256')!=actual or doc.get('template_sha256')!=template_hash():
        raise ValueError('query_reactivation_v1 dataset provenance/hash mismatch')
    return doc

def seal_artifact(payload):
    body=dict(payload)
    body.pop('seal',None)
    digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {**body,'seal':digest}

def verify_sealed_artifact(document):
    body=dict(document); seal=body.pop('seal',None)
    expected=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if seal!=expected: raise ValueError('query_reactivation_v1 artifact seal mismatch')
    return body

def render(row,tokenizer=None,chat=True):
    x,z=row['variables']; v=dict(row['matching_values']); attr=row['attribute']
    if row.get('stage')=='confirmatory' and row.get('pair_direction')==1:
        v['initial_'+row['edited_binding'][-1]]=row['replacement_value']
    q=row['query_id']; entity=x if q.endswith('_x') else z
    lines=[f'The {attr} assigned to {x} was {v["initial_x"]}.',f'The {attr} assigned to {z} was {v["initial_z"]}.',
      f'Later, {x}’s {attr} was changed to {v["current_x"]}.',f'Later, {z}’s {attr} was changed to {v["current_z"]}.']
    question=(f'What is {entity}’s current {attr}?' if q.startswith('current_') else
              f'What was {entity}’s {attr} originally?')
    return _answer_prefix('\n'.join(lines+[question,'Respond with only the value, with no explanation.']),tokenizer,chat)

def signature(row):
    values=row.get('matching_values')
    if values:
        return tuple(values[k] for k in ('initial_x','initial_z','current_x','current_z'))
    return (row['old_x'],row['old_z'],row['current_x'],row['current_z'])

def generate(stage,n,values=VALUES,seed=None,excluded=()):
    if stage not in ('development','frozen_gate','confirmatory'): raise ValueError('invalid stage')
    if n!=COUNTS[stage]: raise ValueError(f'{stage} requires exactly {COUNTS[stage]} histories')
    seed=SEEDS[stage] if seed is None else seed
    if seed!=SEEDS[stage]: raise ValueError('query_reactivation_v1 stage seeds are frozen')
    if any(r.get('seed')==seed for r in excluded): raise ValueError('stage seed overlaps an excluded dataset')
    if len(values)<6 or len(set(values))!=len(values): raise ValueError('candidate values must be distinct (at least six)')
    seen=set()
    for r in excluded:
        sig=signature(r)
        seen.update((sig,(sig[1],sig[0],sig[3],sig[2])))
    rng=random.Random(seed); rows=[]
    for i in range(n):
        while True:
            a,b,c,d=rng.sample(list(values),4); sig=(a,b,c,d)
            if sig not in seen: seen.update((sig,(sig[1],sig[0],sig[3],sig[2]))); break
        # old values have matched, independent replacement targets; only current histories are queried.
        current={'x':c,'z':d}; matching={'initial_x':a,'initial_z':b,'current_x':c,'current_z':d}
        replacements=dict(zip(('old_x','old_z'),rng.sample([v for v in values if v not in matching.values()],2)))
        hid=f'query_reactivation_{stage}_{seed}_{i:06d}'
        base={'schema':SCHEMA,'stage':stage,'record_type':'competence' if stage!='confirmatory' else 'matched_edit',
          'history_id':hid,'history_index':i,'seed':seed,'orientation':i%2,'variables':['x','z'],
          'variable_pair':['x','z'],'attribute':'color','candidate_values':list(values),'matching_values':matching,
          'current_values':current}
        for query in QUERIES:
            answer=matching['current_'+query[-1]] if query.startswith('current') else matching['initial_'+query[-1]]
            row={**base,'query_id':query,'query':query[-1],'answer':answer,'roles':{'target':answer},'example_id':f'{hid}:{query}'}
            if stage!='confirmatory': rows.append(row)
            else:
                for binding in ('old_x','old_z'):
                    source=matching['initial_'+binding[-1]]
                    replacement=replacements[binding]
                    pid=f'{hid}:{binding}:{query}'
                    for direction in (0,1):
                        changed=answer
                        if direction and query==f'initial_{binding[-1]}': changed=replacement
                        rows.append({**row,'example_id':f'{pid}:{direction}','pair_id':pid,'pair_direction':direction,
                          'edited_binding':binding,'source_value':source,'replacement_value':replacement,
                          'answer':changed,'roles':{'target':changed},'edited_value':replacement if direction else source})
    audit(rows)
    return rows

def audit(rows):
    if not rows: raise ValueError('empty query-reactivation dataset')
    stage=rows[0]['stage']; expected=QUERIES if stage!='confirmatory' else tuple((b,q,d) for b in ('old_x','old_z') for q in QUERIES for d in (0,1))
    histories=defaultdict(dict); ids=set(); signatures=set()
    for r in rows:
        if r.get('schema')!=SCHEMA or r.get('stage')!=stage: raise ValueError('mixed/invalid protocol schema')
        if r['example_id'] in ids: raise ValueError('duplicate example')
        if stage!='confirmatory' and {'pair_id','pair_direction','edited_binding','source_value','replacement_value','identity_transfer'} & r.keys():
            raise ValueError('development/gate must not contain edit or causal-effect fields')
        ids.add(r['example_id']); sig=signature(r)
        histories[r['history_id']][(r['edited_binding'],r['query_id'],r.get('pair_direction')) if stage=='confirmatory' else r['query_id']]=r
    if len(histories)!=COUNTS[stage] or len({r['seed'] for r in rows})!=1:
        raise ValueError('wrong history count or mixed seeds')
    for hid, cells in histories.items():
        if set(cells)!=set(expected): raise ValueError(f'incomplete history cells: {hid}')
        ref=next(iter(cells.values())); sig=signature(ref)
        if sig in signatures: raise ValueError('duplicate concrete history')
        signatures.add(sig)
        vals=list(ref['matching_values'].values())
        if len(set(vals))!=4: raise ValueError('history binding values must be distinct')
        replacements={}
        for r in cells.values():
            if signature(r)!=sig or r['variables']!=ref['variables']: raise ValueError('history context/query match changed')
            correct=ref['matching_values'][('current_' if r['query_id'].startswith('current') else 'initial_')+r['query']]
            if stage!='confirmatory' or r['pair_direction']==0:
                if r['answer']!=correct: raise ValueError('current/historical answer semantics mismatch')
            if r['roles']!={'target':r['answer']}: raise ValueError('query target role differs from answer')
            if stage=='confirmatory':
                if r['edited_binding'] not in ('old_x','old_z'): raise ValueError('only obsolete bindings are edited')
                if r['source_value']!=ref['matching_values']['initial_'+r['edited_binding'][-1]]: raise ValueError('edit source is not old binding')
                if r['replacement_value'] in vals: raise ValueError('replacement collides with history values')
                if r['pair_direction'] not in (0,1) or r['pair_id']!=f'{hid}:{r["edited_binding"]}:{r["query_id"]}' or r['example_id']!=f'{r["pair_id"]}:{r["pair_direction"]}':
                    raise ValueError('invalid matched-pair identifier/direction')
                if r['edited_value']!=(r['replacement_value'] if r['pair_direction'] else r['source_value']):
                    raise ValueError('edited value disagrees with pair direction')
                if r['edited_binding'] in replacements and replacements[r['edited_binding']]!=r['replacement_value']:
                    raise ValueError('replacement changes across query cells')
                replacements[r['edited_binding']]=r['replacement_value']
                expected_edit=(r['replacement_value'] if r['query_id']==f'initial_{r["edited_binding"][-1]}' and r['pair_direction']==1 else correct)
                if r['answer']!=expected_edit or r['roles']!={'target':expected_edit}:
                    raise ValueError('counterfactual answer/orientation mismatch')
    return True
