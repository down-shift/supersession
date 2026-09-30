"""Validity-only focal update design and accuracy-only prompt gates."""
import itertools, random
from collections import defaultdict

VARIANTS = ('bracketed', 'active_words', 'accepted_words')
LABELS = {'bracketed': ('APPLIED', 'IGNORED'), 'active_words': ('ACTIVE', 'INACTIVE'),
          'accepted_words': ('YES', 'NO')}

def require_passing_frozen_gate(gate):
    if gate.get('stage')!='frozen_gate' or gate.get('passed') is not True or gate.get('causal_effects_computed') is not False:
        raise ValueError('frozen-gate artifact is not explicitly passing and competence-only')
    if gate.get('selected_variant') not in VARIANTS: raise ValueError('frozen-gate artifact lacks a selected prompt variant')
    return gate['selected_variant']

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
        else: lines.append(f"{status} update: {names[role]} = {v['proposed_'+role]}")
    apply='Use only APPLIED updates.' if variant=='bracketed' else ('Use only ACTIVE updates.' if variant=='active_words' else 'Use only YES updates.')
    query_role=row['query_role']; query=row['query']
    return '\n'.join(lines+['',apply,f"What is {names[query_role]}?",'Respond with only the value, with no explanation.'])

def generate(kind, n, values, seed, variants=('bracketed',)):
    if kind not in ('development','frozen_gate','confirmatory'): raise ValueError('invalid focal stage')
    if kind=='frozen_gate' and n!=24 or kind=='confirmatory' and n!=96: raise ValueError('frozen gate requires 24 histories; confirmatory requires 96')
    variants=tuple(variants)
    if len(set(variants))!=len(variants) or not set(variants)<=set(VARIANTS): raise ValueError('invalid or duplicate prompt variants')
    if kind=='development' and not 2<=len(variants)<=3: raise ValueError('prompt development supports two or three explicit variants')
    if len(set(values))!=len(values) or len(values)<5: raise ValueError('at least five distinct values required')
    if kind!='development' and len(variants)!=1: raise ValueError('gate/final generation requires one prompt variant')
    rng=random.Random(seed); rows=[]; seen=set()
    for i in range(n):
        while True:
            vals=tuple(rng.sample(list(values),4))
            if vals not in seen: seen.add(vals); break
        semantic=dict(zip(('initial_x','initial_z','proposed_x','proposed_z'),vals))
        focal='x' if i%2==0 else 'z'; names={'x':'x','z':'z'} if (i//2)%2==0 else {'x':'z','z':'x'}
        for variant,valid,query_role,order in itertools.product(variants,(True,False),('focal','other'),('focal_first','focal_second')):
            role=focal if query_role=='focal' else ('z' if focal=='x' else 'x')
            seq=(focal, 'z' if focal=='x' else 'x') if order=='focal_first' else (('z' if focal=='x' else 'x'),focal)
            current={r:(f'proposed_{r}' if (r!=focal or valid) else f'initial_{r}') for r in ('x','z')}
            row={'schema':'status_focal_v1','experiment_kind':'status_focal','stage':kind,'history_id':f'focal_{kind}_{seed}_{i:06d}',
                 'history_index':i,'seed':seed,'matching_values':semantic.copy(),'semantic_values':semantic.copy(),
                 'literal_names':names.copy(),'orientation':(i//2)%2,'focal_variable':focal,'focal_valid':valid,
                 'query_role':query_role,'query':role,'update_order':seq,'focal_position':order,
                 'prompt_variant':variant,'current_fields':current,'answer':semantic[current[role]],'candidate_values':list(values)}
            row['example_id']=f"{row['history_id']}:{variant}:{int(valid)}:{query_role}:{order}"
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
        h=r['history_id']; cell=(r['prompt_variant'],r['focal_valid'],r['query_role'],r['focal_position'])
        if r['example_id'] in ids or cell in groups[h]: raise ValueError('duplicate focal cell')
        ids.add(r['example_id']); groups[h][cell]=r
    variants={r['prompt_variant'] for r in rows}; required=set(itertools.product(variants,(True,False),('focal','other'),('focal_first','focal_second')))
    if st=='frozen_gate' and len(groups)!=24: raise ValueError('frozen gate requires 24 histories')
    if st=='confirmatory' and len(groups)!=96: raise ValueError('confirmatory focal data requires 96 histories')
    for hid,cells in groups.items():
        if set(cells)!=required: raise ValueError('incomplete focal validity/query/position cells')
        ref=next(iter(cells.values()))
        for r in cells.values():
            for k in ('matching_values','semantic_values','literal_names','focal_variable','orientation','candidate_values','seed'):
                if r[k]!=ref[k]: raise ValueError(f'history mismatch in {k}')
            if r['current_fields']!={v:('proposed_'+v if v!=ref['focal_variable'] or r['focal_valid'] else 'initial_'+v) for v in ('x','z')}: raise ValueError('validity binding mismatch')
            if r['answer']!=r['semantic_values'][r['current_fields'][r['query']]]: raise ValueError('answer mismatch')
            if set(r['matching_values'].values())!=set(r['semantic_values'].values()): raise ValueError('matched history values changed')
    if st=='frozen_gate' and len({r['seed'] for r in rows})!=1: raise ValueError('mixed frozen gate seeds')
    if st=='confirmatory':
        for cells in groups.values():
            by={(r['query_role'],r['focal_position'],r['focal_valid']):r for r in cells.values()}
            for q,pos in itertools.product(('focal','other'),('focal_first','focal_second')):
                yes,no=by[(q,pos,True)],by[(q,pos,False)]
                if any(yes[k]!=no[k] for k in ('matching_values','semantic_values','literal_names','focal_variable','query','update_order','prompt_variant')):
                    raise ValueError('validity conditions do not share exact matched history')
    if st=='frozen_gate':
        # Every named design cell must have all histories; focal x/z and literal orientation are balanced.
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
        cells[(r['prompt_variant'],r['focal_valid'],r['query_role'],r['focal_position'],r['focal_variable'],r['orientation'])].append(s)
    details=[]
    for key,ss in sorted(cells.items(),key=str):
        details.append({'cell':key,'n':len(ss),'full_vocab_accuracy':sum(x['full_vocab_next_token_accuracy'] for x in ss)/len(ss),
                        'target_rank_max':max(x['target_rank'] for x in ss),'candidate_accuracy':sum(x.get('candidate_accuracy',0) for x in ss)/len(ss),
                        'candidate_rank_max':max(x.get('candidate_target_rank',x['target_rank']) for x in ss)})
    expected={(v,valid,q,pos,focal,orientation) for v in {r['prompt_variant'] for r in dataset} for valid in (True,False) for q in ('focal','other') for pos in ('focal_first','focal_second') for focal in ('x','z') for orientation in (0,1)}
    complete=set(cells)==expected
    passed=complete and all(x['full_vocab_accuracy']==1 and x['target_rank_max']==1 and
                            x['candidate_accuracy']==1 and x['candidate_rank_max']==1 for x in details)
    return {'stage':dataset[0]['stage'],'passed':passed,'complete_cells':complete,'cells':details,'causal_effects_computed':False}
