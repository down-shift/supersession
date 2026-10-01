"""One evolving chain, one stable binding; independent audited value edits."""
import copy
import hashlib
import itertools
import random
from collections import defaultdict, Counter
from pathlib import Path
from src.data.version_chain import render, signature
from src.data.token_validation import continuation_token_id

SCHEMA = 'single_deep_chain_v1'
DEPTHS = (2, 3, 4)


def template_hash():
    return hashlib.sha256(b''.join(Path(p).read_bytes() for p in
        (__file__, 'src/data/version_chain.py', 'src/data/supersession_behavior.py'))).hexdigest()


def generate(stage, n, depths, values, seed, excluded=()):
    depths = tuple(depths)
    if stage not in ('pilot', 'full') or n < 4 or n % 4:
        raise ValueError('pilot/full requires a positive multiple of four histories per depth')
    if not depths or len(set(depths)) != len(depths) or any(d not in DEPTHS for d in depths):
        raise ValueError('depths must be distinct members of 2,3,4')
    if len(set(values)) != len(values) or len(values) < max(depths)+3:
        raise ValueError('need distinct focal versions, a stable value, and an unused replacement')
    rng, seen, rows = random.Random(seed), set(excluded), []
    for depth in depths:
        for h in range(n):
            focal = 'x' if h % 2 == 0 else 'z'
            other = 'z' if focal == 'x' else 'x'
            orientation = (h//2) % 2
            for _ in range(10000):
                vals = rng.sample(list(values), depth+2)
                versions = {focal:vals[:-1], other:[vals[-1]]}
                sig = (depth,tuple(versions['x']),tuple(versions['z']))
                if sig not in seen and (depth,sig[2],sig[1]) not in seen:
                    seen.update((sig,(depth,sig[2],sig[1])))
                    break
            else:
                raise ValueError('could not draw fresh histories')
            spare = [v for v in values if v not in vals]
            edits = [(focal,i) for i in range(depth+1)] + [(other,0)]
            replacements = {cell:rng.choice(spare) for cell in edits}
            hid = f'single_{stage}_{seed}_d{depth}_{h:06d}'
            for (variable,index), query, direction in itertools.product(edits,('x','z'),(0,1)):
                edited = copy.deepcopy(versions)
                replacement = replacements[variable,index]
                if direction:
                    edited[variable][index] = replacement
                pid = f'{hid}:{variable}:v{index}:q{query}'
                answer = edited[query][-1]
                rows.append({'schema':SCHEMA,'experiment_kind':'single_deep_chain','stage':stage,
                    'seed':seed,'history_id':hid,'history_index':h,'depth':depth,
                    'focal_variable':focal,'distractor_variable':other,
                    'versions':edited,'baseline_versions':copy.deepcopy(versions),
                    'literal_names':{'x':'x','z':'z'} if orientation==0 else {'x':'z','z':'x'},
                    'orientation':orientation,'block_order':[focal],
                    'edited_variable':variable,'edit_role':'focal_chain' if variable==focal else 'stable_control',
                    'version_index':index,'age_from_current':depth-index if variable==focal else None,
                    'query':query,'query_role':'focal' if query==focal else 'distractor',
                    'pair_id':pid,'pair_direction':direction,'example_id':f'{pid}:{direction}',
                    'source_value':versions[variable][index],'replacement_value':replacement,
                    'candidate_values':list(values),'answer':answer,'roles':{'target':answer}})
    audit(rows)
    return rows


def audit(rows):
    if not rows:
        raise ValueError('empty single-chain data')
    if len({(r['stage'],r['seed']) for r in rows})!=1 or rows[0]['stage'] not in ('pilot','full'):
        raise ValueError('mixed/invalid stages or seeds')
    ids, histories, seen = set(), defaultdict(list), set()
    candidates = rows[0]['candidate_values']
    if len(set(candidates)) != len(candidates):
        raise ValueError('duplicate candidate values')
    for r in rows:
        if r['schema']!=SCHEMA or r['experiment_kind']!='single_deep_chain' or r['candidate_values']!=candidates:
            raise ValueError('invalid schema/changing candidate map')
        if r['example_id'] in ids:
            raise ValueError('duplicate example ID')
        ids.add(r['example_id']); histories[r['history_id']].append(r)
    for hid, members in histories.items():
        ref = members[0]; d = ref['depth']; f = ref['focal_variable']; other = ref['distractor_variable']
        if d not in DEPTHS or f not in ('x','z') or other != ('z' if f=='x' else 'x'):
            raise ValueError('invalid focal/distractor/depth')
        versions = ref['baseline_versions']
        if set(versions)!= {'x','z'} or len(versions[f])!=d+1 or len(versions[other])!=1 or ref['block_order']!=[f]:
            raise ValueError('stable distractor must receive no updates')
        flat = versions['x']+versions['z']
        if len(flat)!=len(set(flat)):
            raise ValueError('binding values must be distinct')
        sig = signature(ref)
        if sig in seen or (d,sig[2],sig[1]) in seen:
            raise ValueError('duplicate concrete history')
        seen.add(sig)
        edits = [(f,i) for i in range(d+1)]+[(other,0)]
        expected = {(v,i,q,direction) for (v,i),q,direction in itertools.product(edits,('x','z'),(0,1))}
        cells = [(r['edited_variable'],r['version_index'],r['query'],r['pair_direction']) for r in members]
        if len(cells)!=len(set(cells)) or set(cells)!=expected:
            raise ValueError('incomplete query/version/control/direction cells')
        replacements = {}
        for r in members:
            for key in ('baseline_versions','focal_variable','distractor_variable','depth','literal_names','orientation','block_order'):
                if r[key]!=ref[key]:
                    raise ValueError(f'history changes {key}')
            v,i = r['edited_variable'],r['version_index']
            replacement = r['replacement_value']
            if r['source_value']!=versions[v][i] or replacement in flat or replacement not in candidates:
                raise ValueError('invalid independent replacement/source')
            if (v,i) in replacements and replacements[v,i]!=replacement:
                raise ValueError('replacement changes across matched queries')
            replacements[v,i] = replacement
            target_versions = copy.deepcopy(versions)
            if r['pair_direction']:
                target_versions[v][i] = replacement
            if r['versions']!=target_versions:
                raise ValueError('pair must edit exactly its declared value')
            if r['edit_role']!=('focal_chain' if v==f else 'stable_control') or r['age_from_current']!=(d-i if v==f else None):
                raise ValueError('incorrect edit role/version age')
            if r['query_role']!=('focal' if r['query']==f else 'distractor'):
                raise ValueError('query role mismatch')
            if r['answer']!=target_versions[r['query']][-1] or r['roles']!={'target':r['answer']}:
                raise ValueError('incorrect current answer')
            pid = f'{hid}:{v}:v{i}:q{r["query"]}'
            if r['pair_id']!=pid or r['example_id']!=f'{pid}:{r["pair_direction"]}':
                raise ValueError('incorrect pair ID')
        if not set(flat)<=set(candidates) or ref['orientation'] not in (0,1) or ref['literal_names']!=({'x':'x','z':'z'} if ref['orientation']==0 else {'x':'z','z':'x'}):
            raise ValueError('invalid mapped values or literal orientation')
    for d in {r['depth'] for r in rows}:
        counts = Counter((rs[0]['focal_variable'],rs[0]['orientation']) for rs in histories.values() if rs[0]['depth']==d)
        if set(counts)!={('x',0),('x',1),('z',0),('z',1)} or len(set(counts.values()))!=1:
            raise ValueError('focal variable and literal orientation must be crossed and balanced')
    return rows[0]['stage']


def audit_tokens(rows, tokenizer, token_ids, chat=True):
    audit(rows)
    if set(token_ids)!=set(rows[0]['candidate_values']) or len(set(token_ids.values()))!=len(token_ids):
        raise ValueError('candidate token map missing values/collisions')
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r['pair_id']][r['pair_direction']] = r
    for pair in pairs.values():
        base, edit = pair[0],pair[1]
        texts = [render(r,tokenizer,chat) for r in (base,edit)]
        enc = [tokenizer(t,add_special_tokens=False,return_offsets_mapping=True) for t in texts]
        changes = [i for i,(a,b) in enumerate(zip(enc[0]['input_ids'],enc[1]['input_ids'])) if a!=b]
        if len(enc[0]['input_ids'])!=len(enc[1]['input_ids']) or len(changes)!=1:
            raise ValueError(f'exact one-token edit failed at {base["pair_id"]}')
        for r,t,e in zip((base,edit),texts,enc):
            value = r['replacement_value'] if r['pair_direction'] else r['source_value']
            prefix = f"{r['literal_names'][r['edited_variable']]} = "
            start = t.index(prefix+value+'\n')+len(prefix)
            positions = [i for i,(a,b) in enumerate(e['offset_mapping']) if a<start+len(value) and b>start]
            if positions!=changes:
                raise ValueError('changed input token is outside declared binding value')
            for value in token_ids:
                if continuation_token_id(tokenizer,t,' '+value)!=token_ids[value]:
                    raise ValueError('candidate is not stable one-token continuation')
    return {'pairs_checked':len(pairs),'exact_one_input_token_edit':True,'status':'passed'}
