"""Independent, explicitly paired version-chain histories and input audits."""
import copy
import hashlib
import itertools
import random
from collections import defaultdict
from pathlib import Path

from src.data.supersession_behavior import _answer_prefix
from src.data.token_validation import continuation_token_id

SCHEMA = 'audited_version_chain_v1'
DEPTHS = (1, 2, 4, 8)


def template_hash():
    return hashlib.sha256(Path(__file__).read_bytes() +
                          Path('src/data/supersession_behavior.py').read_bytes()).hexdigest()


def signature(row):
    versions = row.get('baseline_versions', row.get('versions'))
    return (row['depth'], tuple(versions['x']), tuple(versions['z']))


def render(row, tokenizer=None, chat=True):
    versions, names = row['versions'], row['literal_names']
    lines = ['Initial:', f"{names['x']} = {versions['x'][0]}",
             f"{names['z']} = {versions['z'][0]}", '', 'Updates:']
    for variable in row['block_order']:
        lines.extend(f'{names[variable]} = {value}' for value in versions[variable][1:])
    lines += ['', f"What is {names[row['query']]} after all updates?",
              'Respond with only the value, with no explanation.']
    return _answer_prefix('\n'.join(lines), tokenizer, chat)


def generate(stage, n, depths, values, seed, excluded=()):
    depths = tuple(depths)
    if stage not in ('pilot', 'full') or n < 4 or n % 4:
        raise ValueError('stage must be pilot/full; histories per depth must be a positive multiple of four')
    if not depths or len(set(depths)) != len(depths) or any(d not in DEPTHS for d in depths):
        raise ValueError('depths must be distinct members of 1,2,4,8')
    if len(values) != len(set(values)) or len(values) < 2*(max(depths)+1)+1:
        raise ValueError('need distinct values for both chains and at least one unused replacement (19 at depth 8)')
    rng = random.Random(seed)
    seen = set(excluded)
    rows = []
    for depth in depths:
        for h in range(n):
            for attempt in range(10000):
                sampled = rng.sample(list(values), 2*(depth+1))
                versions = {'x': sampled[:depth+1], 'z': sampled[depth+1:]}
                sig = (depth, tuple(versions['x']), tuple(versions['z']))
                swapped = (depth, sig[2], sig[1])
                if sig not in seen and swapped not in seen:
                    seen.update((sig, swapped))
                    break
            else:
                raise ValueError('could not draw a fresh history')
            spare = [v for v in values if v not in sampled]
            replacements = {(v, i): rng.choice(spare) for v in ('x', 'z') for i in range(depth+1)}
            hid = f'chain_{stage}_{seed}_d{depth}_{h:06d}'
            for variable, index, query in itertools.product(('x', 'z'), range(depth+1), ('x', 'z')):
                source = versions[variable][index]
                replacement = replacements[variable, index]
                pid = f'{hid}:{variable}:v{index}:q{query}'
                for direction in (0, 1):
                    edited = copy.deepcopy(versions)
                    if direction:
                        edited[variable][index] = replacement
                    answer = edited[query][-1]
                    rows.append({'schema': SCHEMA, 'experiment_kind': 'version_chain',
                        'stage': stage, 'seed': seed, 'history_id': hid, 'depth': depth,
                        'history_index': h, 'versions': edited, 'baseline_versions': copy.deepcopy(versions),
                        'literal_names': {'x': 'x', 'z': 'z'} if h % 2 == 0 else {'x': 'z', 'z': 'x'},
                        'orientation': h % 2, 'block_order': ['x', 'z'] if (h//2) % 2 == 0 else ['z', 'x'],
                        'edited_variable': variable, 'version_index': index, 'age_from_current': depth-index,
                        'query': query, 'pair_id': pid, 'pair_direction': direction,
                        'example_id': f'{pid}:{direction}', 'source_value': source,
                        'replacement_value': replacement, 'candidate_values': list(values),
                        'answer': answer, 'roles': {'target': answer}})
    audit(rows)
    return rows


def audit(rows):
    if not rows:
        raise ValueError('empty chain dataset')
    ids, histories, sigs = set(), defaultdict(list), set()
    stage_seed = {(r['stage'], r['seed']) for r in rows}
    if len(stage_seed) != 1 or rows[0]['stage'] not in ('pilot','full'):
        raise ValueError('mixed stages/seeds')
    candidates = rows[0]['candidate_values']
    if len(candidates) != len(set(candidates)):
        raise ValueError('duplicate candidate values')
    for row in rows:
        if row['schema'] != SCHEMA or row['experiment_kind'] != 'version_chain':
            raise ValueError('invalid chain schema')
        if row['example_id'] in ids:
            raise ValueError('duplicate example ID')
        ids.add(row['example_id'])
        if row['candidate_values'] != candidates or row['orientation'] not in (0,1) or tuple(row['block_order']) not in (('x','z'),('z','x')):
            raise ValueError('invalid or changing candidates/orientation/block order')
        histories[row['history_id']].append(row)
    for hid, members in histories.items():
        ref = members[0]
        depth = ref['depth']
        sig = signature(ref)
        swapped = (depth, sig[2], sig[1])
        if sig in sigs or swapped in sigs:
            raise ValueError('duplicated concrete history')
        sigs.add(sig)
        expected = set(itertools.product(('x', 'z'), range(depth+1), ('x', 'z'), (0, 1)))
        observed = [(r['edited_variable'], r['version_index'], r['query'], r['pair_direction']) for r in members]
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError('incomplete query/version/edit-direction cells')
        flat = ref['baseline_versions']['x'] + ref['baseline_versions']['z']
        if depth not in DEPTHS or any(len(ref['baseline_versions'][v]) != depth+1 for v in ('x', 'z')) or len(set(flat)) != len(flat):
            raise ValueError('invalid distinct chain versions/indexing')
        replacements = {}
        for r in members:
            for key in ('baseline_versions', 'depth', 'literal_names', 'orientation', 'block_order', 'candidate_values'):
                if r[key] != ref[key]:
                    raise ValueError(f'history metadata changes: {key}')
            v, i = r['edited_variable'], r['version_index']
            pid = f'{hid}:{v}:v{i}:q{r["query"]}'
            if r['pair_id'] != pid or r['example_id'] != f'{pid}:{r["pair_direction"]}':
                raise ValueError('pair/example ID does not identify its logical cell')
            expected_versions = copy.deepcopy(ref['baseline_versions'])
            if r['source_value'] != expected_versions[v][i] or r['replacement_value'] in flat:
                raise ValueError('invalid source or non-independent replacement')
            if (v, i) in replacements and replacements[v, i] != r['replacement_value']:
                raise ValueError('replacement differs across queries/directions')
            replacements[v, i] = r['replacement_value']
            if r['pair_direction']:
                expected_versions[v][i] = r['replacement_value']
            if r['versions'] != expected_versions or r['age_from_current'] != depth-i:
                raise ValueError('edit changes more than the indexed value or age is invalid')
            if r['answer'] != r['versions'][r['query']][-1] or r['roles'] != {'target': r['answer']}:
                raise ValueError('incorrect current answer')
            if not set(flat + [r['replacement_value']]) <= set(r['candidate_values']):
                raise ValueError('unmapped history values')
        if ref['literal_names'] != ({'x': 'x', 'z': 'z'} if ref['orientation'] == 0 else {'x': 'z', 'z': 'x'}):
            raise ValueError('literal-name orientation mismatch')
    for depth in {r['depth'] for r in rows}:
        refs = [rs[0] for rs in histories.values() if rs[0]['depth'] == depth]
        counts = defaultdict(int)
        for r in refs:
            counts[(r['orientation'], tuple(r['block_order']))] += 1
        if len(counts) != 4 or len(set(counts.values())) != 1:
            raise ValueError('orientation/block-order counterbalance incomplete')
    return rows[0]['stage']


def audit_tokens(rows, tokenizer, token_ids, chat=True):
    audit(rows)
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError('candidate token collisions')
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r['pair_id']][r['pair_direction']] = r
    for members in pairs.values():
        base, edit = members[0], members[1]
        texts = [render(r, tokenizer, chat) for r in (base, edit)]
        enc = [tokenizer(t, add_special_tokens=False, return_offsets_mapping=True) for t in texts]
        bi, ei = enc[0]['input_ids'], enc[1]['input_ids']
        changes = [i for i, (a, b) in enumerate(zip(bi, ei)) if a != b]
        if len(bi) != len(ei) or len(changes) != 1:
            raise ValueError(f'pair is not a one-token edit: {base["pair_id"]}')
        for row, text, tokens in zip((base, edit), texts, enc):
            value = row['replacement_value'] if row['pair_direction'] else row['source_value']
            # Values are unique throughout each history, so this locates exactly the edited assignment.
            start = text.index(f"{row['literal_names'][row['edited_variable']]} = {value}\n")
            start += len(f"{row['literal_names'][row['edited_variable']]} = ")
            positions = [i for i, (a, b) in enumerate(tokens['offset_mapping']) if a < start+len(value) and b > start]
            if positions != changes:
                raise ValueError('changed token lies outside the declared version value')
            for candidate in row['candidate_values']:
                if continuation_token_id(tokenizer, text, ' '+candidate) != token_ids[candidate]:
                    raise ValueError('candidate is not a stable one-token continuation')
    return {'pairs_checked': len(pairs), 'exact_one_input_token_edit': True, 'status': 'passed'}
