"""Explicit paired controls/status schema, rendering, and pre-inference audits.

The historical history-only generator is not expanded by parsing prompt strings.
Regenerate those inputs with the current generator to obtain this paired schema.
"""
from __future__ import annotations

import copy
import random
from collections import defaultdict

SCHEMA = 'supersession_behavior_v1'
FIELDS = ('initial_x', 'initial_z', 'proposed_x', 'proposed_z')
CONDITIONS = {'controls': ('live', 'superseded', 'irrelevant'),
              'controls_counterbalanced': ('live', 'superseded', 'irrelevant', 'irrelevant_counterbalanced'),
              'status': ('accepted', 'rejected'), 'status_2x2': ('YY', 'YN', 'NY', 'NN')}


def _derived(row):
    row['current_x'] = row['semantic_values'][row['current_fields']['x']]
    row['current_z'] = row['semantic_values'][row['current_fields']['z']]
    row['answer'] = row[f'current_{row["query"]}']
    row['roles'] = {'target': row['answer']}
    old_field = row['obsolete_fields'].get(row['query'])
    row['stale_value'] = row['semantic_values'][old_field] if old_field else None
    return row


def edited_member(base):
    edit = copy.deepcopy(base)
    edit['semantic_values'][base['edited_field']] = base['replacement_value']
    edit['pair_direction'] = 1
    edit['example_id'] = f'{base["pair_id"]}:1'
    return _derived(edit)


def generate_behavior_pairs(kind, n, values, seed=73021):
    """Cross condition × edited variable/value × queried variable × pair member.

    One history_id is the matched unit across all conditions. Replacement targets
    are fixed across query and status/condition for the same variable.
    """
    if kind not in CONDITIONS:
        raise ValueError('unsupported behavioral dataset kind')
    values = list(values)
    if (len(values) < 5 or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError('at least five distinct candidate values are required')
    if n < 1:
        raise ValueError('n must be positive')
    if kind == 'status_2x2' and n % 2:
        raise ValueError('status_2x2 requires an even history count to counterbalance literal update-block order')
    if n > len(values)*(len(values)-1)*(len(values)-2)*(len(values)-3):
        raise ValueError('too many distinct histories for this candidate pool')
    rng = random.Random(seed)
    rows, seen = [], set()
    for index in range(n):
        while True:
            chosen = rng.sample(values, 4)
            if tuple(chosen) not in seen:
                seen.add(tuple(chosen)); break
        original = dict(zip(FIELDS, chosen))
        spares = [v for v in values if v not in chosen]
        rng.shuffle(spares)
        replacements = {field: spares[(0 if field.endswith('x') else 1) % len(spares)] for field in FIELDS}
        hid = f'{kind}{index:06d}'
        variables = ['x', 'z'] if index % 2 == 0 else ['z', 'x']
        for condition in CONDITIONS[kind]:
            is_control = kind in ('controls', 'controls_counterbalanced')
            if kind == 'status_2x2':
                accepted = {'x': condition[0] == 'Y', 'z': condition[1] == 'Y'}
            else:
                new_is_current = condition in ('superseded', 'irrelevant', 'irrelevant_counterbalanced', 'accepted')
                accepted = {'x': new_is_current, 'z': new_is_current}
            current_fields = {v: f'{"proposed" if accepted[v] else "initial"}_{v}' for v in ('x', 'z')}
            obsolete_fields = {v: f'initial_{v}' for v in ('x', 'z') if accepted[v] and condition in ('superseded', 'accepted', 'YY', 'YN', 'NY')}
            important = FIELDS[:2] if condition == 'live' else FIELDS
            statuses = {}
            for field in important:
                v = field[-1]
                if is_control:
                    if condition == 'live': status = 'live_current'
                    elif condition == 'superseded': status = 'superseded_initial' if field.startswith('initial') else 'accepted_current'
                    elif field.startswith('initial'): status = 'irrelevant_occurrence'
                    else: status = 'live_current'
                elif kind == 'status':
                    status = ('accepted_current' if field.startswith('proposed') else 'superseded_initial') if condition == 'accepted' else ('rejected_update' if field.startswith('proposed') else 'accepted_current')
                else:
                    status = ('accepted_current' if field.startswith('proposed') else 'superseded_initial') if accepted[v] else ('rejected_update' if field.startswith('proposed') else 'accepted_current')
                statuses[field] = status
            fields_to_edit = FIELDS[:2] if is_control else FIELDS
            slot_orders = ('xz', 'zx') if condition == 'irrelevant_counterbalanced' else (None,)
            context = {'schema': SCHEMA, 'experiment_kind': kind, 'history_id': hid, 'condition': condition,
                       'seed': seed, 'history_index': index, 'variables': variables, 'orientation': index % 2,
                       'matching_values': original, 'replacement_values': replacements,
                       'semantic_values': {f: original[f] for f in important}, 'semantic_status': statuses,
                       'current_fields': current_fields, 'obsolete_fields': obsolete_fields,
                       'candidate_values': values}
            if kind == 'status':
                context.update(update_accepted=condition == 'accepted', status='YES' if condition == 'accepted' else 'NO')
            if kind == 'status_2x2':
                context.update(update_accepted_by_variable=accepted, status_by_variable={v: ('accepted' if accepted[v] else 'rejected') for v in ('x', 'z')}, status=condition)
            for slot_order in slot_orders:
                for field in fields_to_edit:
                    variable = field[-1]
                    for query in ('x', 'z'):
                        pid = f'{hid}:{condition}:{field}:current_{query}' + (f':slots_{slot_order}' if slot_order else '')
                        base = _derived({**copy.deepcopy(context), 'pair_id': pid, 'pair_direction': 0,
                                         'example_id': f'{pid}:0', 'query': query, 'query_id': f'current_{query}',
                                         'edited_field': field, 'edited_variable': variable, 'unassigned_slot_order': slot_order,
                                         'edit_status': statuses[field], 'source_value': original[field],
                                         'replacement_value': replacements[field]})
                        rows.extend((base, edited_member(base)))
    audit_behavior_dataset(rows, kind)
    if kind == 'status_2x2':
        audit_status_2x2_update_order(rows)
    return rows


def audit_status_2x2_update_order(rows):
    """Require literal x/z update-block order to be balanced within each status cell."""
    histories = {}
    for row in rows:
        if row.get('experiment_kind') != 'status_2x2':
            raise ValueError('update-order audit requires status_2x2 records')
        key = (row['history_id'], row['condition'])
        signature = (tuple(row['variables']), int(row['orientation']))
        if key in histories and histories[key] != signature:
            raise ValueError('status cell changes literal update-block order within a history')
        histories[key] = signature
    if not histories:
        raise ValueError('no status_2x2 history/order records found')
    by_condition = defaultdict(lambda: defaultdict(int))
    per_history = defaultdict(dict)
    for (hid, condition), (variables, orientation) in histories.items():
        if orientation not in (0, 1) or variables not in (('x','z'),('z','x')):
            raise ValueError('unsupported orientation or literal variable order')
        if (orientation == 0 and variables != ('x','z')) or (orientation == 1 and variables != ('z','x')):
            raise ValueError('orientation does not match literal variable order')
        first_literal = variables[0]
        by_condition[condition][first_literal] += 1
        per_history[hid][condition] = first_literal
    expected_conditions = {'YY','YN','NY','NN'}
    if any(set(cells) != expected_conditions for cells in per_history.values()):
        raise ValueError('each history must include all four independent status cells')
    for condition, counts in by_condition.items():
        if counts.get('x',0) != counts.get('z',0):
            raise ValueError(f'update-block order is not balanced for {condition}: {dict(counts)}')
    return {condition: dict(counts) for condition, counts in sorted(by_condition.items())}


def generate_status_2x2_gate(n, values, seed):
    """Generate 8 unedited task prompts per history, without paired-edit records."""
    if n != 24:
        raise ValueError('status_2x2 competence gate is fixed at 24 fresh histories')
    paired = generate_behavior_pairs('status_2x2', n, values, seed)
    rows = []
    keep = {'schema','history_id','history_index','seed','orientation','variables','condition','status',
            'status_by_variable','update_accepted_by_variable','matching_values','candidate_values',
            'semantic_values','semantic_status','current_fields','obsolete_fields','current_x','current_z',
            'query','query_id','answer','roles'}
    selected = [r for r in paired if r['pair_direction'] == 0 and r['edited_field'] == 'initial_x']
    for source in selected:
        row = {key: source[key] for key in keep}
        row['history_id'] = f"status2x2gate{source['history_index']:06d}"
        row.update(experiment_kind='status_2x2', record_type='status_2x2_competence_gate',
                   example_id=f"{row['history_id']}:{source['condition']}:{source['query']}")
        rows.append(row)
    audit_status_2x2_gate_dataset(rows)
    audit_status_2x2_update_order(rows)
    return rows


def audit_status_2x2_gate_dataset(rows):
    if len(rows) != 24*4*2:
        raise ValueError('status_2x2 competence gate must have exactly 24 histories × four cells × two queries')
    histories = defaultdict(dict); ids = set()
    for row in rows:
        if row.get('record_type') != 'status_2x2_competence_gate' or row.get('experiment_kind') != 'status_2x2':
            raise ValueError('gate dataset contains non-gate examples')
        if any(key in row for key in ('edited_field','pair_id','pair_direction','source_value','replacement_value','identity_transfer')):
            raise ValueError('gate rows must not include counterfactual edit or causal-effect fields')
        if row.get('condition') not in ('YY','YN','NY','NN') or row.get('query') not in ('x','z'):
            raise ValueError('invalid gate status/query cell')
        if row['example_id'] in ids: raise ValueError('duplicate gate example_id')
        ids.add(row['example_id'])
        accepted={v:row['condition'][i]=='Y' for i,v in enumerate(('x','z'))}
        expected_status={v:('accepted' if accepted[v] else 'rejected') for v in ('x','z')}
        if row.get('status_by_variable')!=expected_status or row.get('update_accepted_by_variable')!={v:accepted[v] for v in ('x','z')}:
            raise ValueError('gate semantic status metadata disagrees with status cell')
        if row['current_fields']!={v:f'{"proposed" if accepted[v] else "initial"}_{v}' for v in ('x','z')}:
            raise ValueError('gate current binding metadata mismatch')
        expected_answer=row['semantic_values'][row['current_fields'][row['query']]]
        if row['answer']!=expected_answer or row['roles']!={'target':expected_answer}:
            raise ValueError('gate task answer metadata mismatch')
        key=(row['condition'],row['query'])
        if key in histories[row['history_id']]: raise ValueError('duplicate gate status/query cell')
        histories[row['history_id']][key]=row
    required={(c,q) for c in ('YY','YN','NY','NN') for q in ('x','z')}
    if len(histories)!=24 or any(set(cells)!=required for cells in histories.values()):
        raise ValueError('gate must contain 24 complete histories with all status/query cells')
    signatures=set()
    for hid,cells in histories.items():
        ref=next(iter(cells.values()))
        for row in cells.values():
            for key in ('matching_values','variables','candidate_values','seed','history_index','orientation'):
                if row[key]!=ref[key]: raise ValueError(f'gate history {hid} is not matched on {key}')
        signature=(tuple(ref['variables']),tuple(ref['matching_values'][f] for f in FIELDS))
        if signature in signatures: raise ValueError('duplicate concrete gate history')
        signatures.add(signature)
    return True


def render_behavior_example(ex, tokenizer=None, chat=True):
    """Render exclusively from explicit semantic values and condition metadata."""
    if ex.get('experiment_kind') == 'status_focal':
        from src.data.status_focal import render
        prompt = render(ex)
        return _answer_prefix(prompt, tokenizer, chat)
    if 'prompt_variant' in ex:
        from src.data.status_prompt_gate import prompt_text
        prompt = prompt_text(ex)
        return _answer_prefix(prompt, tokenizer, chat)
    kind, condition, values = ex['experiment_kind'], ex['condition'], ex['semantic_values']
    x, z = ex['variables']
    variables = {'x': x, 'z': z}
    if kind in ('controls', 'controls_counterbalanced'):
        if condition == 'live':
            lines = [f'Assignment: {variables[v]} = {values[f"initial_{v}"]}' for v in ('x', 'z')]
        elif condition == 'superseded':
            lines = [f'Initial assignment: {variables[v]} = {values[f"initial_{v}"]}' for v in ('x', 'z')]
            lines += [f'Update: {variables[v]} = {values[f"proposed_{v}"]}' for v in ('x', 'z')]
        elif condition in ('irrelevant', 'irrelevant_counterbalanced'):
            lines = [f'Assignment: {variables[v]} = {values[f"proposed_{v}"]}' for v in ('x', 'z')]
            order = ex.get('unassigned_slot_order') or 'xz'
            lines += [f'Unassigned candidate: {values[f"initial_{v}"]}' for v in order]
        else:
            raise ValueError('unknown control condition')
        question = f'After all updates, what is {variables[ex["query"]]}?' if condition == 'superseded' else f'What is {variables[ex["query"]]}?'
    elif kind in ('status', 'status_2x2'):
        lines = [f'Initial assignment: {variables[v]} = {values[f"initial_{v}"]}' for v in ('x', 'z')]
        for v in ('x', 'z'):
            lines += [f'Proposed update: {variables[v]} = {values[f"proposed_{v}"]}',
                      f'Update accepted: {"YES" if (ex["update_accepted"] if kind == "status" else ex["update_accepted_by_variable"][v]) else "NO"}']
        question = f'After all accepted updates, what is {variables[ex["query"]]}?'
    else:
        raise ValueError('chains and mixed/unknown experiments are not supported')
    prompt = '\n'.join(lines + [question, 'Respond with only the value, with no explanation.'])
    return _answer_prefix(prompt, tokenizer, chat)


def _answer_prefix(prompt, tokenizer, chat):
    if tokenizer is not None and chat and getattr(tokenizer, 'chat_template', None):
        try:
            return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False) + 'Answer:'
        except TypeError as exc:
            raise RuntimeError('The tokenizer chat template must accept enable_thinking=False') from exc
    return prompt + '\nAnswer:'


def _context_expected(base, condition):
    """Canonical unedited context for auditing cross-condition matching."""
    kind = base['experiment_kind']
    fields = FIELDS[:2] if condition == 'live' else FIELDS
    accepted = ({'x': condition[0] == 'Y', 'z': condition[1] == 'Y'} if kind == 'status_2x2' else
                {v: condition in ('superseded', 'irrelevant', 'irrelevant_counterbalanced', 'accepted') for v in ('x', 'z')})
    current = condition in ('superseded', 'irrelevant', 'irrelevant_counterbalanced', 'accepted')
    statuses = {}
    for field in fields:
        if kind in ('controls', 'controls_counterbalanced'):
            statuses[field] = ('live_current' if condition == 'live' else 'superseded_initial' if condition == 'superseded' and field.startswith('initial') else
                               'accepted_current' if condition == 'superseded' else 'irrelevant_occurrence' if field.startswith('initial') else 'live_current')
        elif kind == 'status':
            statuses[field] = ('accepted_current' if field.startswith('proposed') else 'superseded_initial') if condition == 'accepted' else ('rejected_update' if field.startswith('proposed') else 'accepted_current')
        else:
            statuses[field] = ('accepted_current' if field.startswith('proposed') else 'superseded_initial') if accepted[field[-1]] else ('rejected_update' if field.startswith('proposed') else 'accepted_current')
    obsolete = {v: f'initial_{v}' for v in ('x', 'z') if accepted[v] and condition in ('superseded', 'accepted', 'YY', 'YN', 'NY')}
    return ({f: base['matching_values'][f] for f in fields}, statuses,
            {v: f'{"proposed" if accepted[v] else "initial"}_{v}' for v in ('x', 'z')}, obsolete)


def audit_behavior_dataset(rows, kind=None):
    """Fail closed on incomplete cells, mismatched pairs, or ambiguous roles."""
    if rows and any('prompt_variant' in r for r in rows):
        if kind not in (None, 'status_2x2'):
            raise ValueError('prompt variants are only supported for status_2x2')
        from src.data.status_prompt_gate import audit_final_dataset
        return audit_final_dataset(rows)
    if not rows:
        raise ValueError('dataset is empty')
    kinds = {r.get('experiment_kind') for r in rows}
    if len(kinds) != 1 or next(iter(kinds)) not in CONDITIONS:
        raise ValueError('expected a single controls/status dataset; regenerate history-only or chain inputs')
    actual = next(iter(kinds))
    if kind is not None and actual != kind:
        raise ValueError(f'--kind {kind} differs from dataset kind {actual}')
    kind = actual
    groups, histories, ids = defaultdict(dict), defaultdict(dict), set()
    required = {'schema', 'experiment_kind', 'history_id', 'condition', 'semantic_values', 'semantic_status',
                'current_fields', 'obsolete_fields', 'matching_values', 'replacement_values', 'candidate_values',
                'edited_field', 'edited_variable', 'edit_status', 'source_value', 'replacement_value',
                'query', 'query_id', 'pair_id', 'pair_direction', 'example_id', 'roles', 'answer', 'stale_value', 'variables'}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f'missing semantic metadata {sorted(missing)}; regenerate history-only inputs')
        if row['schema'] != SCHEMA or row['condition'] not in CONDITIONS[kind]:
            raise ValueError('unsupported schema or condition')
        direction = row['pair_direction']
        if type(direction) is not int or direction not in (0, 1):
            raise ValueError('pair_direction must be unambiguous integer 0 or 1')
        if row['example_id'] in ids or direction in groups[row['pair_id']]:
            raise ValueError('duplicate example_id or pair direction')
        if row['example_id'] != f'{row["pair_id"]}:{direction}':
            raise ValueError('example_id does not match pair direction')
        ids.add(row['example_id']); groups[row['pair_id']][direction] = row
    for pid, members in groups.items():
        if set(members) != {0, 1}:
            raise ValueError(f'pair {pid} lacks baseline/edit pair_direction 0 and 1')
        base, edit = members[0], members[1]
        condition, field, query = base['condition'], base['edited_field'], base['query']
        fields_to_edit = FIELDS[:2] if kind in ('controls', 'controls_counterbalanced') else FIELDS
        if field not in fields_to_edit or query not in ('x', 'z') or base['edited_variable'] != field[-1] or base['query_id'] != f'current_{query}':
            raise ValueError('invalid edited field/variable or query cell')
        expected_pid = f'{base["history_id"]}:{condition}:{field}:current_{query}' + (f':slots_{base["unassigned_slot_order"]}' if base.get('unassigned_slot_order') else '')
        if pid != expected_pid:
            raise ValueError('pair_id disagrees with semantic metadata')
        if len(base['variables']) != 2 or len(set(base['variables'])) != 2:
            raise ValueError('need two distinct variable names')
        originals = base['matching_values']
        if set(originals) != set(FIELDS) or len(set(originals.values())) != 4:
            raise ValueError('matching_values must contain four distinct semantic values')
        candidates = base['candidate_values']
        if len(candidates) < 5 or len(candidates) != len(set(candidates)) or not set(originals.values()) <= set(candidates):
            raise ValueError('invalid candidate pool')
        if set(base['replacement_values']) != set(FIELDS) or any(v not in candidates or v in originals.values() for v in base['replacement_values'].values()):
            raise ValueError('replacement collides with an important value or is outside candidates')
        values, statuses, current, obsolete = _context_expected(base, condition)
        if (base['semantic_values'], base['semantic_status'], base['current_fields'], base['obsolete_fields']) != (values, statuses, current, obsolete):
            raise ValueError('baseline context or semantic statuses disagree with explicit matching values')
        if base['source_value'] != values[field] or base['replacement_value'] != base['replacement_values'][field] or base['edit_status'] != statuses[field]:
            raise ValueError('source/replacement/edit status metadata mismatch')
        if kind == 'status' and (type(base.get('update_accepted')) is not bool or base['update_accepted'] != (condition == 'accepted') or base.get('status') != ('YES' if condition == 'accepted' else 'NO')):
            raise ValueError('accepted/rejected semantic status mismatch')
        if kind == 'status_2x2' and (base.get('status') != condition or base.get('status_by_variable') != {v: ('accepted' if condition[i] == 'Y' else 'rejected') for i, v in enumerate(('x', 'z'))}):
            raise ValueError('independent x/z acceptance status metadata mismatch')
        if _derived(copy.deepcopy(base)) != base or edited_member(base) != edit:
            raise ValueError('baseline/edit differ beyond the intended semantic value and its derived answer roles')
        hid = base['history_id']; cell = (condition, field, query, base.get('unassigned_slot_order'))
        if cell in histories[hid]:
            raise ValueError('duplicate history query/edit cell')
        histories[hid][cell] = base
    expected_cells = {(c, f, q, slot) for c in CONDITIONS[kind] for f in (FIELDS[:2] if kind in ('controls', 'controls_counterbalanced') else FIELDS) for q in ('x', 'z') for slot in (('xz', 'zx') if c == 'irrelevant_counterbalanced' else (None,))}
    signatures = set()
    for hid, cells in histories.items():
        if set(cells) != expected_cells:
            raise ValueError(f'history {hid} has missing or unexpected required x/z query cells')
        ref = next(iter(cells.values()))
        for row in cells.values():
            for key in ('matching_values', 'replacement_values', 'variables', 'candidate_values', 'seed', 'history_index', 'orientation'):
                if row.get(key) != ref.get(key):
                    raise ValueError(f'history {hid} is not matched across conditions/queries: {key}')
        signature = (tuple(ref['variables']), tuple(ref['matching_values'][f] for f in FIELDS))
        if signature in signatures:
            raise ValueError('concrete histories are duplicated under different history IDs')
        signatures.add(signature)
    return kind


def audit_candidate_tokens(rows, token_ids):
    if not token_ids or len(set(token_ids.values())) != len(token_ids) or any(type(v) is not int or v < 0 for v in token_ids.values()):
        raise ValueError('validated token map must have distinct nonnegative integer token IDs')
    for row in rows:
        for label in ('source_value', 'replacement_value', 'answer'):
            if row[label] not in token_ids:
                raise ValueError(f'{row["example_id"]}: {label} {row[label]!r} missing from validated token map; regenerate using --token-ids')
        for value in row['semantic_values'].values():
            if value not in token_ids:
                raise ValueError(f'semantic value {value!r} missing from validated token map')
        if not set(row['candidate_values']) <= token_ids.keys():
            raise ValueError('candidate pool includes unmapped values; regenerate using --token-ids')


def audit_tokenized_pairs(rows, tokenizer, token_ids, chat=True):
    from src.data.token_validation import continuation_token_id
    audit_candidate_tokens(rows, token_ids)
    prefixes, pairs = {}, defaultdict(dict)
    for row in rows:
        prompt = render_behavior_example(row, tokenizer, chat)
        if prompt not in prefixes:
            ids = list(tokenizer(prompt, add_special_tokens=False)['input_ids'])
            for value, expected in token_ids.items():
                actual = continuation_token_id(tokenizer, prompt, ' ' + value)
                if actual != expected:
                    raise ValueError(f'{value!r}: unstable one-token continuation/token-map mismatch at {row["example_id"]}')
            prefixes[prompt] = ids
        pairs[row['pair_id']][row['pair_direction']] = prefixes[prompt]
    for pid, members in pairs.items():
        base, edit = members[0], members[1]
        if len(base) != len(edit) or sum(a != b for a, b in zip(base, edit)) != 1:
            raise ValueError(f'{pid}: edit must change exactly one input token and preserve prompt length')
    return {'pairs_checked': len(pairs), 'unique_prefixes_checked': len(prefixes),
            'candidate_count': len(token_ids), 'status': 'passed'}
