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
CONDITIONS = {'controls': ('live', 'superseded', 'irrelevant'), 'status': ('accepted', 'rejected')}


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
        raise ValueError('behavior scoring supports only controls and status')
    values = list(values)
    if (len(values) < 5 or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError('at least five distinct candidate values are required')
    if n < 1:
        raise ValueError('n must be positive')
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
            proposed_current = condition in ('superseded', 'irrelevant', 'accepted')
            current_fields = {v: f'{"proposed" if proposed_current else "initial"}_{v}' for v in ('x', 'z')}
            obsolete_fields = {v: f'initial_{v}' for v in ('x', 'z')} if condition in ('superseded', 'accepted') else {}
            important = FIELDS if condition != 'live' else FIELDS[:2]
            statuses = {}
            for field in important:
                if kind == 'controls':
                    if field.startswith('initial'):
                        statuses[field] = {'live': 'live_current', 'superseded': 'superseded_initial', 'irrelevant': 'irrelevant_occurrence'}[condition]
                    else:
                        statuses[field] = 'accepted_current' if condition == 'superseded' else 'live_current'
                else:
                    statuses[field] = ('accepted_current' if field.startswith('proposed') else 'superseded_initial') if condition == 'accepted' else ('rejected_update' if field.startswith('proposed') else 'accepted_current')
            fields_to_edit = FIELDS[:2] if kind == 'controls' else FIELDS
            context = {'schema': SCHEMA, 'experiment_kind': kind, 'history_id': hid, 'condition': condition,
                       'seed': seed, 'history_index': index, 'variables': variables, 'orientation': index % 2,
                       'matching_values': original, 'replacement_values': replacements,
                       'semantic_values': {f: original[f] for f in important}, 'semantic_status': statuses,
                       'current_fields': current_fields, 'obsolete_fields': obsolete_fields,
                       'candidate_values': values}
            if kind == 'status':
                context.update(update_accepted=condition == 'accepted', status='YES' if condition == 'accepted' else 'NO')
            for field in fields_to_edit:
                variable = field[-1]  # Schema field identifier, never prompt text.
                for query in ('x', 'z'):
                    pid = f'{hid}:{condition}:{field}:current_{query}'
                    base = _derived({**copy.deepcopy(context), 'pair_id': pid, 'pair_direction': 0,
                                     'example_id': f'{pid}:0', 'query': query, 'query_id': f'current_{query}',
                                     'edited_field': field, 'edited_variable': variable,
                                     'edit_status': statuses[field], 'source_value': original[field],
                                     'replacement_value': replacements[field]})
                    rows.extend((base, edited_member(base)))
    audit_behavior_dataset(rows, kind)
    return rows


def render_behavior_example(ex, tokenizer=None, chat=True):
    """Render exclusively from explicit semantic values and condition metadata."""
    kind, condition, values = ex['experiment_kind'], ex['condition'], ex['semantic_values']
    x, z = ex['variables']
    variables = {'x': x, 'z': z}
    if kind == 'controls':
        if condition == 'live':
            lines = [f'Assignment: {variables[v]} = {values[f"initial_{v}"]}' for v in ('x', 'z')]
        elif condition == 'superseded':
            lines = [f'Initial assignment: {variables[v]} = {values[f"initial_{v}"]}' for v in ('x', 'z')]
            lines += [f'Update: {variables[v]} = {values[f"proposed_{v}"]}' for v in ('x', 'z')]
        elif condition == 'irrelevant':
            lines = [f'Assignment: {variables[v]} = {values[f"proposed_{v}"]}' for v in ('x', 'z')]
            # x/z are analytic slot labels only: neither mention is bound to a variable.
            lines += [f'Unassigned candidate: {values[f"initial_{v}"]}' for v in ('x', 'z')]
        else:
            raise ValueError('unknown control condition')
        question = f'After all updates, what is {variables[ex["query"]]}?' if condition == 'superseded' else f'What is {variables[ex["query"]]}?'
    elif kind == 'status':
        lines = [f'Initial assignment: {variables[v]} = {values[f"initial_{v}"]}' for v in ('x', 'z')]
        for v in ('x', 'z'):
            lines += [f'Proposed update: {variables[v]} = {values[f"proposed_{v}"]}',
                      f'Update accepted: {"YES" if ex["update_accepted"] else "NO"}']
        question = f'After all accepted updates, what is {variables[ex["query"]]}?'
    else:
        raise ValueError('chains and mixed/unknown experiments are not supported')
    prompt = '\n'.join(lines + [question, 'Respond with only the value, with no explanation.'])
    if tokenizer is not None and chat and getattr(tokenizer, 'chat_template', None):
        try:
            return tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False) + 'Answer:'
        except TypeError as exc:
            raise RuntimeError('The tokenizer chat template must accept enable_thinking=False') from exc
    return prompt + '\nAnswer:'


def _context_expected(base, condition):
    """Canonical unedited context for auditing cross-condition matching."""
    fields = FIELDS[:2] if condition == 'live' else FIELDS
    current = condition in ('superseded', 'irrelevant', 'accepted')
    statuses = {}
    for field in fields:
        if base['experiment_kind'] == 'controls':
            statuses[field] = ({'live': 'live_current', 'superseded': 'superseded_initial', 'irrelevant': 'irrelevant_occurrence'}[condition]
                               if field.startswith('initial') else ('accepted_current' if condition == 'superseded' else 'live_current'))
        else:
            statuses[field] = ('accepted_current' if field.startswith('proposed') else 'superseded_initial') if condition == 'accepted' else ('rejected_update' if field.startswith('proposed') else 'accepted_current')
    return ({f: base['matching_values'][f] for f in fields}, statuses,
            {v: f'{"proposed" if current else "initial"}_{v}' for v in ('x', 'z')},
            {v: f'initial_{v}' for v in ('x', 'z')} if condition in ('superseded', 'accepted') else {})


def audit_behavior_dataset(rows, kind=None):
    """Fail closed on incomplete cells, mismatched pairs, or ambiguous roles."""
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
        fields_to_edit = FIELDS[:2] if kind == 'controls' else FIELDS
        if field not in fields_to_edit or query not in ('x', 'z') or base['edited_variable'] != field[-1] or base['query_id'] != f'current_{query}':
            raise ValueError('invalid edited field/variable or query cell')
        expected_pid = f'{base["history_id"]}:{condition}:{field}:current_{query}'
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
        if _derived(copy.deepcopy(base)) != base or edited_member(base) != edit:
            raise ValueError('baseline/edit differ beyond the intended semantic value and its derived answer roles')
        hid = base['history_id']; cell = (condition, field, query)
        if cell in histories[hid]:
            raise ValueError('duplicate history query/edit cell')
        histories[hid][cell] = base
    expected_cells = {(c, f, q) for c in CONDITIONS[kind] for f in (FIELDS[:2] if kind == 'controls' else FIELDS) for q in ('x', 'z')}
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
