"""Frozen relational and order robustness design, kept separate from cross_model_v1."""
import random

VERSION = 'cross_model_relational_v2'
COUNTS = {'development': 24, 'frozen_gate': 24, 'confirmatory': 96}
SEEDS = {'development': 20261031, 'frozen_gate': 20261101, 'confirmatory': 20261102}
VALUES = ['amber', 'coral', 'jade', 'pearl', 'slate', 'teal', 'violet', 'ivory']
CONDITIONS = ('superseded', 'early_unassigned', 'late_unassigned', 'entity_mention', 'other_attribute', 'live')


def generate(stage):
    """Create histories and all independent 2x2 order cells deterministically."""
    if stage not in COUNTS:
        raise ValueError(f'unknown stage {stage!r}')
    rng = random.Random(SEEDS[stage])
    names = ('Nora', 'Liam', 'Ava', 'Omar', 'Mila', 'Eli', 'Iris', 'Noah', 'Zoe', 'Theo', 'Maya', 'Leo')
    attrs = ('badge', 'color', 'code', 'label')
    histories, rows = [], []
    seen = set()
    for i in range(COUNTS[stage]):
        vals = rng.sample(VALUES, 4)
        while tuple(vals) in seen:
            vals = rng.sample(VALUES, 4)
        seen.add(tuple(vals))
        a, b, replacement_a, replacement_b = vals
        hid = f'{VERSION}:{stage}:{i:03d}'
        histories.append({'history_id': hid, 'seed': SEEDS[stage], 'history_index': i,
                          'matching_values': {'x': a, 'z': b}, 'replacement_values': {'x': replacement_a, 'z': replacement_b},
                          'entities': [names[(2*i) % len(names)], names[(2*i+1) % len(names)]],
                          'attribute': attrs[i % len(attrs)], 'semantic_orientation': i % 2})
        h = histories[-1]
        for condition in CONDITIONS:
            for historical_order in (0, 1):
                for current_order in (0, 1):
                    if condition in ('early_unassigned', 'late_unassigned', 'entity_mention'):
                        # These controls also cross the placement order of their two mentions.
                        h_order = historical_order
                    else:
                        h_order = historical_order
                    for edited in ('x', 'z'):
                        for query in ('x', 'z'):
                          for direction in (0, 1):
                            pair_id = f'{hid}:{condition}:h{h_order}:c{current_order}:q{query}:e{edited}'
                            base_values = dict(h['matching_values'])
                            source = base_values[edited]
                            replacement = h['replacement_values'][edited]
                            changed = direction == 1
                            base_values[edited] = replacement if changed else source
                            answer = h['matching_values'][query] if condition == 'live' else h['replacement_values'][query]
                            # For superseded/live the current query is the same across paired edits.
                            row = {'schema': VERSION, 'experiment_kind': VERSION, 'protocol': VERSION,
                                'history_id': hid, 'history_index': i, 'seed': h['seed'], 'condition': condition,
                                'historical_entity_order': h_order, 'current_entity_order': current_order,
                                'mention_order': h_order if condition.endswith('unassigned') or condition == 'entity_mention' else None,
                                'entities': h['entities'], 'attribute': h['attribute'], 'query': query,
                                'edited_entity': edited, 'edited_variable': edited, 'edited_field': f'historical_{edited}',
                                'source_value': source, 'replacement_value': replacement,
                                'pair_id': pair_id, 'pair_direction': int(changed), 'example_id': pair_id+f':{int(changed)}',
                                'answer': answer, 'current_answer': answer,
                                'semantic_values': base_values, 'candidate_values': VALUES,
                                'matching_values': h['matching_values'], 'replacement_values': h['replacement_values'],
                                'variables': h['entities'] if h['semantic_orientation'] == 0 else h['entities'][::-1],
                                'orientation': h['semantic_orientation'], 'attribute_relation': 'same' if condition in ('superseded','live','early_unassigned','late_unassigned') else ('mention_only' if condition == 'entity_mention' else 'different'),
                                'control_type': condition, 'edit_status': 'historical_value', 'edited': changed}
                            rows.append(row)
    return histories, rows


def render(row):
    """Concise structurally matched rendering for six preregistered conditions."""
    entities = {'x': row['entities'][0], 'z': row['entities'][1]}
    h_order = ('x','z') if row['historical_entity_order'] == 0 else ('z','x')
    c_order = ('x','z') if row['current_entity_order'] == 0 else ('z','x')
    vals = row['semantic_values']
    attr = row['attribute']
    if row['condition'] == 'superseded':
        lines = [f"At first, {entities[v]}'s {attr} was {vals[v]}." for v in h_order]
        lines += [f"Later, {entities[v]}'s {attr} became {row['replacement_values'][v]}." for v in c_order]
    elif row['condition'] == 'live':
        lines = [f"Initially, {entities[v]}'s {attr} was {vals[v]}." for v in h_order]
        lines += [f"Currently, {entities[v]}'s {attr} remains {row['matching_values'][v]}." for v in c_order]
    elif row['condition'] in ('early_unassigned','late_unassigned'):
        assignments = [f"{entities[v]}'s {attr} is {row['replacement_values'][v]}." for v in c_order]
        mentions = [f"An unrelated note mentioned {vals[v]}." for v in h_order]
        lines = mentions + assignments if row['condition'] == 'early_unassigned' else assignments + mentions
    elif row['condition'] == 'entity_mention':
        lines = [f"An unrelated note about {entities[v]} mentioned {vals[v]}." for v in h_order]
        lines += [f"{entities[v]}'s {attr} is {row['replacement_values'][v]}." for v in c_order]
    elif row['condition'] == 'other_attribute':
        other = 'tag' if attr != 'tag' else 'badge'
        lines = [f"At first, {entities[v]}'s {other} was {vals[v]}." for v in h_order]
        lines += [f"{entities[v]}'s {attr} is {row['replacement_values'][v]}." for v in c_order]
    else:
        raise ValueError('unknown relational condition')
    return '\n'.join(lines + [f"What is {entities[row['query']]}'s current {attr}?", 'Respond with only the value.']) + '\nAnswer:'


def validate(histories, rows, stage):
    if len(histories) != COUNTS[stage]:
        raise ValueError('wrong history count')
    if len({h['history_id'] for h in histories}) != len(histories):
        raise ValueError('duplicate history ids')
    cells = set()
    bypair = {}
    for row in rows:
        if row['schema'] != VERSION or row['seed'] != SEEDS[stage]:
            raise ValueError('protocol/seed mismatch')
        if row['pair_direction'] in bypair.setdefault(row['pair_id'], {}):
            raise ValueError('duplicate pair direction')
        bypair[row['pair_id']][row['pair_direction']] = row
        cells.add((row['history_id'], row['condition'], row['historical_entity_order'], row['current_entity_order'], row['query']))
    for hid in (h['history_id'] for h in histories):
        expected = {(c,ho,co,q) for c in CONDITIONS for ho in (0,1) for co in (0,1) for q in ('x','z')}
        actual = {(c,ho,co,q) for hh,c,ho,co,q in cells if hh == hid}
        if actual != expected: raise ValueError(f'{hid}: incomplete condition/order/query cells')
    for pid, pair in bypair.items():
        if set(pair) != {0,1}: raise ValueError(f'{pid}: unmatched pair')
        before, after = pair[0], pair[1]
        if before['answer'] != after['answer'] or before['current_answer'] != after['current_answer']:
            raise ValueError('historical/control edit changed current answer')
        for key in ('history_id','condition','historical_entity_order','current_entity_order','query','source_value','replacement_value'):
            if before[key] != after[key]: raise ValueError(f'matched edit mismatch: {key}')
    return True
