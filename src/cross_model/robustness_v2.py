"""Relational/order histories; all values, orders and paired edits are explicit."""
import copy
import json
import random
from itertools import product
from pathlib import Path

from src.cross_model.protocol import VALUES, digest
from src.data.supersession_behavior import FIELDS, _answer_prefix, _derived

VERSION = 'cross_model_relational_v2'
DESIGN_REVISION = 'factorial_relation_counterbalanced_geometryfix_exclusions_20261003'
COUNTS = {'development': 24, 'frozen_gate': 24, 'confirmatory': 96}
SEEDS = {'validation': 20261030, 'development': 20261031,
         'frozen_gate': 20261101, 'confirmatory': 20261102}
CONDITIONS = ('superseded', 'early_unassigned', 'late_unassigned',
              'entity_mention', 'other_attribute', 'live')
NAMES = ('Nora', 'Liam', 'Ava', 'Omar', 'Mila', 'Eli', 'Iris', 'Noah', 'Zoe', 'Theo', 'Maya', 'Leo')
ATTRIBUTES = ('badge', 'color', 'code', 'label')
ENTITY_PAIRS = tuple(zip(NAMES[::2], NAMES[1::2]))
ALT_RELATIONS = ('team', 'project')
# Confirmation crosses pair, target attribute and orientation twice, with one
# replicate per alternate relation. The 24-history table balances pair and
# attribute margins and both alternate relations within each pair/attribute.
FRACTIONAL_RELATION_ONE = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
CONFIRMATION_ALLOCATION = tuple(
    (pair, attribute, orientation, replicate)
    for pair in range(6) for attribute in range(4)
    for orientation in (0, 1) for replicate in (0, 1))
FRACTIONAL_ALLOCATION = tuple(
    (pair, attribute, (pair + attribute) % 2,
     0 if attribute in FRACTIONAL_RELATION_ONE[pair] else 1)
    for pair in range(6) for attribute in range(4))
ALLOCATION_TABLES = {'development': FRACTIONAL_ALLOCATION,
                     'frozen_gate': FRACTIONAL_ALLOCATION,
                     'confirmatory': CONFIRMATION_ALLOCATION}
ROWS_PER_HISTORY = 6 * 2 * 2 * 2 * 2 * 2
EXCLUSION_LEDGER = Path(__file__).resolve().parents[2] / 'configs/cross_model_relational_v2_geometryfix_exclusions/prior_history_exclusions.json'
RENDERING = {
    'historical_assignment': 'Previously, {entity}’s {attribute} was {value}.',
    'current_assignment': 'Currently, {entity}’s {attribute} is {value}.',
    'unassigned': 'The unassigned {attribute} value was {value}.',
    'entity_mention': '{entity} mentioned {value} in an unrelated note.',
    'query': 'What is {entity}’s current {attribute}?',
    'instruction': 'Respond with only the value, with no explanation.',
}


def concrete_signature(row):
    """Physical entity/binding signature, ignoring IDs, edits, template and donors.

    This also understands saved v1 rows and the faulty first preview schema.
    Replacement choice must not make a repeated semantic history appear fresh.
    """
    values = row['matching_values']
    if set(values) == set(FIELDS):
        names = row['variables']
        bindings = [(names[i], values[f'initial_{v}'], values[f'proposed_{v}'])
                    for i, v in enumerate(('x', 'z'))]
    elif set(values) == {'x', 'z'} and set(row.get('replacement_values', {})) == {'x', 'z'}:
        # The original preview renderer ignored variables and used entities.
        names = row['entities']
        bindings = [(names[i], values[v], row['replacement_values'][v])
                    for i, v in enumerate(('x', 'z'))]
    else:
        raise ValueError('cannot reconstruct concrete entity/attribute history signature')
    if len(names) != 2 or len(set(names)) != 2 or not row.get('attribute'):
        raise ValueError('concrete history requires two named entities and an attribute')
    return digest({'bindings': sorted(bindings), 'attribute': row['attribute']})


def _history_definitions(stage):
    if stage not in SEEDS:
        raise ValueError(f'unknown stage {stage!r}')
    count = 4 if stage == 'validation' else COUNTS[stage]
    rng = random.Random(SEEDS[stage])
    ledger = json.loads(EXCLUSION_LEDGER.read_text())
    if ledger.get('design_revision') != DESIGN_REVISION:
        raise ValueError('wrong fixed prior-history exclusion ledger')
    # Fixed structural rejection within the same seeded stream, in stage order.
    # There is no seed retry and no dependence on competence or causal outcomes.
    stages = list(SEEDS)
    seen = set(ledger['concrete_signatures'])
    for earlier in stages[:stages.index(stage)]:
        seen.update(h['concrete_signature'] for h in _history_definitions(earlier))
    histories = []
    allocation = ALLOCATION_TABLES.get(stage)
    if stage == 'validation':
        allocation = FRACTIONAL_ALLOCATION[:count]
    if allocation is None or len(allocation) != count:
        raise ValueError(f'no exact allocation table for {stage}')
    for index, (pair_index, attribute_index, orientation, relation_index) in enumerate(allocation):
        entities = list(ENTITY_PAIRS[pair_index])
        h = {'history_index': index, 'seed': SEEDS[stage], 'stage': stage,
             'entities': entities, 'variables': entities[::1 if orientation == 0 else -1],
             'attribute': ATTRIBUTES[attribute_index], 'orientation': orientation,
             'allocation_cell': {'entity_pair': pair_index, 'attribute': attribute_index,
                                 'orientation': orientation},
             'other_attribute': ALT_RELATIONS[relation_index]}
        while True:
            # Neither replacement identity is any historical or current identity.
            chosen = rng.sample(VALUES, 6)
            h['matching_values'] = dict(zip(FIELDS, chosen[:4]))
            h['replacement_values'] = {'initial_x': chosen[4], 'initial_z': chosen[5]}
            signature = concrete_signature(h)
            if signature not in seen:
                seen.add(signature)
                break
        h['concrete_signature'] = signature
        h['history_id'] = f'{VERSION}:{stage}:{index:03d}:{signature[:12]}'
        histories.append(copy.deepcopy(h))
    return histories


def required_cells():
    return set(product(CONDITIONS, (0, 1), (0, 1), ('x', 'z'), ('x', 'z'), (0, 1)))


def _member(h, condition, historical_order, current_order, edited_variable, query, direction):
    live = condition == 'live'
    values = {f: h['matching_values'][f] for f in (FIELDS[:2] if live else FIELDS)}
    field = f'initial_{edited_variable}'
    source, replacement = h['matching_values'][field], h['replacement_values'][field]
    if direction:
        values[field] = replacement
    status = 'live_current' if live else ('superseded_initial' if condition == 'superseded' else condition)
    pair_id = (f'{h["history_id"]}:{condition}:h{historical_order}:c{current_order}'
               f':edit_{edited_variable}:query_{query}')
    row = {**copy.deepcopy(h), 'schema': VERSION, 'experiment_kind': VERSION,
           'cross_model_protocol': VERSION, 'design_revision': DESIGN_REVISION,
           'cross_model_stage': h['stage'], 'condition': condition,
           'prompt_family': 'natural_entity_attribute_relational_v2', 'prompt_variant': 'nora_relational_v2',
           'historical_entity_order': historical_order, 'current_entity_order': current_order,
           'mention_order': historical_order if condition.endswith('unassigned') else None,
           'unassigned_slot_order': ('xz' if historical_order == 0 else 'zx') if condition.endswith('unassigned') else None,
           'attribute_relation': {'superseded': 'queried_attribute', 'live': 'queried_attribute',
                                  'early_unassigned': 'unassigned', 'late_unassigned': 'unassigned',
                                  'entity_mention': 'no_assignment', 'other_attribute': 'different_attribute'}[condition],
           'other_attribute': h['other_attribute'], 'control_type': condition,
           'query': query, 'query_id': f'current_{query}',
           'query_entity': h['variables'][('x', 'z').index(query)],
           'edited_variable': edited_variable, 'edited_field': field,
           'edited_entity': h['variables'][('x', 'z').index(edited_variable)],
           'edit_status': status, 'source_value': source, 'replacement_value': replacement,
           'pair_id': pair_id, 'pair_direction': direction, 'example_id': f'{pair_id}:{direction}',
           'candidate_values': list(VALUES), 'semantic_values': values,
           'semantic_status': {f: status if f.startswith('initial') else 'accepted_current' for f in values},
           'current_fields': {v: f'{"initial" if live else "proposed"}_{v}' for v in ('x', 'z')},
           'obsolete_fields': {v: f'initial_{v}' for v in ('x', 'z')} if condition == 'superseded' else {}}
    row = _derived(row)
    row['current_answer'] = row['answer']
    return row


def generate(stage, prior_paths=()):
    """Return fixed histories and the complete 192-member product per history."""
    ledger = json.loads(EXCLUSION_LEDGER.read_text())
    if stage == 'confirmatory' and ledger.get('confirmation_migration_hold'):
        raise ValueError('confirmation held: failed Gemma attempt histories must be audited and added to this revision exclusion ledger')
    histories = _history_definitions(stage)
    current = {h['concrete_signature'] for h in histories}
    # Stage IDs alone are never the freshness test. Do not retry seeds on overlap.
    for earlier in COUNTS:
        if earlier != stage and current & {h['concrete_signature'] for h in _history_definitions(earlier)}:
            raise ValueError(f'fixed semantic histories overlap with {earlier}')
    rows = [_member(h, *cell) for h in histories for cell in sorted(required_cells())]
    audit_structure(histories, rows)
    disjoint(rows, prior_paths)
    return histories, rows


def audit_structure(histories, rows):
    """Check the full product and exact edit semantics, including the edited axis."""
    if not histories or len(rows) != len(histories) * ROWS_PER_HISTORY:
        raise ValueError('wrong exact history/member count')
    by_history = {}
    signatures = set()
    for h in histories:
        if h['history_id'] in by_history:
            raise ValueError('duplicate history ID')
        assigned = list(h['matching_values'].values())
        replacements = list(h['replacement_values'].values())
        if (set(h['matching_values']) != set(FIELDS) or set(h['replacement_values']) != {'initial_x', 'initial_z'}
                or len(set(assigned + replacements)) != 6 or not set(assigned + replacements) <= set(VALUES)):
            raise ValueError('four assigned identities and two independent replacements must be distinct')
        if h['orientation'] not in (0, 1) or h['variables'] != (h['entities'] if h['orientation'] == 0 else h['entities'][::-1]):
            raise ValueError('inconsistent semantic orientation/entity mapping')
        signature = concrete_signature(h)
        if signature in signatures or h.get('concrete_signature') != signature:
            raise ValueError('duplicate or inconsistent concrete history signature')
        if h['attribute'] == 'tag':
            raise ValueError('other attribute must differ from queried attribute')
        signatures.add(signature)
        by_history[h['history_id']] = h
    seen, cells = set(), {hid: set() for hid in by_history}
    for row in rows:
        if row.get('example_id') in seen:
            raise ValueError('duplicate example ID')
        seen.add(row.get('example_id'))
        hid = row.get('history_id')
        if hid not in by_history:
            raise ValueError('unexpected history ID')
        cell = tuple(row.get(k) for k in ('condition', 'historical_entity_order', 'current_entity_order',
                                         'edited_variable', 'query', 'pair_direction'))
        if any(type(row.get(k)) is not int for k in ('historical_entity_order', 'current_entity_order', 'orientation', 'pair_direction')):
            raise ValueError('order/orientation/member axes must be integer labels')
        if cell not in required_cells() or cell in cells[hid]:
            raise ValueError('duplicate or unexpected condition/order/edit/query/member cell')
        # All required metadata and all nonedited semantics must match construction.
        expected = _member(by_history[hid], *cell)
        if any(row.get(k) != value for k, value in expected.items()):
            raise ValueError(f'{row["example_id"]}: invalid pair construction or derived answer/metadata')
        cells[hid].add(cell)
    if any(c != required_cells() for c in cells.values()):
        raise ValueError('incomplete condition/order/edited-entity/query/member product')
    stage = histories[0]['stage']
    allocation = FRACTIONAL_ALLOCATION[:len(histories)] if stage == 'validation' else ALLOCATION_TABLES[stage]
    observed = [(h['allocation_cell']['entity_pair'], h['allocation_cell']['attribute'], h['orientation'],
                 ALT_RELATIONS.index(h['other_attribute'])) for h in histories]
    if observed != list(allocation):
        raise ValueError('history allocation differs from the declared allocation table')
    from collections import Counter
    pair_counts = Counter(h['allocation_cell']['entity_pair'] for h in histories)
    attribute_counts = Counter(h['attribute'] for h in histories)
    orientation_counts = Counter(h['orientation'] for h in histories)
    if stage == 'confirmatory':
        factorial = Counter((h['allocation_cell']['entity_pair'], h['attribute'], h['orientation']) for h in histories)
        relation_cell = Counter((h['allocation_cell']['entity_pair'], h['attribute'], h['orientation'], h['other_attribute']) for h in histories)
        if set(factorial.values()) != {2} or set(relation_cell.values()) != {1} or set(pair_counts.values()) != {16} or set(attribute_counts.values()) != {24} or set(orientation_counts.values()) != {48}:
            raise ValueError('confirmation allocation must have two histories per factorial cell, one per alternate relation')
    elif stage != 'validation' and (set(pair_counts.values()) != {4} or set(attribute_counts.values()) != {6} or set(orientation_counts.values()) != {12}):
        raise ValueError('fractional allocation margins are unbalanced')
    if stage != 'validation' and set(Counter(h['other_attribute'] for h in histories).values()) != ({48} if len(histories) == 96 else {12}):
        raise ValueError('alternate relation allocation is unbalanced')
    if stage in ('development', 'frozen_gate'):
        relation_attr = Counter((h['other_attribute'], h['attribute']) for h in histories)
        relation_pair = Counter((h['other_attribute'], h['allocation_cell']['entity_pair']) for h in histories)
        if set(relation_attr.values()) != {3} or set(relation_pair.values()) != {2}:
            raise ValueError('fractional alternate-relation allocation is unbalanced over attributes and pairs')
    return True


def validate(histories, rows, stage):
    if histories != _history_definitions(stage):
        raise ValueError('history definitions differ from fixed stage seeds/counts')
    return audit_structure(histories, rows)


def validate_rows(rows, stage):
    return validate(_history_definitions(stage), rows, stage)


def audit_rendered_pairs(rows):
    """Verify every baseline/edit rendering differs only at its intended value."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['pair_id'], {})[row['pair_direction']] = row
    if any(set(pair) != {0, 1} for pair in grouped.values()):
        raise ValueError('render audit found a missing paired member')
    counts = {condition: 0 for condition in CONDITIONS}
    for pair in grouped.values():
        baseline, edited = pair[0], pair[1]
        b_body, b_spans = render_body(baseline)
        e_body, e_spans = render_body(edited)
        field = baseline['edited_field']
        b0, b1 = b_spans[field]
        e0, e1 = e_spans[field]
        if (b_body[:b0] != e_body[:e0] or b_body[b1:] != e_body[e1:]
                or b_body[b0:b1] != baseline['source_value']
                or e_body[e0:e1] != baseline['replacement_value']):
            raise ValueError(f"{baseline['pair_id']}: rendered edit changes text outside its target value")
        counts[baseline['condition']] += 1
    return {'baseline_edit_pairs': len(grouped), 'pairs_by_condition': counts,
            'status': 'passed; target value span is the only semantic change'}


def read_history_rows(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get('rows'), list):
        raise ValueError(f'{path}: expected a dataset JSONL or preview rows bundle')
    return value['rows']


def disjoint(rows, prior_paths):
    current = {concrete_signature(row) for row in rows}
    for path in prior_paths:
        if current & {concrete_signature(row) for row in read_history_rows(path)}:
            raise ValueError(f'concrete semantic history overlap with {path}')
    return sorted(current)


def render_body(row):
    """Return body and exact character spans from the renderer, before chat wrapping."""
    names = dict(zip(('x', 'z'), row['variables']))
    ho = ('x', 'z') if row['historical_entity_order'] == 0 else ('z', 'x')
    co = ('x', 'z') if row['current_entity_order'] == 0 else ('z', 'x')
    condition, attr, values = row['condition'], row['attribute'], row['semantic_values']
    lines, spans = [], {}

    def add(kind, variable, field=None, attribute=None):
        line = RENDERING[kind].format(entity=names[variable], attribute=attribute or attr,
                                      value=values[field] if field else '')
        offset = sum(len(s) + 1 for s in lines)
        if field:
            start = line.rfind(values[field])
            spans[field] = [offset + start, offset + start + len(values[field])]
        if kind != 'unassigned':
            start = line.index(names[variable])
            key = ('queried_entity' if kind == 'query' else
                   f'{"current" if field and field.startswith("proposed") else "historical"}_entity_{variable}')
            spans[key] = [offset + start, offset + start + len(names[variable])]
        lines.append(line)

    def historical():
        for v in ho:
            kind = 'current_assignment' if condition == 'live' else (
                'unassigned' if condition.endswith('unassigned') else
                'entity_mention' if condition == 'entity_mention' else 'historical_assignment')
            add(kind, v, f'initial_{v}', row['other_attribute'] if condition == 'other_attribute' else attr)

    def current():
        for v in co:
            add('current_assignment', v, f'proposed_{v}')

    if condition == 'live':
        # Original v1 positive control: edit the sole live assignment itself.
        # No current block exists; current_order is a duplicated matched nuisance level.
        historical()
    elif condition == 'late_unassigned':
        current()
        historical()
    elif condition in CONDITIONS:
        historical()
        current()
    else:
        raise ValueError('unknown relational condition')
    add('query', row['query'])
    lines.append(RENDERING['instruction'])
    return '\n'.join(lines), spans


def render(row, tokenizer=None, chat=True):
    body, _ = render_body(row)
    return _answer_prefix(body, tokenizer, chat)


def semantic_positions(row, tokenizer):
    from src.cross_model.tokens import encode, token_span
    body, spans = render_body(row)
    prompt = render(row, tokenizer, True)
    if prompt.count(body) != 1:
        raise ValueError('chat template does not preserve a unique exact renderer body')
    shift = prompt.index(body)
    sites = {key: token_span(tokenizer, prompt, a + shift, b + shift) for key, (a, b) in spans.items()}
    sites['final_preanswer'] = [len(encode(tokenizer, prompt)) - 1]
    return prompt, sites


def span_audit(row, tokenizer):
    from src.cross_model.tokens import encode
    body, chars = render_body(row)
    prompt, sites = semantic_positions(row, tokenizer)
    shift, ids = prompt.index(body), encode(tokenizer, prompt)
    return {'example_id': row['example_id'], 'prompt_sha256': digest(prompt), 'prompt_token_length': len(ids),
            'spans': {k: {'text': prompt[a+shift:b+shift], 'char_start': a+shift, 'char_end': b+shift,
                          'token_start': sites[k][0], 'token_end_exclusive': sites[k][-1]+1,
                          'token_length': len(sites[k]), 'token_ids': [ids[i] for i in sites[k]]}
                      for k, (a, b) in chars.items()}}
