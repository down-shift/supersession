"""Separate, small follow-up designs; v2 data and estimands remain untouched."""
from __future__ import annotations

import random
from collections import defaultdict
from itertools import product

import numpy as np

from src.cross_model.protocol import VALUES, digest

VERSION = 'relational_followups_v1'
MARKER_CELLS = tuple(product(('superseded', 'entity_mention'), (0, 1)))
HARD_CONDITIONS = ('superseded', 'entity_mention', 'early_unassigned', 'late_unassigned')
DIFFICULTIES = (2, 4, 6)  # distractor entities; selected on development only
BOOTSTRAP = {'unit': 'history', 'draws': 2000, 'seed': 73021}
NAMES = ('Nora', 'Liam', 'Ava', 'Omar', 'Mila', 'Eli', 'Iris', 'Noah', 'Zoe', 'Theo', 'Maya', 'Leo')
DISTRACTOR_VALUES = ('ochre', 'cobalt', 'crimson', 'indigo', 'magenta', 'turquoise', 'lavender',
                     'bronze', 'maroon', 'mustard', 'silver', 'navy', 'plum', 'chartreuse',
                     'saffron', 'burgundy')
STAGE_OFFSETS = {'pilot': 2_000_003, 'development': 0, 'test': 1_000_003, 'confirmatory': 3_000_007}


def _assignment(entity, attribute, value, marker):
    prefix = 'Previously, ' if marker else ''
    return f"{prefix}{entity}'s {attribute} was {value}."


def _mention(entity, value, marker):
    prefix = 'Previously, ' if marker else ''
    return f'{prefix}{entity} mentioned {value} in an unrelated note.'


def marker_example_rows():
    """Minimal grammaticality/marker-only audit examples, not scored data."""
    return {
        ('superseded', 0): "Nora's badge was amber.",
        ('superseded', 1): "Previously, Nora's badge was amber.",
        ('entity_mention', 0): 'Nora mentioned amber in an unrelated note.',
        ('entity_mention', 1): 'Previously, Nora mentioned amber in an unrelated note.',
    }


def validate_marker_templates():
    rows = marker_example_rows()
    assert set(rows) == set(MARKER_CELLS)
    for construction in ('superseded', 'entity_mention'):
        bare, marked = rows[(construction, 0)], rows[(construction, 1)]
        if marked != 'Previously, ' + bare:
            raise ValueError('marker addition changed content beyond its prefix')
    if any(not s.endswith('.') or '  ' in s for s in rows.values()):
        raise ValueError('malformed template example')
    return {'status': 'passed', 'examples': [
                {'construction': c, 'marker': m, 'text': rows[(c, m)]}
                for c, m in MARKER_CELLS],
            'marker_change': 'exactly the prefix "Previously, " in both constructions'}


def physical_history_signature(entities, attribute, current_values, historical_values,
                               distractor_assignments=None):
    """Donor-independent signature of target history and distractor bindings."""
    bindings = sorted((entity, current_values[entity], historical_values[entity]) for entity in entities[:2])
    return digest({'attribute': attribute, 'target_bindings': bindings,
                   'distractor_bindings': distractor_assignments or {}})


def _history(stage, index, seed, n_distractors, used_signatures):
    stage_seed = seed + STAGE_OFFSETS[stage]
    entities = list(NAMES[:2 + n_distractors])
    for attempt in range(10000):
        rng = random.Random(stage_seed + index * 7919 + attempt * 104729)
        values = rng.sample(VALUES, 6)
        current = dict(zip(entities[:2], values[:2]))
        historical = dict(zip(entities[:2], values[2:4]))
        distractor_values = rng.sample(DISTRACTOR_VALUES, 2 * n_distractors)
        distractor_assignments = {}
        for i, entity in enumerate(entities[2:]):
            distractor_assignments[entity] = {
                'historical': distractor_values[2*i], 'current': distractor_values[2*i+1]}
        target_signature = physical_history_signature(entities, 'badge', current, historical)
        signature = physical_history_signature(entities, 'badge', current, historical, distractor_assignments)
        if signature in used_signatures or target_signature in used_signatures:
            continue
        used_signatures.add(signature)
        used_signatures.add(target_signature)
        return {'history_id': f'{VERSION}:{stage}:{index:04d}', 'history_index': index,
            'seed': stage_seed, 'stage': stage, 'entities': entities, 'attribute': 'badge',
            'current_values': current, 'historical_values': historical,
            'replacement_values': dict(zip(entities[:2], values[4:6])),
            'distractor_assignments': distractor_assignments, 'n_distractors': n_distractors,
            'history_signature': signature, 'target_history_signature': target_signature,
            'history_attempt': attempt}
    raise ValueError(f'could not draw a fresh physical history for {stage}:{index}')


def generate_marker(stage='pilot', n_histories=2, seed=20261004, excluded_signatures=None):
    if stage not in ('pilot', 'confirmatory') or n_histories < 1:
        raise ValueError('invalid marker stage/count')
    if stage == 'confirmatory' and excluded_signatures is None:
        raise ValueError('confirmatory marker data require explicit pilot-stage history exclusions')
    rows, used = [], set(excluded_signatures or ())
    for i in range(n_histories):
        h = _history(stage, i, seed, 0, used)
        for historical_order, current_order, marker, construction, edited_variable, query_variable, edited in product(
                (0, 1), (0, 1), (0, 1), ('superseded', 'entity_mention'), ('x', 'z'), ('x', 'z'), (0, 1)):
            old_order = h['entities'] if historical_order == 0 else h['entities'][::-1]
            now_order = h['entities'] if current_order == 0 else h['entities'][::-1]
            histories = []
            for entity in old_order:
                value = h['historical_values'][entity]
                histories.append(_assignment(entity, h['attribute'], value, 0) if construction == 'superseded'
                                 else _mention(entity, value, 0))
            if marker:
                histories = ['Previously, ' + line for line in histories]
            current_lines = [f"Currently, {entity}'s {h['attribute']} is {h['current_values'][entity]}." for entity in now_order]
            query_entity = h['entities'][('x', 'z').index(query_variable)]
            body = '\n'.join([*histories, *current_lines,
                f"What is {query_entity}'s current {h['attribute']}?",
                'Respond with only the value, with no explanation.'])
            edited_entity = h['entities'][('x', 'z').index(edited_variable)]
            source, replacement = h['historical_values'][edited_entity], h['replacement_values'][edited_entity]
            rendered = body.replace(source, replacement, 1) if edited else body
            rows.append({**h, 'condition': construction, 'marker': marker,
                'historical_order': historical_order, 'current_order': current_order,
                'edited_variable': edited_variable, 'query_variable': query_variable,
                'edited': edited, 'source_value': source, 'replacement_value': replacement,
                'query_entity': query_entity, 'answer': h['current_values'][query_entity],
                'stale_value': h['historical_values'][query_entity] if construction == 'superseded' else None,
                'candidate_values': list(VALUES), 'prompt': rendered,
                'example_id': f"{h['history_id']}:{construction}:m{marker}:h{historical_order}:c{current_order}:v{edited_variable}:q{query_variable}:e{edited}"})
    validate_marker_dataset(rows)
    return rows


def marker_required_cells():
    return set(product(('superseded', 'entity_mention'), (0, 1), (0, 1), (0, 1),
                       ('x', 'z'), ('x', 'z'), (0, 1)))


def validate_marker_dataset(rows):
    if not rows:
        raise ValueError('empty marker dataset')
    stages = {r.get('stage') for r in rows}
    if len(stages) != 1 or not stages <= {'pilot', 'confirmatory'}:
        raise ValueError('marker dataset has mixed or incompatible stage labels')
    histories = {}
    cells = defaultdict(set)
    ids, signatures, target_signatures = set(), set(), set()
    pairs = defaultdict(dict)
    for row in rows:
        hid = row['history_id']
        signature = row['history_signature']
        target_signature = row['target_history_signature']
        recomputed_target = physical_history_signature(row['entities'], row['attribute'],
            row['current_values'], row['historical_values'])
        recomputed_full = physical_history_signature(row['entities'], row['attribute'],
            row['current_values'], row['historical_values'], row['distractor_assignments'])
        if target_signature != recomputed_target or signature != recomputed_full:
            raise ValueError('marker physical history signature does not match its assignments')
        if row.get('candidate_values') != VALUES:
            raise ValueError('marker candidate set differs from the frozen v2 vocabulary')
        if signature in signatures and hid not in histories:
            raise ValueError('duplicate physical marker history')
        if hid not in histories:
            if signature in signatures:
                raise ValueError('duplicate donor-independent marker history signature')
            histories[hid] = signature
            signatures.add(signature)
            if target_signature in target_signatures:
                raise ValueError('duplicate donor-independent target history signature')
            target_signatures.add(target_signature)
        elif histories[hid] != signature:
            raise ValueError('marker history signature changes within a history')
        cell = (row['condition'], row['marker'], row['historical_order'], row['current_order'],
                row['edited_variable'], row['query_variable'], row['edited'])
        if cell not in marker_required_cells() or cell in cells[hid]:
            raise ValueError('duplicate or unexpected marker factorial cell')
        if row['example_id'] in ids:
            raise ValueError('duplicate marker example ID')
        ids.add(row['example_id']); cells[hid].add(cell)
        pair = (hid,) + cell[:-1]
        pairs[pair][row['edited']] = row
    expected_per_history = marker_required_cells()
    if any(c != expected_per_history for c in cells.values()):
        raise ValueError('incomplete marker factorial; every history needs the exact full product')
    if len(rows) != len(histories) * len(expected_per_history):
        raise ValueError('marker history/member count mismatch')
    for members in pairs.values():
        if set(members) != {0, 1}:
            raise ValueError('marker pair missing a baseline or edited member')
        base, edit = members[0], members[1]
        if base['answer'] != edit['answer'] or base['current_values'] != edit['current_values']:
            raise ValueError('marker edit changed the correct current answer')
        source, donor = base['source_value'], base['replacement_value']
        if source == donor or edit['source_value'] != source or edit['replacement_value'] != donor:
            raise ValueError('marker source/donor metadata mismatch')
        b, e = base['prompt'], edit['prompt']
        if b.count(source) != 1 or e.count(donor) != 1:
            raise ValueError('marker edit value must occur exactly once in its paired prompt')
        bi, ei = b.index(source), e.index(donor)
        if b[:bi] != e[:ei] or b[bi+len(source):] != e[ei+len(donor):]:
            raise ValueError('marker paired histories differ beyond the edited value')
    # Check marker manipulation line by line: both historical sentences change, nothing else does.
    by = {(r['history_id'], r['condition'], r['historical_order'], r['current_order'],
           r['edited_variable'], r['query_variable'], r['edited'], r['marker']): r for r in rows}
    for h in {r['history_id'] for r in rows}:
        for condition, ho, co, ev, qv, edited in product(('superseded', 'entity_mention'), (0, 1), (0, 1), ('x', 'z'), ('x', 'z'), (0, 1)):
            a = by[(h, condition, ho, co, ev, qv, edited, 0)]['prompt']
            b = by[(h, condition, ho, co, ev, qv, edited, 1)]['prompt']
            al, al2, *ar = a.split('\n'); bl, bl2, *br = b.split('\n')
            if bl != 'Previously, ' + al or bl2 != 'Previously, ' + al2 or ar != br:
                raise ValueError('marker altered content beyond historical sentence')
    return {'n_histories': len(histories), 'n_members': len(rows),
            'history_signatures': sorted(signatures),
            'target_history_signatures': sorted(target_signatures), 'status': 'passed'}


def generate_harder(stage, n_histories=4, n_distractors=2, seed=20261005, excluded_signatures=None):
    if stage not in ('pilot', 'development', 'test') or n_distractors not in DIFFICULTIES:
        raise ValueError('harder stage or prespecified difficulty invalid')
    if stage in ('development', 'test') and excluded_signatures is None:
        raise ValueError(f'{stage} harder data require explicit earlier-stage history exclusions')
    rows, used = [], set(excluded_signatures or ())
    for i in range(n_histories):
        h = _history(stage, i, seed, n_distractors, used)
        for condition, historical_order, current_order, edited_variable, query_variable, edited in product(
                HARD_CONDITIONS, (0, 1), (0, 1), ('x', 'z'), (0, 1), (0, 1)):
            entities = h['entities']
            query_variable = ('x', 'z')[query_variable]
            target = entities[('x', 'z').index(query_variable)]
            old_order = entities[:2] if historical_order == 0 else entities[1::-1]
            now_order = entities[:2] if current_order == 0 else entities[1::-1]
            history_lines = []
            for e in old_order:
                old_value = h['historical_values'][e]
                if condition == 'superseded':
                    history_lines.append(_assignment(e, h['attribute'], old_value, 1))
                elif condition == 'entity_mention':
                    history_lines.append(_mention(e, old_value, 0))
                else:
                    history_lines.append(f'The unassigned {h["attribute"]} value was {old_value}.')
            current_lines = [f"Currently, {e}'s {h['attribute']} is {h['current_values'][e]}." for e in now_order]
            lines = (current_lines + history_lines if condition == 'late_unassigned' else
                     history_lines + current_lines)
            for distractor in entities[2:]:
                values = h['distractor_assignments'][distractor]
                lines.append(_assignment(distractor, h['attribute'], values['historical'], 1))
                lines.append(f"Currently, {distractor}'s {h['attribute']} is {values['current']}.")
            prompt = '\n'.join(lines + [f"What is {target}'s current {h['attribute']}?", 'Respond with only the value, with no explanation.'])
            edited_entity = entities[('x', 'z').index(edited_variable)]
            hist_value = h['historical_values'][edited_entity]
            replacement = h['replacement_values'][edited_entity]
            if edited:
                prompt = prompt.replace(hist_value, replacement, 1)
            rows.append({**h, 'condition': condition,
                'historical_order': historical_order, 'current_order': current_order,
                'edited_variable': edited_variable, 'query_entity': target,
                'edited': edited, 'source_value': hist_value,
                'replacement_value': replacement, 'answer': h['current_values'][target],
                'original_stale_value': h['historical_values'][target],
                'stale_value': replacement if edited and edited_entity == target else h['historical_values'][target],
                'source_value_for_query': h['historical_values'][target],
                'replacement_value_for_query': h['replacement_values'][target],
                'candidate_values': list(VALUES), 'prompt': prompt,
                'query_variable': query_variable,
                'pair_id': f'{h["history_id"]}:{condition}:h{historical_order}:c{current_order}:v{edited_variable}:q{query_variable}',
                'context_id': f'{h["history_id"]}:{condition}:h{historical_order}:c{current_order}:v{edited_variable}:e{edited}',
                'example_id': f'{h["history_id"]}:{condition}:h{historical_order}:c{current_order}:v{edited_variable}:q{query_variable}:e{edited}'})
    validate_harder_dataset(rows)
    return rows


def harder_required_cells():
    return set(product(HARD_CONDITIONS, (0, 1), (0, 1), ('x', 'z'), ('x', 'z'), (0, 1)))


def validate_harder_dataset(rows):
    if not rows:
        raise ValueError('empty harder dataset')
    stages = {r.get('stage') for r in rows}
    if len(stages) != 1 or not stages <= {'pilot', 'development', 'test'}:
        raise ValueError('harder dataset has mixed or incompatible stage labels')
    cells, signatures, ids, pairs = defaultdict(set), {}, set(), defaultdict(dict)
    all_signatures, all_target_signatures = set(), set()
    for row in rows:
        hid = row['history_id']
        recomputed_target = physical_history_signature(row['entities'], row['attribute'],
            row['current_values'], row['historical_values'])
        recomputed_full = physical_history_signature(row['entities'], row['attribute'],
            row['current_values'], row['historical_values'], row['distractor_assignments'])
        if (row['target_history_signature'] != recomputed_target
                or row['history_signature'] != recomputed_full):
            raise ValueError('harder physical history signature does not match its assignments')
        if row.get('candidate_values') != VALUES:
            raise ValueError('harder candidate set differs from the frozen v2 vocabulary')
        if row['answer'] != row['current_values'][row['query_entity']]:
            raise ValueError('harder answer does not match the queried current assignment')
        if hid not in signatures:
            if row['history_signature'] in signatures.values():
                raise ValueError('duplicate donor-independent harder history')
            signatures[hid] = row['history_signature']
            if row['target_history_signature'] in all_target_signatures:
                raise ValueError('duplicate donor-independent harder target history')
            all_target_signatures.add(row['target_history_signature'])
            if row['history_signature'] in all_signatures:
                raise ValueError('duplicate physical harder history')
            all_signatures.add(row['history_signature'])
        elif signatures[hid] != row['history_signature']:
            raise ValueError('history signature changes within a history')
        cell = (row['condition'], row['historical_order'], row['current_order'],
                row['edited_variable'], row['query_variable'], row['edited'])
        if cell not in harder_required_cells() or cell in cells[hid]:
            raise ValueError('duplicate or unexpected harder factorial cell')
        cells[hid].add(cell)
        if row['example_id'] in ids:
            raise ValueError('duplicate harder example ID')
        ids.add(row['example_id'])
        if row['answer'] == row['stale_value']:
            raise ValueError('current answer and stale value must differ')
        pair_key = row['pair_id']
        if row['edited'] in pairs[pair_key]:
            raise ValueError('duplicate harder paired direction')
        pairs[pair_key][row['edited']] = row
    expected = harder_required_cells()
    if any(values != expected for values in cells.values()) or len(rows) != len(cells)*len(expected):
        raise ValueError('incomplete harder factorial; exact condition/order/edit/query/direction product required')
    for pair in pairs.values():
        if set(pair) != {0, 1}:
            raise ValueError('harder pair requires baseline and edited members')
        base, edit = pair[0], pair[1]
        if base['answer'] != edit['answer'] or base['current_values'] != edit['current_values']:
            raise ValueError('historical edit changed target current assignment or answer')
        source, donor = base['source_value'], base['replacement_value']
        b, e = base['prompt'], edit['prompt']
        if b.count(source) != 1 or e.count(donor) != 1:
            raise ValueError('harder source/donor must appear once in paired histories')
        bi, ei = b.index(source), e.index(donor)
        if b[:bi] != e[:ei] or b[bi+len(source):] != e[ei+len(donor):]:
            raise ValueError('harder edit changed prompt content beyond the historical value')
    # Query is the only prompt difference for paired queries within one edited context.
    contexts = defaultdict(dict)
    for row in rows:
        key = row['context_id']
        contexts[key][row['query_entity']] = row
    for query_rows in contexts.values():
        if len(query_rows) != 2:
            raise ValueError('each fixed harder context must query both target entities')
        prompts = [r['prompt'].split('\n') for r in query_rows.values()]
        if len(prompts[0]) != len(prompts[1]) or [x for x in prompts[0] if not x.startswith('What is ')] != [x for x in prompts[1] if not x.startswith('What is ')]:
            raise ValueError('paired harder queries changed history context')
    return {'n_histories': len(cells), 'n_members': len(rows),
            'history_signatures': sorted(set(signatures.values())),
            'target_history_signatures': sorted(all_target_signatures), 'status': 'passed'}


def choose_difficulty(development_summaries):
    """Prespecified: hardest level with complete-answer accuracy in [0.65, .90]."""
    selected = [d for d in DIFFICULTIES if .65 <= development_summaries[str(d)]['accuracy'] <= .90]
    return max(selected) if selected else min(DIFFICULTIES,
             key=lambda d: abs(development_summaries[str(d)]['accuracy'] - .775))


def complete_answer_outcome(text, current_value, stale_value, candidates=VALUES):
    """Deterministic parse: trim whitespace and surrounding punctuation, casefold exact match."""
    parsed = text.strip().strip('"\'`.,;:!? ').casefold()
    values = {v.casefold(): v for v in candidates}
    answer = values.get(parsed)
    if answer == current_value:
        category = 'correct'
    elif stale_value is not None and answer == stale_value:
        category = 'stale'
    else:
        category = 'other'
    return {'complete_answer': text, 'parsed_answer': answer, 'category': category,
            'parse_rule': 'trim whitespace and edge punctuation; exact case-insensitive candidate match'}


def _bootstrap(values):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('history bootstrap requires finite complete histories')
    rng = np.random.default_rng(BOOTSTRAP['seed'])
    indices = rng.integers(0, len(x), size=(BOOTSTRAP['draws'], len(x)))
    draws = x[indices].mean(axis=1)
    return {'n_histories': len(x), 'mean': float(x.mean()),
            'ci95_history_bootstrap': [float(v) for v in np.quantile(draws, [.025, .975])]}


def analyze_marker(rows, scores):
    """Compute v2-style E/R plus marker factorial contrasts from saved masses."""
    validate_marker_dataset(rows)
    expected = {r['example_id']: r for r in rows}
    actual = {r['example_id']: r for r in scores}
    if len(expected) != len(rows) or len(actual) != len(scores) or expected.keys() != actual.keys():
        raise ValueError('marker dataset/score IDs are duplicated or incomplete')
    effects = defaultdict(dict)
    for eid, row in expected.items():
        score = actual[eid]
        if any(score.get(k) != v for k, v in row.items()):
            raise ValueError(f'marker score metadata mismatch at {eid}')
        masses = score.get('semantic_log_mass')
        if (not isinstance(masses, dict) or set(masses) != set(VALUES)
                or not np.isfinite(list(masses.values())).all()):
            raise ValueError('complete v2 semantic candidate masses required')
        cell = (row['history_id'], row['condition'], row['marker'], row['historical_order'], row['current_order'],
                row['edited_variable'], row['query_variable'])
        direction = row['edited']
        if direction in effects[cell]:
            raise ValueError('duplicate marker edit direction')
        effects[cell][direction] = masses
    e_rows = {}
    for cell, directions in effects.items():
        if set(directions) != {0, 1}:
            raise ValueError('incomplete marker baseline/edit pair')
        hid, cond, marker, ho, co, variable, query = cell
        source = expected[next(r['example_id'] for r in rows if
            (r['history_id'], r['condition'], r['marker'], r['historical_order'], r['current_order'],
             r['edited_variable'], r['query_variable']) == cell and r['edited'] == 0)]['source_value']
        replacement = next(r['replacement_value'] for r in rows if
            (r['history_id'], r['condition'], r['marker'], r['historical_order'], r['current_order'],
             r['edited_variable'], r['query_variable']) == cell)
        before, after = directions[0], directions[1]
        # E = delta(replacement) - delta(source), retaining both components.
        replacement_component = after[replacement] - before[replacement]
        source_component = after[source] - before[source]
        e_rows[cell] = {
            'history_id': hid,
            'E': replacement_component - source_component,
            'E_replacement_component': replacement_component,
            'E_source_component': source_component}
    by_history = defaultdict(dict)
    for (hid, cond, marker, ho, co, variable, query), e in e_rows.items():
        order = 'aligned' if ho == co else 'reversed'
        key = (cond, marker, order, variable, query)
        if key in by_history[hid]:
            prior = by_history[hid][key]
            for metric in ('E', 'E_replacement_component', 'E_source_component'):
                prior[metric].append(e[metric])
        else:
            by_history[hid][key] = {m: [e[m]] for m in ('E', 'E_replacement_component', 'E_source_component')}
    history_rows = []
    for hid, cells in sorted(by_history.items()):
        out = {'history_id': hid}
        for cond, marker, order in product(('superseded', 'entity_mention'), (0, 1), ('all', 'aligned', 'reversed')):
            def cell_value(variable, query, metric):
                selected = [values[metric] for (c, m, o, v, q), values in cells.items()
                            if c == cond and m == marker and v == variable and q == query and (order == 'all' or o == order)]
                if not selected:
                    raise ValueError(f'missing marker query/edit cell: {cond}/{marker}/{order}/{variable}/{query}')
                return float(np.mean([np.mean(v) for v in selected]))
            for metric in ('E', 'E_replacement_component', 'E_source_component'):
                r_x = cell_value('x', 'x', metric) - cell_value('x', 'z', metric)
                r_z = cell_value('z', 'z', metric) - cell_value('z', 'x', metric)
                out[f'{cond}_m{marker}_{order}_{metric}'] = .5 * (r_x + r_z)
        for order in ('all', 'aligned', 'reversed'):
            for cond in ('superseded', 'entity_mention'):
                out[f'{cond}_marker_effect_{order}'] = out[f'{cond}_m1_{order}_E'] - out[f'{cond}_m0_{order}_E']
            out[f'marker_by_construction_interaction_{order}'] = (
                out[f'superseded_marker_effect_{order}'] - out[f'entity_mention_marker_effect_{order}'])
        history_rows.append(out)
    metric_names = [k for k in history_rows[0] if k != 'history_id']
    edit_effect_rows = [{'history_id': hid, 'condition': cond, 'marker': marker,
        'historical_order': ho, 'current_order': co, 'edited_variable': variable,
        'query_variable': query, **{metric: float(row[metric]) for metric in
        ('E', 'E_replacement_component', 'E_source_component')}}
        for (hid, cond, marker, ho, co, variable, query), row in sorted(e_rows.items())]
    return {'edit_effect_rows': edit_effect_rows, 'history_rows': history_rows,
            'summary': {k: _bootstrap([r[k] for r in history_rows]) for k in metric_names},
            'bootstrap': BOOTSTRAP,
            'estimand': 'v2 bounded surface-class continuation mass; history-level E then symmetric query-specific R'}


def analyze_harder(rows, scores):
    """All-trial answer transitions, stale rates, candidate margins, and secondary v2-style R."""
    validate_harder_dataset(rows)
    expected = {r['example_id']: r for r in rows}
    actual = {r['example_id']: r for r in scores}
    if len(expected) != len(rows) or actual.keys() != expected.keys():
        raise ValueError('harder dataset/score IDs are duplicated or incomplete')
    paired = defaultdict(dict)
    for eid, row in expected.items():
        score = actual[eid]
        if any(score.get(k) != v for k, v in row.items()):
            raise ValueError(f'harder score metadata mismatch at {eid}')
        if score.get('answer_category') not in ('correct', 'stale', 'other'):
            raise ValueError('missing parsed complete-answer outcome')
        masses = score.get('semantic_log_mass', {})
        if set(masses) != set(VALUES) or not np.isfinite(list(masses.values())).all():
            raise ValueError('complete candidate masses required')
        if 'generated_answer' not in score:
            raise ValueError('complete generated answer required for harder analysis')
        outcome = complete_answer_outcome(score['generated_answer'], row['answer'], row['stale_value'])
        if score.get('parsed_answer') != outcome['parsed_answer'] or score['answer_category'] != outcome['category']:
            raise ValueError('harder answer parse/category disagrees with the frozen parser')
        if score.get('source_response') is not (outcome['parsed_answer'] == row['source_value']):
            raise ValueError('source response indicator disagrees with generated answer')
        if score.get('donor_response') is not (outcome['parsed_answer'] == row['replacement_value']):
            raise ValueError('donor response indicator disagrees with generated answer')
        margin = score.get('current_minus_historical_log_mass')
        if margin is None or not np.isfinite(margin):
            raise ValueError('current-versus-historical candidate margin required')
        expected_margin = masses[row['answer']] - masses[row['stale_value']]
        if not np.isclose(margin, expected_margin, rtol=1e-7, atol=1e-7):
            raise ValueError('current-versus-historical margin disagrees with saved candidate masses')
        key = (row['history_id'], row['condition'], row['historical_order'], row['current_order'],
               row['edited_variable'], row['query_entity'])
        if row['edited'] in paired[key]:
            raise ValueError('duplicate harder edit direction')
        paired[key][row['edited']] = score
    effects, transitions = [], defaultdict(int)
    for key, directions in paired.items():
        if set(directions) != {0, 1}:
            raise ValueError('incomplete harder baseline/edit pair')
        hid, cond, ho, co, variable, query_entity = key
        b, e = directions[0], directions[1]
        transitions[(cond, 'aligned' if ho == co else 'reversed',
                     b['answer_category'], e['answer_category'])] += 1
        src, rep = b['source_value'], b['replacement_value']
        e_rep = e['semantic_log_mass'][rep] - b['semantic_log_mass'][rep]
        e_src = e['semantic_log_mass'][src] - b['semantic_log_mass'][src]
        effects.append({'history_id': hid, 'condition': cond, 'historical_order': ho,
            'current_order': co, 'edited_variable': variable,
            'query': directions[0].get('query_variable') or next(
                r['query_variable'] for r in rows if r['history_id'] == hid and r['condition'] == cond
                and r['historical_order'] == ho and r['current_order'] == co
                and r['edited_variable'] == variable and r['query_entity'] == query_entity),
            'E': e_rep - e_src, 'E_replacement_component': e_rep, 'E_source_component': e_src,
            'baseline_category': b['answer_category'], 'edited_category': e['answer_category'],
            'baseline_stale': int(b['answer_category'] == 'stale'),
            'edited_stale': int(e['answer_category'] == 'stale'),
            'baseline_source_response': int(b.get('source_response', False)),
            'edited_source_response': int(e.get('source_response', False)),
            'baseline_donor_response': int(b.get('donor_response', False)),
            'edited_donor_response': int(e.get('donor_response', False)),
            'candidate_margin_baseline': b['current_minus_historical_log_mass'],
            'candidate_margin_edited': e['current_minus_historical_log_mass']})
    grouped = defaultdict(list)
    for row in effects:
        grouped[(row['history_id'], row['condition'])].append(row)
    history_rows = []
    for (hid, condition), cells in sorted(grouped.items()):
        out = {'history_id': hid, 'condition': condition}
        out['stale_edit_change'] = float(np.mean([r['edited_stale'] - r['baseline_stale'] for r in cells]))
        out['source_response_edit_change'] = float(np.mean([r['edited_source_response'] - r['baseline_source_response'] for r in cells]))
        out['donor_response_edit_change'] = float(np.mean([r['edited_donor_response'] - r['baseline_donor_response'] for r in cells]))
        out['candidate_margin_edit_change'] = float(np.mean([
            r['candidate_margin_edited'] - r['candidate_margin_baseline'] for r in cells
            if r['candidate_margin_baseline'] is not None and r['candidate_margin_edited'] is not None]))
        margin_pairs = [r for r in cells if r['candidate_margin_baseline'] is not None
                        and r['candidate_margin_edited'] is not None]
        out['candidate_margin_baseline'] = float(np.mean([r['candidate_margin_baseline'] for r in margin_pairs]))
        out['candidate_margin_edited'] = float(np.mean([r['candidate_margin_edited'] for r in margin_pairs]))
        for order_name, predicate in (('all', lambda r: True), ('aligned', lambda r: r['historical_order'] == r['current_order']),
                                      ('reversed', lambda r: r['historical_order'] != r['current_order'])):
            chosen = [r for r in cells if predicate(r)]
            def mean_e(variable, query):
                vals = [r['E'] for r in chosen if r['edited_variable'] == variable and r['query'] == query]
                if not vals:
                    raise ValueError(f'missing harder R cell: {condition}/{order_name}/{variable}/{query}')
                return float(np.mean(vals))
            out[f'R_{order_name}'] = .5 * ((mean_e('x', 'x')-mean_e('x', 'z')) +
                                            (mean_e('z', 'z')-mean_e('z', 'x')))
        history_rows.append(out)
    by_condition = defaultdict(list)
    for row in history_rows:
        by_condition[row['condition']].append(row)
    summaries = {}
    for condition, group in by_condition.items():
        for metric in ('stale_edit_change', 'source_response_edit_change', 'donor_response_edit_change',
                       'candidate_margin_baseline', 'candidate_margin_edited', 'candidate_margin_edit_change',
                       'R_all', 'R_aligned', 'R_reversed'):
            summaries[f'{condition}_{metric}'] = _bootstrap([r[metric] for r in group])
    stale_contrasts = {}
    for control in ('entity_mention', 'early_unassigned', 'late_unassigned'):
        by_id = {c: {r['history_id']: r['stale_edit_change'] for r in g} for c, g in by_condition.items()}
        common = sorted(set(by_id.get('superseded', {})) & set(by_id.get(control, {})))
        if common:
            stale_contrasts[f'superseded_minus_{control}_stale_edit_change'] = _bootstrap([
                by_id['superseded'][h] - by_id[control][h] for h in common])
    overall_accuracy = float(np.mean([actual[eid]['answer_category'] == 'correct' for eid in expected]))
    overall_stale = float(np.mean([actual[eid]['answer_category'] == 'stale' for eid in expected]))
    outcome_summaries = {}
    for condition in HARD_CONDITIONS:
        for edited in (0, 1):
            group = [actual[r['example_id']] for r in rows
                     if r['condition'] == condition and r['edited'] == edited]
            outcome_summaries[f'{condition}_edited_{edited}'] = {
                'n_members': len(group),
                'complete_answer_accuracy': float(np.mean([s['answer_category'] == 'correct' for s in group])),
                'stale_answer_frequency': float(np.mean([s['answer_category'] == 'stale' for s in group])),
                'other_answer_frequency': float(np.mean([s['answer_category'] == 'other' for s in group]))}
    return {'overall_complete_answer_accuracy': overall_accuracy,
        'overall_stale_answer_frequency': overall_stale,
        'outcome_summaries_by_condition_and_edit': outcome_summaries,
        'condition_order_transition_counts': {f'{c}:{o}:{b}->{e}': n for (c,o,b,e),n in sorted(transitions.items())},
        'history_rows': history_rows, 'paired_edit_effect_rows': effects, 'summaries': summaries,
        'stale_edit_change_contrasts': stale_contrasts, 'bootstrap': BOOTSTRAP,
        'all_test_histories_policy': 'all generated scored rows included; no error-only selection',
        'source_donor_response_edit_changes': {
            c: {'source_mean_change': float(np.mean([r['source_response_edit_change'] for r in g])),
                'donor_mean_change': float(np.mean([r['donor_response_edit_change'] for r in g]))}
            for c, g in by_condition.items()}}
