"""Updated-other-attribute experiment (docs/update_control_v1.md).

Separates the two properties joined in Experiment 1's superseded-minus-other-attribute contrast:
having been the queried attribute, and being contradicted by a later update. Three constructions
share a six-line structure (two historical lines, two extra current lines, two current lines for the
queried attribute):

- superseded:      Previously <badge> ... ; Currently <filler attribute> ... ; Currently <badge> ...
- other_updated:   Previously <other>  ... ; Currently <other>            ... ; Currently <badge> ...
- other_static:    Previously <other>  ... ; Currently <filler attribute> ... ; Currently <badge> ...

`other` and the filler attribute are team/project, counterbalanced across histories. Every history
uses all eight vocabulary values: two historical, two extra-current, two queried-attribute current,
and two donors. Scope: generation, validation and analysis; inference is scripts/run_followups.py.
"""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import product

import numpy as np

from src.cross_model.followups import (BOOTSTRAP, DISTANCE_ATTRIBUTES, DISTANCE_PAIRS, STAGE_OFFSETS,
                                       _bootstrap, physical_history_signature)
from src.cross_model.protocol import VALUES

UPDATE_VERSION = 'relational_followups_update_v1'
CONDITIONS = ('superseded', 'other_updated', 'other_static')
ORDERS = ('all', 'aligned', 'reversed')


def _lines(condition, hist, now, attribute, other, filler, historical, extra, current):
    hist_attr = attribute if condition == 'superseded' else other
    extra_attr = other if condition == 'other_updated' else filler
    return ([f"Previously, {e}'s {hist_attr} was {historical[e]}." for e in hist]
            + [f"Currently, {e}'s {extra_attr} is {extra[e]}." for e in now]
            + [f"Currently, {e}'s {attribute} is {current[e]}." for e in now])


def generate_update(stage='confirmatory', n_histories=96, seed=20261109, excluded_signatures=None):
    if stage not in ('pilot', 'confirmatory'):
        raise ValueError('invalid update-control stage')
    cells = list(product(range(len(DISTANCE_PAIRS)), range(len(DISTANCE_ATTRIBUTES)), (0, 1)))
    if n_histories % len(cells):
        raise ValueError(f'histories must be a multiple of {len(cells)}')
    used, rows = set(excluded_signatures or ()), []
    stage_seed = seed + STAGE_OFFSETS[stage]
    for index in range(n_histories):
        pair, attribute_index, orientation = cells[index % len(cells)]
        entities = list(DISTANCE_PAIRS[pair])[::(-1 if orientation else 1)]
        attribute = DISTANCE_ATTRIBUTES[attribute_index]
        other, filler = ('team', 'project') if (index // len(cells)) % 2 == 0 else ('project', 'team')
        for attempt in range(10000):
            rng = random.Random(stage_seed + index * 7919 + attempt * 104729)
            v = rng.sample(VALUES, 8)
            historical, extra = dict(zip(entities, v[0:2])), dict(zip(entities, v[2:4]))
            current, donors = dict(zip(entities, v[4:6])), dict(zip(entities, v[6:8]))
            signature = physical_history_signature(entities, f'{attribute}|update|{other}', current,
                                                   {e: f'{historical[e]}>{extra[e]}' for e in entities})
            if signature not in used:
                used.add(signature)
                break
        else:
            raise ValueError(f'could not draw a fresh update-control history {index}')
        h = {'history_id': f'{UPDATE_VERSION}:{stage}:{index:04d}', 'history_index': index, 'stage': stage,
             'seed': stage_seed, 'entities': entities, 'attribute': attribute, 'other_attribute': other,
             'filler_attribute': filler, 'entity_pair': pair, 'orientation': orientation,
             'historical_values': historical, 'extra_current_values': extra, 'current_values': current,
             'replacement_values': donors, 'distractor_assignments': {}, 'n_distractors': 0,
             'history_signature': signature, 'target_history_signature': signature, 'history_attempt': attempt}
        for condition, ho, co, ev, qv, edited in product(CONDITIONS, (0, 1), (0, 1), ('x', 'z'), ('x', 'z'), (0, 1)):
            hist = entities if ho == 0 else entities[::-1]
            now = entities if co == 0 else entities[::-1]
            query_entity = entities[('x', 'z').index(qv)]
            edited_entity = entities[('x', 'z').index(ev)]
            source, donor = historical[edited_entity], donors[edited_entity]
            body = '\n'.join(_lines(condition, hist, now, attribute, other, filler, historical, extra, current)
                             + [f"What is {query_entity}'s current {attribute}?",
                                'Respond with only the value, with no explanation.'])
            prompt = body.replace(f' {source}.', f' {donor}.', 1) if edited else body
            rows.append({**h, 'condition': condition, 'historical_order': ho, 'current_order': co,
                         'edited_variable': ev, 'query_variable': qv, 'query_entity': query_entity,
                         'edited': edited, 'source_value': source, 'replacement_value': donor,
                         'answer': current[query_entity],
                         'stale_value': historical[query_entity] if condition == 'superseded' else None,
                         'candidate_values': list(VALUES), 'prompt': prompt,
                         'example_id': f"{h['history_id']}:{condition}:h{ho}:c{co}:v{ev}:q{qv}:e{edited}"})
    validate_update_dataset(rows)
    return rows


def validate_update_dataset(rows):
    if not rows or len({r['stage'] for r in rows}) != 1:
        raise ValueError('update-control dataset must have one stage')
    expected = set(product(CONDITIONS, (0, 1), (0, 1), ('x', 'z'), ('x', 'z'), (0, 1)))
    cells, ids, signatures, pairs = defaultdict(set), set(), {}, defaultdict(dict)
    for r in rows:
        if signatures.setdefault(r['history_id'], r['history_signature']) != r['history_signature']:
            raise ValueError('signature changes within a history')
        cell = (r['condition'], r['historical_order'], r['current_order'], r['edited_variable'],
                r['query_variable'], r['edited'])
        if cell in cells[r['history_id']] or r['example_id'] in ids:
            raise ValueError('duplicate update-control cell')
        cells[r['history_id']].add(cell); ids.add(r['example_id'])
        lines = r['prompt'].split('\n')
        if len(lines) != 8 or sum(f' {v}.' in r['prompt'] for v in VALUES) != 6:
            raise ValueError('update-control prompts need six statement lines over six distinct values')
        pairs[(r['history_id'],) + cell[:-1]][r['edited']] = r
    if len(set(signatures.values())) != len(signatures) or any(c != expected for c in cells.values()):
        raise ValueError('incomplete update-control factorial or duplicate history')
    for m in pairs.values():
        base, edit = m[0], m[1]
        s, d = base['source_value'], base['replacement_value']
        if base['prompt'].count(f' {s}.') != 1 or base['prompt'].replace(f' {s}.', f' {d}.', 1) != edit['prompt']:
            raise ValueError('update-control edit must change only the historical value')
    return {'n_histories': len(signatures), 'n_members': len(rows), 'status': 'passed'}


def analyze_update(rows, scores):
    """R per construction and order group; prespecified contrasts (docs/update_control_v1.md)."""
    validate_update_dataset(rows)
    actual = {s['example_id']: s for s in scores}
    if set(actual) != {r['example_id'] for r in rows}:
        raise ValueError('update-control scores do not match dataset')
    pairs = defaultdict(dict)
    for r in rows:
        pairs[(r['history_id'], r['condition'], r['historical_order'], r['current_order'],
               r['edited_variable'], r['query_variable'])][r['edited']] = (actual[r['example_id']]['semantic_log_mass'], r)
    cell_e = defaultdict(list)
    for (hid, cond, ho, co, ev, qv), d in pairs.items():
        (before, r), (after, _) = d[0], d[1]
        s, rep = r['source_value'], r['replacement_value']
        e = (after[rep] - before[rep]) - (after[s] - before[s])
        for order in ('all', 'aligned' if ho == co else 'reversed'):
            cell_e[(hid, cond, order, ev, qv)].append(e)
    history = []
    for hid in sorted({k[0] for k in cell_e}):
        out = {'history_id': hid}
        for cond, order in product(CONDITIONS, ORDERS):
            m = {(ev, qv): float(np.mean(cell_e[(hid, cond, order, ev, qv)])) for ev, qv in product('xz', 'xz')}
            out[f'R_{cond}_{order}'] = .5 * ((m[('x', 'x')] - m[('x', 'z')]) + (m[('z', 'z')] - m[('z', 'x')]))
        for order in ORDERS:
            out[f'superseded_minus_other_updated_{order}'] = out[f'R_superseded_{order}'] - out[f'R_other_updated_{order}']
            out[f'other_updated_minus_other_static_{order}'] = out[f'R_other_updated_{order}'] - out[f'R_other_static_{order}']
            out[f'superseded_minus_other_static_{order}'] = out[f'R_superseded_{order}'] - out[f'R_other_static_{order}']
        history.append(out)
    keys = [k for k in history[0] if k != 'history_id']
    return {'history_rows': history, 'bootstrap': BOOTSTRAP,
            'summary': {k: _bootstrap([h[k] for h in history]) for k in keys},
            'estimand': 'v2 bounded surface-class continuation mass; history-level E then symmetric query-specific R'}
