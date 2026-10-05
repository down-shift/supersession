"""Update-chain experiment (docs/chain_v1.md): can a stale value change generated answers below ceiling?

Each history gives two entities a chain of `depth` distinct values for one attribute, revealed in
rounds: `Initially, …was v.`, then `Then, …changed to v.`, and finally `Currently, …is v.` The edit
replaces the edited entity's immediately superseded value (round depth-1) with a donor that appears
nowhere else in the prompt. Scoring records both greedy generated answers and candidate masses over
an extended vocabulary, so the stale rate, answer transitions and query relevance R are all
estimable. Scope: dataset generation, validation and analysis only; inference is in
scripts/run_followups.py.
"""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import product

import numpy as np

from src.cross_model.followups import (BOOTSTRAP, DISTANCE_ATTRIBUTES, DISTANCE_PAIRS, STAGE_OFFSETS,
                                       _bootstrap, physical_history_signature)
from src.cross_model.protocol import VALUES

CHAIN_VERSION = 'relational_followups_chain_v1'
DEPTHS = (3, 4, 5)
# Extension candidates in recorded preference order; four are chosen by the tokenizer audit
# (scripts/chain_audit.py) before any chain dataset is generated.
EXTENSION_CANDIDATES = ('olive', 'plum', 'navy', 'rust', 'mint', 'sand', 'lilac', 'cream', 'gold', 'khaki')
N_EXTENSION = 4


def chain_vocabulary(extension):
    vocab = list(VALUES) + list(extension)
    if len(extension) != N_EXTENSION or len(set(vocab)) != len(vocab):
        raise ValueError('chain vocabulary must add exactly four new distinct values')
    if any(a != b and a in b for a in vocab for b in vocab):
        raise ValueError('no chain value may be a substring of another')
    return vocab


def chain_lines(entities_hist, entities_now, attribute, chains, depth):
    lines = []
    for round_index in range(depth):
        order = entities_now if round_index == depth - 1 else entities_hist
        for e in order:
            v = chains[e][round_index]
            if round_index == 0:
                lines.append(f"Initially, {e}'s {attribute} was {v}.")
            elif round_index < depth - 1:
                lines.append(f"Then, {e}'s {attribute} changed to {v}.")
            else:
                lines.append(f"Currently, {e}'s {attribute} is {v}.")
    return lines


def generate_chain(stage, n_histories, depth, seed, vocabulary, excluded_signatures=None):
    if stage not in ('development', 'confirmatory') or depth not in DEPTHS:
        raise ValueError('invalid chain stage or depth')
    if 2 * depth + 2 > len(vocabulary):
        raise ValueError('vocabulary too small for disjoint chains plus two donors')
    cells = list(product(range(len(DISTANCE_PAIRS)), range(len(DISTANCE_ATTRIBUTES)), (0, 1)))
    used = set(excluded_signatures or ())
    rows = []
    stage_seed = seed + STAGE_OFFSETS[stage] + 101 * depth
    for index in range(n_histories):
        pair, attribute_index, orientation = cells[index % len(cells)]
        entities = list(DISTANCE_PAIRS[pair])[::(-1 if orientation else 1)]
        attribute = DISTANCE_ATTRIBUTES[attribute_index]
        for attempt in range(10000):
            rng = random.Random(stage_seed + index * 7919 + attempt * 104729)
            drawn = rng.sample(vocabulary, 2 * depth + 2)
            chains = {entities[0]: drawn[:depth], entities[1]: drawn[depth:2 * depth]}
            current = {e: chains[e][-1] for e in entities}
            stale = {e: chains[e][-2] for e in entities}
            signature = physical_history_signature(entities, f'{attribute}|chain{depth}', current,
                                                   {e: '>'.join(chains[e][:-1]) for e in entities})
            if signature not in used:
                used.add(signature)
                break
        else:
            raise ValueError(f'could not draw a fresh chain history {index}')
        h = {'history_id': f'{CHAIN_VERSION}:{stage}:d{depth}:{index:04d}', 'history_index': index,
             'stage': stage, 'seed': stage_seed, 'depth': depth, 'entities': entities, 'attribute': attribute,
             'entity_pair': pair, 'orientation': orientation, 'chains': chains, 'current_values': current,
             'historical_values': stale, 'replacement_values': dict(zip(entities, drawn[2 * depth:])),
             'distractor_assignments': {}, 'n_distractors': 0, 'history_signature': signature,
             'target_history_signature': signature, 'history_attempt': attempt}
        for historical_order, current_order, edited_variable, query_variable, edited in product(
                (0, 1), (0, 1), ('x', 'z'), ('x', 'z'), (0, 1)):
            hist = entities if historical_order == 0 else entities[::-1]
            now = entities if current_order == 0 else entities[::-1]
            query_entity = entities[('x', 'z').index(query_variable)]
            edited_entity = entities[('x', 'z').index(edited_variable)]
            source, donor = stale[edited_entity], h['replacement_values'][edited_entity]
            body = '\n'.join([*chain_lines(hist, now, attribute, chains, depth),
                              f"What is {query_entity}'s current {attribute}?",
                              'Respond with only the value, with no explanation.'])
            prompt = body.replace(f' {source}.', f' {donor}.', 1) if edited else body
            query_stale = donor if edited and edited_entity == query_entity else stale[query_entity]
            rows.append({**h, 'condition': 'superseded_chain', 'historical_order': historical_order,
                'current_order': current_order, 'edited_variable': edited_variable,
                'query_variable': query_variable, 'query_entity': query_entity, 'edited': edited,
                'source_value': source, 'replacement_value': donor, 'answer': current[query_entity],
                'stale_value': query_stale, 'candidate_values': list(vocabulary), 'prompt': prompt,
                'example_id': f"{h['history_id']}:h{historical_order}:c{current_order}:v{edited_variable}:q{query_variable}:e{edited}"})
    validate_chain_dataset(rows)
    return rows


def validate_chain_dataset(rows):
    if not rows or len({r['stage'] for r in rows}) != 1 or len({r['depth'] for r in rows}) != 1:
        raise ValueError('chain dataset must have one stage and one depth')
    cells, ids, pairs, signatures = defaultdict(set), set(), defaultdict(dict), {}
    expected_cells = set(product((0, 1), (0, 1), ('x', 'z'), ('x', 'z'), (0, 1)))
    for r in rows:
        prior = signatures.setdefault(r['history_id'], r['history_signature'])
        if prior != r['history_signature'] or r['example_id'] in ids:
            raise ValueError('chain history signature changes or duplicate example')
        cell = (r['historical_order'], r['current_order'], r['edited_variable'], r['query_variable'], r['edited'])
        if cell in cells[r['history_id']]:
            raise ValueError('duplicate chain cell')
        cells[r['history_id']].add(cell); ids.add(r['example_id'])
        values_in_prompt = [v for v in r['candidate_values'] if f' {v}.' in r['prompt']]
        if len(values_in_prompt) != 2 * r['depth'] or r['answer'] not in values_in_prompt:
            raise ValueError('chain prompt must contain exactly the two disjoint chains')
        pairs[(r['history_id'],) + cell[:-1]][r['edited']] = r
    if len(set(signatures.values())) != len(signatures) or any(c != expected_cells for c in cells.values()):
        raise ValueError('incomplete chain factorial or duplicate physical history')
    for m in pairs.values():
        base, edit = m[0], m[1]
        s, d = base['source_value'], base['replacement_value']
        if base['answer'] != edit['answer'] or base['prompt'].count(f' {s}.') != 1 or f' {d}.' in base['prompt']:
            raise ValueError('chain edit must replace one superseded value with an absent donor')
        if base['prompt'].replace(f' {s}.', f' {d}.', 1) != edit['prompt']:
            raise ValueError('chain paired prompts differ beyond the edited value')
    return {'n_histories': len(signatures), 'n_members': len(rows), 'status': 'passed'}


def analyze_chain(rows, scores):
    """Accuracy, stale rate, query-specific donor following in generated answers, and R from masses."""
    validate_chain_dataset(rows)
    actual = {s['example_id']: s for s in scores}
    if set(actual) != {r['example_id'] for r in rows}:
        raise ValueError('chain scores do not match dataset')
    per_history = defaultdict(lambda: defaultdict(list))
    e_cells = defaultdict(dict)
    for r in rows:
        s = actual[r['example_id']]
        hid, parsed = r['history_id'], s.get('parsed_answer')
        if 'parsed_answer' not in s:  # candidate-only scoring: R from masses, no answer metrics
            masses = s['semantic_log_mass']
            cell = (hid, r['historical_order'], r['current_order'], r['edited_variable'], r['query_variable'])
            e_cells[cell][r['edited']] = (masses, r['source_value'], r['replacement_value'])
            continue
        per_history[hid]['correct'].append(parsed == r['answer'])
        per_history[hid]['stale'].append(parsed == r['stale_value'])
        # Any value shown in the prompt other than the two current values is an earlier value.
        earlier = {v for v in r['candidate_values'] if f' {v}.' in r['prompt']} - set(r['current_values'].values())
        per_history[hid]['any_earlier'].append(parsed in earlier)
        if r['edited']:
            key = 'donor_matched' if r['edited_variable'] == r['query_variable'] else 'donor_other'
            per_history[hid][key].append(parsed == r['replacement_value'])
        else:
            key = 'source_matched' if r['edited_variable'] == r['query_variable'] else 'source_other'
            per_history[hid][key].append(parsed == r['source_value'])
        masses = s['semantic_log_mass']
        cell = (hid, r['historical_order'], r['current_order'], r['edited_variable'], r['query_variable'])
        e_cells[cell][r['edited']] = (masses, r['source_value'], r['replacement_value'])
    r_rows = defaultdict(lambda: defaultdict(list))
    for (hid, ho, co, ev, qv), d in e_cells.items():
        (before, s, rep), (after, _, _) = d[0], d[1]
        e = (after[rep] - before[rep]) - (after[s] - before[s])
        r_rows[hid][(ev, qv)].append(e)
    history = []
    for hid in sorted(r_rows):
        h = {k: float(np.mean(v)) for k, v in per_history[hid].items()}
        cell = {k: float(np.mean(v)) for k, v in r_rows[hid].items()}
        h['R'] = .5 * ((cell[('x', 'x')] - cell[('x', 'z')]) + (cell[('z', 'z')] - cell[('z', 'x')]))
        if 'donor_matched' in h:
            h['donor_following'] = h['donor_matched'] - h['donor_other']
        history.append({'history_id': hid, **h})
    keys = [k for k in history[0] if k != 'history_id']
    return {'history_rows': history, 'bootstrap': BOOTSTRAP,
            'summary': {k: _bootstrap([r[k] for r in history]) for k in keys},
            'n_members': len(rows), 'depth': rows[0]['depth'],
            'estimand': ('generated answers: accuracy, stale rate (queried entity''s immediately superseded value), '
                         'donor following = P(donor | edited, matched query) - P(donor | edited, other query); '
                         'R from bounded candidate masses over the chain vocabulary')}
