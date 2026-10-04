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


def _history(seed, index, stage, n_distractors=0):
    rng = random.Random(seed + index * 7919)
    names = ['Nora', 'Liam', 'Ava', 'Omar', 'Mila', 'Eli', 'Iris', 'Noah',
             'Zoe', 'Theo', 'Maya', 'Leo']
    entities = names[:2 + n_distractors]
    values = rng.sample(VALUES, max(3, 2 + n_distractors))
    current, historical = values[:2], values[2:]
    return {'history_id': f'{VERSION}:{stage}:{index:04d}', 'seed': seed,
            'stage': stage, 'entities': entities, 'attribute': 'badge',
            'current_values': dict(zip(entities[:2], current)),
            'historical_value': historical[0] if historical else 'ivory',
            'distractor_values': historical[1:], 'n_distractors': n_distractors,
            'history_signature': digest([entities, current, historical])}


def generate_marker(stage='pilot', n_histories=2, seed=20261004):
    if stage not in ('pilot', 'confirmatory') or n_histories < 1:
        raise ValueError('invalid marker stage/count')
    rows = []
    stage_seed = seed + (0 if stage == 'pilot' else 1_000_003)
    for i in range(n_histories):
        rng = random.Random(stage_seed + i * 7919)
        h = _history(stage_seed, i, stage)
        values = rng.sample(VALUES, 6)
        h.update(current_values=dict(zip(h['entities'], values[:2])),
                 historical_values=dict(zip(h['entities'], values[2:4])),
                 replacement_values=dict(zip(h['entities'], values[4:6])))
        h['history_signature'] = digest([h['entities'], h['current_values'], h['historical_values']])
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
                histories[0] = 'Previously, ' + histories[0]
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
    # Paired cells must differ only by the marker prefix; value edit only the historical value.
    by = {r['example_id']: r for r in rows}
    for h in {r['history_id'] for r in rows}:
        keys = {(r['condition'], r['historical_order'], r['current_order'], r['edited_variable'],
                 r['query_variable'], r['edited'], r['marker']): r for r in rows if r['history_id'] == h}
        for condition, ho, co, ev, qv, edited in product(('superseded', 'entity_mention'), (0, 1), (0, 1), ('x', 'z'), ('x', 'z'), (0, 1)):
            a = keys[(condition, ho, co, ev, qv, edited, 0)]['prompt']
            b = keys[(condition, ho, co, ev, qv, edited, 1)]['prompt']
            al, *ar = a.split('\n'); bl, *br = b.split('\n')
            if bl != 'Previously, ' + al or ar != br:
                raise ValueError('marker altered content beyond historical sentence')
    return rows


def generate_harder(stage, n_histories=4, n_distractors=2, seed=20261005):
    if stage not in ('development', 'test') or n_distractors not in DIFFICULTIES:
        raise ValueError('harder stage or prespecified difficulty invalid')
    # Independent, disjoint deterministic namespaces.
    stage_seed = seed + (0 if stage == 'development' else 1_000_003)
    rows = []
    for i in range(n_histories):
        h = _history(stage_seed, i, stage, n_distractors)
        for condition, query_index, edited in product(HARD_CONDITIONS, (0, 1), (0, 1)):
            entities = h['entities']
            target = entities[query_index]
            hist_entity = entities[query_index]
            hist_value = h['historical_value']
            hist_line = (_assignment(hist_entity, h['attribute'], hist_value, 1)
                         if condition == 'superseded' else
                         _mention(hist_entity, hist_value, 1) if condition == 'entity_mention' else
                         f'The unassigned {h["attribute"]} value was {hist_value}.')
            if condition == 'late_unassigned':
                current_lines = [f"Currently, {e}'s {h['attribute']} is {h['current_values'][e]}." for e in entities[:2]]
                lines = current_lines + [hist_line]
            else:
                lines = [hist_line] + [f"Currently, {e}'s {h['attribute']} is {h['current_values'][e]}." for e in entities[:2]]
            lines += [f"{e} was also recorded in an unrelated note." for e in entities[2:]]
            prompt = '\n'.join(lines + [f"What is {target}'s current {h['attribute']}?", 'Respond with only the value, with no explanation.'])
            replacement = h['distractor_values'][0]
            if edited:
                prompt = prompt.replace(hist_value, replacement, 1)
            rows.append({**h, 'condition': condition,
                'query_entity': target, 'edited': edited, 'source_value': hist_value,
                'replacement_value': replacement, 'answer': h['current_values'][target],
                'stale_value': hist_value if condition == 'superseded' else None,
                'candidate_values': list(VALUES), 'prompt': prompt,
                'example_id': f'{h["history_id"]}:{condition}:q{query_index}:e{edited}'})
    return rows


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
        if not isinstance(masses, dict) or set(masses) != set(VALUES):
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
