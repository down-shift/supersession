"""Name-value distance experiment: generation, validation and analysis algebra (no model needed)."""

from __future__ import annotations

import random

import pytest

from src.cross_model.followups import (DISTANCE_ATTRIBUTES, DISTANCE_PAIRS, SUPERSEDED_NEAR_CANDIDATES,
                                       analyze_distance, generate_distance, validate_distance_dataset)
from src.cross_model.protocol import VALUES

NEAR = SUPERSEDED_NEAR_CANDIDATES[0]


def test_distance_dataset_is_a_complete_balanced_factorial():
    rows = generate_distance(n_histories=48, superseded_near=NEAR)
    assert len(rows) == 48 * 128
    histories = {r['history_id']: r for r in rows}
    assert len({r['history_signature'] for r in histories.values()}) == 48
    # v2 allocation: every entity pair x attribute x orientation cell once at 48 histories.
    cells = {(r['entity_pair'], r['attribute'], r['orientation']) for r in histories.values()}
    assert len(cells) == len(DISTANCE_PAIRS) * len(DISTANCE_ATTRIBUTES) * 2
    assert validate_distance_dataset(rows)['status'] == 'passed'


def test_distance_templates_and_edits():
    rows = generate_distance(n_histories=48, superseded_near=NEAR)
    lines = {(r['condition'], r['distance']): r['prompt'].split('\n')[0]
             for r in rows if r['history_index'] == 0 and not r['edited'] and r['historical_order'] == 0}
    e = rows[0]['entities'][0]
    v = rows[0]['historical_values'][e]
    a = rows[0]['attribute']
    assert lines[('superseded', 'far')] == f"Previously, {e}'s {a} was {v}."
    assert lines[('superseded', 'near')] == NEAR.format(entity=e, attribute=a, value=v)
    assert lines[('entity_mention', 'near')] == f'Previously, {e} mentioned {v} in an unrelated note.'
    assert lines[('entity_mention', 'far')] == f'Previously, {e}, in an unrelated note, mentioned {v}.'


def test_distance_rejects_unrecorded_template_and_excluded_histories():
    with pytest.raises(ValueError):
        generate_distance(n_histories=48, superseded_near="Previously, {entity}'s {attribute} is now {value}.")
    first = generate_distance(n_histories=48, superseded_near=NEAR)
    second = generate_distance(n_histories=48, superseded_near=NEAR,
                               excluded_signatures={r['history_signature'] for r in first})
    assert {r['history_signature'] for r in first}.isdisjoint({r['history_signature'] for r in second})


def test_distance_analysis_recovers_planted_query_relevance():
    """Plant a donor shift only under the matching query: R equals the planted size per cell."""
    rows = generate_distance(n_histories=48, superseded_near=NEAR)
    planted = {('superseded', 'near'): 3.0, ('superseded', 'far'): 1.0,
               ('entity_mention', 'near'): 6.0, ('entity_mention', 'far'): 2.0}
    rng = random.Random(0)
    base_masses = {}
    scores = []
    for r in rows:
        key = (r['history_id'], r['condition'], r['distance'], r['historical_order'], r['current_order'],
               r['edited_variable'], r['query_variable'])
        base = base_masses.setdefault(key, {v: -rng.uniform(5, 20) for v in VALUES})
        masses = dict(base)
        if r['edited'] and r['edited_variable'] == r['query_variable']:
            masses[r['replacement_value']] += planted[(r['condition'], r['distance'])]
        scores.append({**r, 'semantic_log_mass': masses})
    result = analyze_distance(rows, scores)['summary']
    for (cond, dist), size in planted.items():
        assert result[f'{cond}_{dist}_all_R']['mean'] == pytest.approx(size)
    assert result['superseded_distance_effect_all']['mean'] == pytest.approx(2.0)
    assert result['entity_mention_distance_effect_all']['mean'] == pytest.approx(4.0)
    assert result['distance_by_construction_interaction_all']['mean'] == pytest.approx(-2.0)
    assert result['construction_gap_near_all']['mean'] == pytest.approx(3.0)
    assert result['construction_gap_far_all']['mean'] == pytest.approx(1.0)
