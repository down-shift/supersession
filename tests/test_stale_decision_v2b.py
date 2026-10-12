"""stale_decision_v2b: balanced other-entity rows and the routing-only analysis on synthetic scores."""
import json

import pytest

from src.data import stale_decision_v2b as data
from src.experiments import stale_decision_v1 as v1
from src.experiments import stale_decision_v2b as exp
from tests.test_stale_decision_v2 import _score as v2_score


def _score(r):
    rec = v2_score(r)
    rec['row_hash'] = data.digest(r)
    return rec


@pytest.fixture(scope='module')
def rows():
    return data.generate('balanced', 2)


def test_balanced_rows(rows):
    data.validate(rows, expected_histories=192)
    assert {r['task'] for r in rows} == {'routing'}
    for r in rows:
        assert r['currents'][1] == r['currents'][0] and r['policy'][r['currents'][1]] == r['policy'][r['currents'][0]]
    bad = dict(rows[0], currents=[rows[0]['currents'][0], rows[0]['old_values'][0]])
    with pytest.raises(ValueError):
        data.validate([bad])


def test_config_and_analysis(rows):
    exp.check_config(json.load(open('configs/stale_decision_v2b/protocol.json')))
    scores = [_score(r) for r in rows]
    for s, r in zip(scores, rows):
        if r['family'] == 'superseded' and r['member'] == 0:
            s['raw_text'] = r['wrong_action']
            s['generated'] = v1.classify(r['wrong_action'], r['actions'], r['answer'])
    out = exp.analyze_balanced(rows, scores)
    assert out['n_histories'] == 192
    assert out['switch_superseded']['mean'] == 1.0 and out['switch_counts_superseded']['to_wrong'] == 192
    assert out['tracked_superseded']['n'] == 192 and out['switch_entity_mention']['mean'] == 0.0
