"""stale_decision_v2s: static other-attribute rows and the paired analysis on synthetic scores."""
import json

import pytest

from src.data import stale_decision_v2 as v2
from src.data import stale_decision_v2s as data
from src.experiments import stale_decision_v1 as v1
from src.experiments import stale_decision_v2s as exp
from tests.test_stale_decision_v2 import _score as v2_score


def _score(r, model='Qwen/Qwen3-8B'):
    rec = v2_score(r)
    rec['row_hash'] = data.digest(r)
    rec['model_id'] = model
    return rec


@pytest.fixture(scope='module')
def rows():
    return data.generate()


def test_static_rows(rows):
    assert sum(r['family'] == 'static_other' for r in rows) == 768
    assert sum(r['family'] == 'updated_other' for r in rows) == 2 * data.CHECK_HISTORIES
    src = {r['example_id']: r for r in data.sources()}
    for r in rows:
        s = src[r['source_example_id']]
        if r['family'] == 'updated_other':
            assert r['prompt'] == s['prompt']
            continue
        # only the queried entity's archive-tag update changes; the edited word stays in its slot
        assert data.archive_update(r) in s['prompt'] and data.archive_update(r) not in r['prompt']
        assert r['prompt'].replace(data.FILLER.format(entity=r['entities'][0]), data.archive_update(r)) == s['prompt']
        assert r['prompt'][slice(*r['edit_span'])] == r['old_values'][r['member']]
        assert len(data.FILLER.format(entity='E').split()) == len(data.archive_update(dict(r, entities=['E'])).split())
    bad = dict(rows[0], prompt=rows[0]['prompt'] + ' ')
    with pytest.raises(ValueError):
        data.validate([bad] + rows[1:])


def test_configs():
    for name in ('qwen3_8b', 'qwen3_14b'):
        exp.check_config(json.load(open(f'configs/stale_decision_v2s/{name}.json')))


def test_analysis(rows):
    sealed_rows = v2.generate('confirmation', 4)
    sealed = [_score(r) for r in sealed_rows]
    sealed_by_row = {r['example_id']: r for r in sealed_rows}
    for s in sealed:
        s['row_hash'] = v2.digest(sealed_by_row[s['example_id']])
    scores = [_score(r) for r in rows]
    for s, r in zip(scores, rows):  # static member 0 switches to the wrong action in every history
        if r['family'] == 'static_other' and r['member'] == 0:
            s['raw_text'] = r['wrong_action']
            s['generated'] = v1.classify(r['wrong_action'], r['actions'], r['answer'], s['terminated'])
    out = exp.analyze_static(rows, scores, sealed_rows, sealed)
    assert out['n_histories'] == 384
    assert out['constructions']['static_other']['switch']['mean'] == 1.0
    assert out['contrasts']['Delta_update']['switch']['mean'] == -1.0
    assert out['reproducibility']['pairs_records'] == 2 * data.CHECK_HISTORIES
