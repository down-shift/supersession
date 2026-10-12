"""stale_decision_v2: the v1b tests run against the v2 modules, plus the v2 data changes."""
import math

import pytest

from src.data import stale_decision_v2 as data
from src.experiments import stale_decision_v1 as v1
from src.experiments import stale_decision_v2 as exp
from tests.test_stale_decision_v1 import Tokenizer


@pytest.fixture(scope='module')
def rows():
    return data.generate()


def test_action_events_have_no_leading_space_and_end_with_eos():
    tok = Tokenizer()
    events = exp.action_events(tok, 'prompt', ['APPROVE', 'DENY'])
    assert events['APPROVE'][0]['text'] == 'APPROVE'
    assert events['APPROVE'][0]['ids'] == [ord(c) for c in 'APPROVE'] + [tok.eos_token_id]
    assert v1.action_events(tok, 'prompt', ['APPROVE', 'DENY'])['APPROVE'][0]['ids'][0] == ord(' ')


def test_historical_prompt_asks_for_the_edited_old_value(rows):
    sup = [r for r in rows if r['family'] == 'superseded']
    for r in sup[:8]:
        body, answer = exp.historical_prompt(r)
        assert answer == r['old_values'][r['member']]
        assert body.startswith(exp.HISTORICAL_INTRO)
        assert 'INITIAL' in body and 'CURRENT clearance of' not in body.split('\n')[-2]
        assert f"had {'clearance' if r['task'] == 'access' else 'routing state'} {answer}." in body
    with pytest.raises(ValueError):
        exp.historical_prompt(next(r for r in rows if r['family'] == 'entity_mention'))


def _score(r):
    lp = {a: -3. for a in r['actions']}
    rec = dict(example_id=r['example_id'], row_hash=data.digest(r), model_id='test', action_logp=lp,
               easy_target={v: -3. for v in r['policy']}, easy_other={v: -3. for v in r['policy']},
               raw_text=r['answer'], easy_raw_text=r['correct_state'], generated_token_ids=[1, 0],
               easy_token_ids=[1, 0], eos_token_id=0, terminated=True, easy_terminated=True,
               generated=v1.classify(r['answer'], r['actions'], r['answer']),
               easy_generated=v1.classify(r['correct_state'], list(r['policy']), r['correct_state']),
               canonical_valid_mass=sum(math.exp(v) for v in lp.values()), prompt_tokens=100, history_position=10,
               historical=None)
    if r['family'] == 'superseded':
        old = r['old_values'][r['member']]
        rec['historical'] = {'scores': {v: -3. for v in r['policy']}, 'raw_text': old, 'token_ids': [1, 0],
                             'terminated': True, 'answer': old,
                             'generated': v1.classify(old, list(r['policy']), old, True)}
    return rec


def test_score_record_requires_historical_only_in_superseded(rows):
    sup = next(r for r in rows if r['family'] == 'superseded')
    other = next(r for r in rows if r['family'] == 'unassigned')
    exp.check_score_record(sup, _score(sup))
    exp.check_score_record(other, _score(other))
    bad = _score(sup); bad['historical'] = None
    with pytest.raises(ValueError):
        exp.check_score_record(sup, bad)
    bad = _score(other); bad['historical'] = _score(sup)['historical']
    with pytest.raises(ValueError):
        exp.check_score_record(other, bad)


def test_h3_keeps_downstream_confidence_out_of_pre_downstream_predictors(rows):
    pytest.importorskip('sklearn')
    from src.analysis import stale_decision_v1 as v1a
    from src.analysis import stale_decision_v2 as an
    scores = [_score(r) for r in rows]
    # Make failures depend on downstream confidence only: a leak would show up as a near-perfect pre model.
    for s, r in zip(scores, rows):
        if r['family'] in an.HISTORICAL and data.digest(r['example_id'])[-1] in '0123':
            s['action_logp'] = {r['answer']: -4., r['wrong_action']: -0.1}
            s['raw_text'] = r['wrong_action']
            s['generated'] = v1.classify(r['wrong_action'], r['actions'], r['answer'])
            s['canonical_valid_mass'] = sum(math.exp(v) for v in s['action_logp'].values())
    with an.v1b_validation():
        effects = v1a.cells(rows, scores)
    out = an.prediction(rows, scores, effects)
    assert out['available']
    assert out['contemporaneous_action_confidence']['auroc'] > 0.99
    assert out['pre_downstream']['auroc'] < 0.75


def test_full_analysis_runs_and_reports_historical_retrieval(rows):
    pytest.importorskip('sklearn')
    from src.analysis import stale_decision_v2 as an
    result = an.analyze(rows, [_score(r) for r in rows])
    assert result['historical_retrieval']['overall']['correct']['mean'] == 1.0
    assert result['H3_v1b']['failure']['overall']['available'] is False
    assert 'H3_secondary' not in result


def test_config_requires_scorer_choice():
    import json
    config = json.load(open('configs/stale_decision_v2/protocol.json'))
    exp.check_config(config)
    with pytest.raises(ValueError):
        exp.check_config({**config, 'scorer': 'fast'})
    with pytest.raises(ValueError):
        exp.check_config({**config, 'protocol': 'stale_decision_v1b'})


def test_float32_rounding_tolerance_accepts_tiny_positive_log_probability(rows):
    pytest.importorskip('sklearn')
    from src.analysis import stale_decision_v2 as an
    scores = [_score(r) for r in rows]
    r0 = rows[0]
    scores[0]['easy_target'] = {v: (2.4e-7 if v == r0['correct_state'] else -20.) for v in r0['policy']}
    with pytest.raises(ValueError):
        v1.check_score_record(r0, scores[0])
    exp.check_score_record(r0, scores[0])
    exp.validate_easy(rows, [{k: scores[0][k] for k in ('example_id', 'row_hash', 'easy_target', 'easy_other',
                                                         'easy_raw_text', 'easy_token_ids', 'easy_terminated',
                                                         'eos_token_id', 'easy_generated')}])
    an.analyze(rows, scores)
    bad = dict(scores[0], easy_target={**scores[0]['easy_target'], r0['correct_state']: 1e-3})
    with pytest.raises(ValueError):
        exp.check_score_record(r0, bad)


def test_generated_switches_count_paired_wrong_changes(rows):
    pytest.importorskip('sklearn')
    from src.analysis import stale_decision_v2 as an
    scores = [_score(r) for r in rows]
    for s, r in zip(scores, rows):
        if r['family'] == 'superseded' and r['member'] == 0 and r['task'] == 'access':
            s['raw_text'] = r['wrong_action']
            s['generated'] = v1.classify(r['wrong_action'], r['actions'], r['answer'])
    out = an.generated_switches(rows, scores)
    assert out['access:all']['switch_superseded']['mean'] == 1.0
    assert out['access:all']['switch_counts']['superseded'] == {'to_wrong': 96, 'to_correct': 0, 'n_histories': 96}
    assert out['access:all']['switch_Delta_semantic']['mean'] == 1.0
    assert out['routing:all']['switch_superseded']['mean'] == 0.0


def test_v2_data_changes_only_format_seeds_and_pools(rows):
    from src.data import stale_decision_v1 as v1data
    old = v1data.generate()
    assert data.SEEDS == {'development': 82001, 'frozen_gate': 82002, 'confirmation': 82003}
    assert all(not r['prompt'].endswith('Answer:') for r in rows)
    assert all(r['entities'][0].startswith('Dvb') for r in rows)
    assert len(rows) == len(old) == 2112
    # same factorial structure: every non-prompt design field matches cell by cell, except names/seed/schema
    keep = ('task', 'difficulty', 'vocabulary', 'family', 'member', 'historical_order', 'current_order', 'order_stratum', 'template')
    assert [tuple(r[k] for k in keep) for r in rows] == [tuple(r[k] for k in keep) for r in old]
    data.verify_no_overlap(rows, data.generate('frozen_gate', 2))
