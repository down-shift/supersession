import copy
import math
from collections import defaultdict

import pytest

from src.cross_model.dataset import generate
from src.cross_model.protocol import VALUES
from src.cross_model.reanalysis import pair_diagnostics, reanalyze_rows
from src.cross_model.score_checks import checked_scores, strict_rank
from src.data.supersession_behavior import render_behavior_example


def saved_scores(rows):
    scores = []
    for r in rows:
        masses = {v: -.1 if v == r['answer'] else -10. for v in VALUES}
        if r['pair_direction'] and r['condition'] in ('superseded', 'irrelevant_counterbalanced'):
            associated = r['edited_variable'] == r['query']
            effect = (6. if associated else 0.) if r['condition'] == 'superseded' else (
                4. if associated == (r['unassigned_slot_order'] == 'xz') else 0.)
            masses[r['replacement_value']] += effect
        rank = strict_rank(masses, r['answer'])
        scores.append({**copy.deepcopy(r), 'score_kind': 'confirmatory',
                       'prompt': render_behavior_example(r, None, False),
                       'semantic_log_mass': masses, 'semantic_rank': rank, 'semantic_accuracy': int(rank == 1),
                       'surface_likelihoods': {v: [{'text': ' '+v, 'ids': [i], 'log_probability': masses[v]}]
                                               for i, v in enumerate(VALUES)}})
    return scores


@pytest.fixture
def archive():
    rows = generate('validation', 2)
    return rows, saved_scores(rows)


def test_member_specific_margin_and_fixed_competitor_are_distinct(archive):
    rows, scores = archive
    b = next(s for s in scores if s['condition'] == 'superseded' and s['pair_direction'] == 0)
    e = copy.deepcopy(next(s for s in scores if s['pair_id'] == b['pair_id'] and s['pair_direction'] == 1))
    answer = b['answer']
    alternatives = [v for v in VALUES if v != answer]
    first, second = alternatives[:2]
    b['semantic_log_mass'] = {v: -10. for v in VALUES}
    e['semantic_log_mass'] = {v: -10. for v in VALUES}
    b['semantic_log_mass'].update({answer: -1., first: -2.})
    e['semantic_log_mass'].update({answer: -1., first: -4., second: -.5})
    ranks = {s['example_id']: strict_rank(s['semantic_log_mass'], s['answer']) for s in (b, e)}
    out = pair_diagnostics(b, e, ranks)
    assert out['baseline_competitor'] == first
    assert out['edited_competitor'] == second
    assert out['decision_margin_before'] == pytest.approx(1.)
    assert out['decision_margin_after'] == pytest.approx(-.5)
    assert out['decision_margin_change'] == pytest.approx(-1.5)
    assert out['member_target_vs_fixed_baseline_competitor_change'] == pytest.approx(2.)


def test_order_specific_R_sign_reversal_and_decomposition(archive):
    rows, scores = archive
    report = reanalyze_rows(rows, scores)
    order = report['counterbalanced_unassigned_R_by_mention_order']['history_rows']
    assert len(order) == 2
    for r in order:
        assert r['R_xz_identity_transfer'] == pytest.approx(4.)
        assert r['R_zx_identity_transfer'] == pytest.approx(-4.)
    for r in report['history_rows']:
        assert r['R_irrelevant_counterbalanced'] == pytest.approx(0.)
        assert r['R_superseded_minus_R_irrelevant_counterbalanced'] == pytest.approx(6.)
    components = report['complete_relevance_decomposition']
    for total, source, replacement in zip(report['history_rows'], components['negative_source_component']['history_rows'],
                                           components['replacement_component']['history_rows']):
        for k in total:
            if k != 'history_id': assert total[k] == pytest.approx(source[k]+replacement[k])


def test_secondary_uses_complete_histories_and_both_irrelevant_orders(archive):
    rows, scores = archive
    bad_hid = rows[0]['history_id']
    for s in scores:
        if s['history_id'] == bad_hid and s['condition'] == 'superseded':
            # The same baseline prompt represents both edit metadata labels.
            # Change one fixed competitor consistently for every such prompt.
            s['semantic_log_mass'][rows[0]['replacement_values']['initial_x']] = 0.
            s['semantic_rank'] = strict_rank(s['semantic_log_mass'], s['answer'])
            s['semantic_accuracy'] = 0
            s['surface_likelihoods'][rows[0]['replacement_values']['initial_x']][0]['log_probability'] = 0.
    report = reanalyze_rows(rows, scores)
    assert report['n_histories'] == 2 and report['all_trial_primary']
    secondary = report['secondary_conditioned_supersession_fully_correct']
    assert secondary['n_histories'] == 1 and secondary['n_matched_members'] == 40
    assert bad_hid not in secondary['included_history_ids']
    assert secondary['primary_contrast']['mean'] == pytest.approx(6.)
    assert secondary['history_rows'][0]['R_irrelevant_counterbalanced'] == pytest.approx(0.)


@pytest.mark.parametrize('failure', ['duplicate_score', 'missing_score', 'duplicate_pair_member', 'wrong_rank',
                                   'tie', 'nonfinite', 'metadata', 'missing_order', 'surface_mass'])
def test_reanalysis_rejects_incomplete_or_inconsistent_archive(archive, failure):
    rows, scores = archive
    if failure == 'duplicate_score':
        scores.append(copy.deepcopy(scores[0]))
    elif failure == 'missing_score':
        scores.pop()
    elif failure == 'duplicate_pair_member':
        # Keep unique IDs and count, but overwrite a pair/member identity.
        for r in (rows[-1], scores[-1]):
            r['pair_id'] = rows[0]['pair_id']
            r['pair_direction'] = 0
    elif failure == 'wrong_rank':
        scores[0]['semantic_rank'] = 2
    elif failure == 'tie':
        scores[0]['semantic_log_mass'] = {v: -.1 for v in VALUES}
        # A saved rank of 1 is stale: strict rank is 8 and must be compared.
    elif failure == 'nonfinite':
        scores[0]['semantic_log_mass'][VALUES[0]] = math.nan
    elif failure == 'metadata':
        scores[0]['query'] = 'z' if rows[0]['query'] == 'x' else 'x'
    elif failure == 'missing_order':
        rows = [r for r in rows if r.get('unassigned_slot_order') != 'zx']
        scores = [s for s in scores if s.get('unassigned_slot_order') != 'zx']
    else:
        scores[0]['surface_likelihoods'][VALUES[0]][0]['log_probability'] -= 1
    with pytest.raises(ValueError):
        reanalyze_rows(rows, scores)


def test_rank_recomputation_treats_ties_as_incorrect():
    masses = {v: -10. for v in VALUES}
    masses['amber'] = masses['jade'] = -1.
    assert strict_rank(masses, 'jade') == 2


def test_live_reports_answer_identity_changes_separately(archive):
    report = reanalyze_rows(*archive)
    conditions = report['condition_diagnostics']
    assert conditions['live']['answer_identity_changed_pairs'] == 4
    assert 'identities can differ' in conditions['live']['member_target_score_change_definition']
    for condition in ('superseded', 'irrelevant', 'irrelevant_counterbalanced'):
        assert conditions[condition]['answer_identity_changed_pairs'] == 0
        assert 'same fixed current-answer identity' in conditions[condition]['member_target_score_change_definition']
    assert 'paired_current_score_change' not in report  # no pooled live/fixed-answer statistic
