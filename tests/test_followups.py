from collections import defaultdict

from src.cross_model.followups import (DIFFICULTIES, choose_difficulty,
    analyze_marker, complete_answer_outcome, generate_harder, generate_marker,
    validate_marker_templates)


def test_marker_examples_and_prefix_audit():
    audit = validate_marker_templates()
    assert audit['status'] == 'passed'
    rows = generate_marker(n_histories=2)
    assert len(rows) == 2 * 2 * 2 * 2 * 2 * 2 * 2 * 2
    pairs = defaultdict(dict)
    for row in rows:
        key = (row['history_id'], row['condition'], row['historical_order'], row['current_order'],
               row['edited_variable'], row['query_variable'], row['edited'])
        pairs[key][row['marker']] = row['prompt']
    assert all(pair[1].split('\n')[0] == 'Previously, ' + pair[0].split('\n')[0]
               and pair[1].split('\n')[1:] == pair[0].split('\n')[1:]
               for pair in pairs.values())
    assert {r['historical_order'] for r in rows} == {0, 1}
    assert {r['current_order'] for r in rows} == {0, 1}
    confirm = generate_marker('confirmatory', n_histories=2)
    assert {r['history_signature'] for r in rows}.isdisjoint({r['history_signature'] for r in confirm})


def test_harder_stage_separation_conditions_and_unambiguous_answer():
    dev = generate_harder('development', n_histories=2, n_distractors=2)
    test = generate_harder('test', n_histories=2, n_distractors=2)
    assert {r['history_id'] for r in dev}.isdisjoint({r['history_id'] for r in test})
    assert {r['history_signature'] for r in dev}.isdisjoint({r['history_signature'] for r in test})
    assert {r['condition'] for r in dev} == {'superseded', 'entity_mention', 'early_unassigned', 'late_unassigned'}
    for row in dev + test:
        assert row['answer'] in row['candidate_values']
        assert row['answer'] != row['stale_value']
        assert row['n_distractors'] in DIFFICULTIES


def test_difficulty_rule_and_complete_answer_parser():
    assert choose_difficulty({'2': {'accuracy': .8}, '4': {'accuracy': .7}, '6': {'accuracy': .4}}) == 4
    assert complete_answer_outcome('  "amber."\n', 'amber', 'coral')['category'] == 'correct'
    assert complete_answer_outcome('coral', 'amber', 'coral')['category'] == 'stale'
    assert complete_answer_outcome('amber is the badge', 'amber', 'coral')['category'] == 'other'


def test_marker_analysis_uses_complete_history_factorial():
    rows = generate_marker(n_histories=1)
    scores = [{**row, 'semantic_log_mass': {v: (0.0 if v == 'amber' else -1.0) for v in row['candidate_values']}}
              for row in rows]
    result = analyze_marker(rows, scores)
    assert len(result['history_rows']) == 1
    assert result['summary']['marker_by_construction_interaction_all']['n_histories'] == 1
    assert result['summary']['superseded_m0_all_E']['mean'] == 0
