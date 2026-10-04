from argparse import Namespace
from collections import defaultdict
import json
from pathlib import Path

import pytest

from src.cross_model.followups import (DIFFICULTIES, analyze_harder, analyze_marker,
    choose_difficulty, complete_answer_outcome, generate_harder, generate_marker,
    validate_marker_templates)
from src.cross_model.protocol import VALUES, sealed, read_sealed
from scripts.followups import main as generate_cli
from scripts.run_followups import (_code_hash, select_difficulty, run as score_cli,
                                  dataset_lineage, validate_test_runtime)


def test_marker_marker_prefixes_both_history_lines_and_stages_are_fresh():
    audit = validate_marker_templates()
    assert audit['status'] == 'passed'
    pilot = generate_marker(n_histories=24)
    confirm = generate_marker('confirmatory', n_histories=24,
                              excluded_signatures={r['history_signature'] for r in pilot})
    assert {r['history_signature'] for r in pilot}.isdisjoint({r['history_signature'] for r in confirm})
    assert len({r['history_signature'] for r in pilot + confirm}) == 48
    pairs = defaultdict(dict)
    for row in pilot:
        key = (row['history_id'], row['condition'], row['historical_order'], row['current_order'],
               row['edited_variable'], row['query_variable'], row['edited'])
        pairs[key][row['marker']] = row['prompt'].split('\n')
    assert all(marked[:2] == ['Previously, ' + bare[0], 'Previously, ' + bare[1]]
               and marked[2:] == bare[2:] for pair in pairs.values()
               for bare, marked in [(pair[0], pair[1])])


def test_marker_requires_complete_factorial_and_known_nonzero_estimands():
    rows = generate_marker(n_histories=1)
    assert all(r['stale_value'] == r['replacement_value'] for r in rows
               if r['condition'] == 'superseded' and r['edited']
               and r['edited_variable'] == r['query_variable'])
    scores = []
    for row in rows:
        masses = {v: 0.0 for v in VALUES}
        if row['edited']:
            construction = row['condition']
            marker = row['marker']
            if construction == 'superseded':
                desired_r = 1.0 if marker == 0 else 4.0
            else:
                desired_r = 2.0 if marker == 0 else 4.0
            desired_e = desired_r if row['edited_variable'] == row['query_variable'] else 0.0
            masses[row['replacement_value']] = desired_e
        scores.append({**row, 'semantic_log_mass': masses})
    result = analyze_marker(rows, scores)
    summary = result['summary']
    assert summary['superseded_m0_all_R']['mean'] == pytest.approx(1.0)
    assert summary['superseded_m1_all_R']['mean'] == pytest.approx(4.0)
    assert summary['entity_mention_m0_all_R']['mean'] == pytest.approx(2.0)
    assert summary['entity_mention_m1_all_R']['mean'] == pytest.approx(4.0)
    assert summary['superseded_marker_effect_all']['mean'] == pytest.approx(3.0)
    assert summary['entity_mention_marker_effect_all']['mean'] == pytest.approx(2.0)
    assert summary['marker_by_construction_interaction_all']['mean'] == pytest.approx(1.0)
    assert summary['superseded_m1_all_R_replacement_component']['mean'] == pytest.approx(4.0)
    assert summary['superseded_m1_all_R_source_component']['mean'] == pytest.approx(0.0)
    known_cell = next(r for r in result['edit_effect_rows'] if r['condition'] == 'superseded'
        and r['marker'] == 0 and r['historical_order'] == 0 and r['current_order'] == 0
        and r['edited_variable'] == r['query_variable'] == 'x')
    assert known_cell['E'] == pytest.approx(1.0)
    assert known_cell['E_replacement_component'] == pytest.approx(1.0)
    assert known_cell['E_source_component'] == pytest.approx(0.0)
    incomplete = [r for r in rows if (r['historical_order'], r['current_order']) != (0, 0)]
    incomplete_scores = [s for s in scores if s['example_id'] in {r['example_id'] for r in incomplete}]
    with pytest.raises(ValueError, match='incomplete marker factorial'):
        analyze_marker(incomplete, incomplete_scores)


def test_harder_context_crosses_both_queries_and_uses_real_distractor_updates():
    pilot = generate_harder('pilot', n_histories=2, n_distractors=2)
    dev = generate_harder('development', n_histories=24, n_distractors=2,
                          excluded_signatures={r[k] for r in pilot for k in ('history_signature', 'target_history_signature')})
    test = generate_harder('test', n_histories=24, n_distractors=2,
                           excluded_signatures={r[k] for r in pilot + dev for k in ('history_signature', 'target_history_signature')})
    assert {r['history_signature'] for r in pilot + dev}.isdisjoint({r['history_signature'] for r in test})
    assert len({r['history_signature'] for r in pilot + dev + test}) == 50
    for row in dev:
        assert row['answer'] != row['stale_value']
        for entity, values in row['distractor_assignments'].items():
            assert values['historical'] != values['current']
            assert f"{entity}'s badge was {values['historical']}" in row['prompt']
            assert f"{entity}'s badge is {values['current']}" in row['prompt']
    context = defaultdict(set)
    for row in dev:
        context[row['context_id']].add(row['query_variable'])
    assert all(v == {'x', 'z'} for v in context.values())
    edited = next(r for r in dev if r['edited'] and r['edited_variable'] == r['query_variable'])
    assert edited['stale_value'] == edited['replacement_value']
    assert complete_answer_outcome(edited['replacement_value'], edited['answer'],
                                   edited['stale_value'])['category'] == 'stale'


def test_harder_analysis_uses_fixed_context_query_crossing_and_all_transition_cells():
    rows = generate_harder('pilot', n_histories=1, n_distractors=2)
    scores = []
    for row in rows:
        if row['edited'] and row['edited_variable'] != row['query_variable']:
            answer = row['source_value'] if row['edited_variable'] == 'x' else row['replacement_value']
        else:
            answer = row['answer'] if not row['edited'] else row['stale_value']
        category = complete_answer_outcome(answer, row['answer'], row['stale_value'])['category']
        masses = {v: -1.0 for v in VALUES}
        masses[row['answer']] = 1.0
        scores.append({**row, 'answer_category': category, 'parsed_answer': answer,
            'generated_answer': answer,
            'query_source_response': answer == row['source_value_for_query'],
            'query_donor_response': answer == row['replacement_value_for_query'],
            'edited_source_response': answer == row['source_value'],
            'edited_donor_response': answer == row['replacement_value'],
            'semantic_log_mass': masses, 'current_minus_historical_log_mass': 2.0})
    result = analyze_harder(rows, scores)
    assert result['overall_complete_answer_accuracy'] == pytest.approx(.5)
    assert sum(result['condition_order_transition_counts'].values()) == len(rows) // 2
    assert result['summaries']['superseded_stale_edit_change']['mean'] == pytest.approx(.5)
    assert 'superseded_minus_entity_mention_stale_edit_change' in result['stale_edit_change_contrasts']
    cross_entity = [r for r in result['paired_edit_effect_rows'] if r['edited_variable'] != r['query']]
    assert any(r['edited_category'] == 'other' for r in cross_entity)
    assert any(r['intervention_source_response'] == 1 for r in cross_entity)
    assert any(r['intervention_donor_response'] == 1 for r in cross_entity)
    corrupted = [dict(s) for s in scores]
    corrupted[0]['answer_category'] = 'other'
    with pytest.raises(ValueError, match='parser'):
        analyze_harder(rows, corrupted)
    corrupted = [dict(s) for s in scores]
    corrupted[0]['current_minus_historical_log_mass'] = 99.0
    with pytest.raises(ValueError, match='margin disagrees'):
        analyze_harder(rows, corrupted)
    with pytest.raises(ValueError, match='duplicate'):
        analyze_harder(rows, scores + [scores[0]])


def test_stage_labels_reject_invalid_experiment_combinations(monkeypatch, tmp_path):
    monkeypatch.setattr('sys.argv', ['followups.py', 'marker', '--stage', 'test', '--output', str(tmp_path/'x')])
    with pytest.raises(SystemExit):
        generate_cli()


def test_difficulty_rule_freeze_binds_matched_development_histories(tmp_path):
    assert choose_difficulty({'2': {'accuracy': .8}, '4': {'accuracy': .7}, '6': {'accuracy': .4}}) == 4
    paths = []
    targets = [f'h{i:02d}' for i in range(12)]
    runtime_fp = {'configuration_sha256': 'cfg', 'parameter_dtypes': ['torch.float16'],
                  'packages': {'torch': '2.14.0'}, 'gpu_name': 'test-gpu'}
    for level, accuracy in zip((2, 4, 6), (.95, .70, .45)):
        report = sealed({'protocol': 'relational_followups_v1', 'experiment': 'harder',
            'stage': 'development', 'overall_complete_answer_accuracy': accuracy,
            'n_distractors': level, 'history_signatures': [f'full{level}'],
            'target_history_signatures': targets,
            'dataset_sha256': f'data{level}', 'scores_sha256': f'score{level}',
            'code_sha256': _code_hash(),
                'dataset_seeds': [20261005], 'inference_provenance': {'fingerprint': {
                    'experiment': 'harder', 'model_id': 'Qwen/Qwen3-8B',
                    'model_revision': 'abc', 'max_new_tokens': 32, 'seed': 20261006,
                    'code_sha256': _code_hash(), 'runtime_fingerprint': runtime_fp}}})
        path = tmp_path / f'n{level}.json'
        path.write_text(json.dumps(report))
        paths.append(str(path))
    output = tmp_path / 'freeze.json'
    select_difficulty(Namespace(development_reports=paths, output=str(output)))
    frozen = read_sealed(output)
    assert frozen['selected_n_distractors'] == 4
    assert frozen['excluded_history_signatures'] == targets
    assert frozen['test_histories'] == 24
    runtime_fp['gpu_name'] = 'different-gpu'
    report = sealed({'protocol': 'relational_followups_v1', 'experiment': 'harder',
        'stage': 'development', 'overall_complete_answer_accuracy': .45,
        'n_distractors': 6, 'history_signatures': ['full6'], 'target_history_signatures': targets,
        'dataset_sha256': 'data6', 'scores_sha256': 'score6', 'code_sha256': _code_hash(),
        'dataset_seeds': [20261005], 'inference_provenance': {'fingerprint': {
            'experiment': 'harder', 'model_id': 'Qwen/Qwen3-8B', 'model_revision': 'abc',
            'max_new_tokens': 32, 'seed': 20261006, 'code_sha256': _code_hash(),
            'runtime_fingerprint': runtime_fp}}})
    paths[-1] = str(tmp_path / 'different_runtime.json')
    (tmp_path / 'different_runtime.json').write_text(json.dumps(report))
    with pytest.raises(ValueError, match='incompatible configurations or runtimes'):
        select_difficulty(Namespace(development_reports=paths, output=str(tmp_path/'bad_freeze.json')))
    for level, accuracy in zip((2, 4, 6), (.91, .95, 1.0)):
        report_path = tmp_path / f'n{level}.json'
        report = read_sealed(report_path)
        report['overall_complete_answer_accuracy'] = accuracy
        report_path.write_text(json.dumps(sealed(report)))
    paths[-1] = str(tmp_path / 'n6.json')
    stopped_freeze = tmp_path / 'stopped_freeze.json'
    with pytest.raises(ValueError, match='amended stopping rule'):
        select_difficulty(Namespace(development_reports=paths, output=str(stopped_freeze)))
    assert not stopped_freeze.exists()
    for model, history_count, error in (
            ('google/gemma-3-4b-it', 12, 'requires Qwen'),
            ('Qwen/Qwen3-8B', 2, 'requires 12 histories')):
        for path in paths:
            report = read_sealed(path)
            report['overall_complete_answer_accuracy'] = .8
            report['target_history_signatures'] = targets[:history_count]
            report['inference_provenance']['fingerprint']['model_id'] = model
            Path(path).write_text(json.dumps(sealed(report)))
        with pytest.raises(ValueError, match=error):
            select_difficulty(Namespace(development_reports=paths, output=str(stopped_freeze)))
        assert not stopped_freeze.exists()


def test_failed_difficulty_gate_cannot_generate_or_score_test_data(monkeypatch, tmp_path):
    freeze_path = tmp_path / 'fallback.json'
    freeze_path.write_text(json.dumps(sealed({'protocol': 'relational_followups_v1',
        'experiment': 'harder', 'stage': 'frozen_test_protocol',
        'selection_gate_passed': False, 'selected_n_distractors': 2})))
    prior = tmp_path / 'pilot.jsonl'
    prior.write_text('\n'.join(json.dumps(r) for r in generate_harder('pilot', 1)))
    output = tmp_path / 'test.jsonl'
    monkeypatch.setattr('sys.argv', ['followups.py', 'harder', '--stage', 'test',
        '--freeze', str(freeze_path), '--prior-dataset', str(prior), '--output', str(output)])
    with pytest.raises(ValueError, match='qualifying sealed protocol'):
        generate_cli()
    assert not output.exists()
    rows = generate_harder('test', 1, excluded_signatures=set())
    output.write_text('\n'.join(json.dumps(r) for r in rows))
    scores = tmp_path / 'scores.jsonl'
    with pytest.raises(ValueError, match='qualifying sealed protocol'):
        score_cli(Namespace(config='configs/cross_model_relational_v2/qwen3_8b.yaml',
            dataset=str(output), experiment='harder', mode='both', freeze=str(freeze_path),
            output=str(scores)))
    assert not scores.exists()


def test_generation_exclusions_survive_scoring_lineage_and_detect_changed_prior(monkeypatch, tmp_path):
    pilot = tmp_path / 'pilot.jsonl'
    prior_rows = generate_harder('pilot', 1)
    pilot.write_text('\n'.join(json.dumps(r) for r in prior_rows))
    dataset = tmp_path / 'development.jsonl'
    report = tmp_path / 'development_report.json'
    monkeypatch.setattr('sys.argv', ['followups.py', 'harder', '--stage', 'development',
        '--histories', '1', '--prior-dataset', str(pilot),
        '--output', str(dataset), '--report', str(report)])
    generate_cli()
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    lineage = dataset_lineage(str(dataset), rows)
    assert set(lineage['exclusions']) == {r[k] for r in prior_rows
        for k in ('history_signature', 'target_history_signature')}
    assert lineage['prior_datasets'][0]['sha256']
    assert lineage['dataset_report_sha256']
    pilot.write_text(pilot.read_text() + '\n')
    with pytest.raises(ValueError, match='exclusion ledger is missing or changed'):
        dataset_lineage(str(dataset), rows)


def test_qwen_test_requires_frozen_runtime_but_gemma_can_use_its_own():
    runtime = {'python_version': '3.13.5', 'uv_lock_sha256': 'lock', 'gpu_name': 'gpu'}
    frozen = {'model_id': 'Qwen/Qwen3-8B', 'model_revision': 'qwen-revision',
        'selected_n_distractors': 4, 'development_levels': {'4': {'runtime_fingerprint': runtime}}}
    config = {'id': 'Qwen/Qwen3-8B', 'revision': 'qwen-revision'}
    validate_test_runtime(frozen, config, runtime)
    for changed in ({**runtime, 'uv_lock_sha256': 'different'},
                    {**runtime, 'python_version': '3.14.0'},
                    {**runtime, 'gpu_name': 'different'}):
        with pytest.raises(ValueError, match='runtime differs'):
            validate_test_runtime(frozen, config, changed)
    with pytest.raises(ValueError, match='runtime differs'):
        validate_test_runtime(frozen, {**config, 'revision': 'different'}, runtime)
    validate_test_runtime(frozen, {'id': 'google/gemma-3-4b-it', 'revision': 'gemma-revision'},
                          {'gpu_name': 'gemma-gpu'})
