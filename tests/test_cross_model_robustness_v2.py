import copy
import json
from collections import defaultdict

import pytest

from src.cross_model import robustness_v2 as data
from src.cross_model import robustness_protocol as design
from src.cross_model.robustness_analysis import contrasts, summary, normalized
from src.cross_model.score_checks import strict_rank
from src.cross_model.tokens import validate as validate_tokens
from src.data.io import write_jsonl
from src.utils import load_config


@pytest.fixture(scope='module')
def development():
    return data.generate('development')


def synthetic_scores(rows, kind='confirmatory'):
    result = []
    for r in rows:
        masses = {v: -.1 if v == r['answer'] else -10. for v in data.VALUES}
        result.append({**copy.deepcopy(r), 'score_kind': kind, 'prompt': data.render(r, None, False),
                       'semantic_log_mass': masses, 'semantic_rank': strict_rank(masses, r['answer']),
                       'semantic_accuracy': 1})
    return result


def test_full_factorial_includes_edited_entity_and_every_member(development):
    histories, rows = development
    assert len(rows) == 24 * 192
    assert len({r['example_id'] for r in rows}) == len(rows)
    for h in histories:
        cells = {tuple(r[k] for k in ('condition', 'historical_entity_order', 'current_entity_order',
                                     'edited_variable', 'query', 'pair_direction'))
                 for r in rows if r['history_id'] == h['history_id']}
        assert cells == data.required_cells()
    assert data.validate(histories, rows, 'development')


def test_declared_history_allocation_is_balanced_and_factorially_crossed():
    from collections import Counter
    confirmation, _ = data.generate('confirmatory')
    counts = Counter((h['allocation_cell']['entity_pair'], h['attribute'], h['orientation']) for h in confirmation)
    assert set(counts.values()) == {2}
    assert set(Counter(h['allocation_cell']['entity_pair'] for h in confirmation).values()) == {16}
    for stage in ('development', 'frozen_gate'):
        histories, _ = data.generate(stage)
        assert set(Counter(h['allocation_cell']['entity_pair'] for h in histories).values()) == {4}
        assert set(Counter(h['attribute'] for h in histories).values()) == {6}
        assert set(Counter(h['orientation'] for h in histories).values()) == {12}
        pair_attr = Counter((h['allocation_cell']['entity_pair'], h['attribute']) for h in histories)
        assert set(pair_attr.values()) == {1}
        relation_attr = Counter((h['other_attribute'], h['attribute']) for h in histories)
        assert set(relation_attr.values()) == {3}
    relation_full_cells = Counter((h['other_attribute'], h['attribute']) for h in confirmation)
    assert set(relation_full_cells.values()) == {12}
    assert all(h['other_attribute'] in data.ALT_RELATIONS and h['other_attribute'] != h['attribute']
               for h in confirmation)


def test_allocation_counterexample_is_rejected(development):
    histories, rows = development
    broken = copy.deepcopy(histories)
    broken_rows = copy.deepcopy(rows)
    broken[0]['allocation_cell']['attribute'] = 3
    for row in broken_rows:
        if row['history_id'] == broken[0]['history_id']:
            row['allocation_cell']['attribute'] = 3
    with pytest.raises(ValueError, match='allocation table'):
        data.audit_structure(broken, broken_rows)


def test_rendered_pair_changes_only_the_target_value_span(development):
    _, rows = development
    by_pair = defaultdict(dict)
    for row in rows:
        by_pair[row['pair_id']][row['pair_direction']] = row
    for pair in by_pair.values():
        baseline, edited = pair[0], pair[1]
        body0, spans0 = data.render_body(baseline)
        body1, spans1 = data.render_body(edited)
        field = baseline['edited_field']
        start0, end0 = spans0[field]
        start1, end1 = spans1[field]
        assert body0[:start0] == body1[:start1]
        assert body0[end0:] == body1[end1:]
        assert body0[start0:end0] == baseline['source_value']
        assert body1[start1:end1] == baseline['replacement_value']


@pytest.mark.parametrize('corruption', ['remove_z', 'remove_member', 'duplicate_id', 'duplicate_pair',
                                      'answer', 'unintended_edit', 'current_value', 'order', 'seed'])
def test_structural_failures_rejected(development, corruption):
    histories, original = development
    rows = copy.deepcopy(original)
    if corruption == 'remove_z':
        rows = [r for r in rows if r['edited_variable'] != 'z']
    elif corruption == 'remove_member':
        rows.pop()
    elif corruption == 'duplicate_id':
        rows[-1]['example_id'] = rows[0]['example_id']
    elif corruption == 'duplicate_pair':
        rows[-1] = copy.deepcopy(rows[0])
    elif corruption == 'answer':
        rows[0]['answer'] = rows[0]['replacement_value']
    elif corruption == 'unintended_edit':
        rows[1]['semantic_values']['initial_z'] = rows[1]['replacement_value']
    elif corruption == 'current_value':
        rows[0]['semantic_values']['proposed_x'] = rows[0]['source_value']
    elif corruption == 'order':
        rows[0]['historical_entity_order'] = 1-rows[0]['historical_entity_order']
    else:
        rows[0]['seed'] = 1
    with pytest.raises(ValueError):
        data.validate(histories, rows, 'development')


def test_six_distinct_values_and_live_answer_semantics(development):
    histories, rows = development
    for h in histories:
        assigned, replacements = set(h['matching_values'].values()), set(h['replacement_values'].values())
        assert len(assigned) == 4 and len(replacements) == 2 and not assigned & replacements
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r['pair_id']][r['pair_direction']] = r
    for pair in pairs.values():
        b, e = pair[0], pair[1]
        field = b['edited_field']
        assert b['semantic_values'][field] == b['source_value']
        assert e['semantic_values'][field] == b['replacement_value']
        assert all(b['semantic_values'][k] == e['semantic_values'][k] for k in b['semantic_values'] if k != field)
        if b['condition'] == 'live' and b['query'] == b['edited_variable']:
            assert e['answer'] == b['replacement_value'] and e['answer'] != b['answer']
        else:
            assert b['answer'] == e['answer']
        for r in (b, e):
            assert r['answer'] == r['semantic_values'][r['current_fields'][r['query']]]
        if b['condition'] != 'live':
            assert b['answer'] not in {b['source_value'], b['replacement_value']}


def test_actual_order_entity_mapping_and_control_semantics(development):
    _, rows = development
    selected = [r for r in rows if r['history_index'] < 2 and r['query'] == r['edited_variable'] == 'x' and r['pair_direction'] == 0]
    for r in selected:
        body, spans = data.render_body(r)
        assert body[spans['queried_entity'][0]:spans['queried_entity'][1]] == r['query_entity'] == r['variables'][0]
        first_h = 'x' if r['historical_entity_order'] == 0 else 'z'
        assert spans[f'initial_{first_h}'][0] < spans[f'initial_{"z" if first_h == "x" else "x"}'][0]
        if r['condition'] != 'live':
            first_c = 'x' if r['current_entity_order'] == 0 else 'z'
            assert spans[f'proposed_{first_c}'][0] < spans[f'proposed_{"z" if first_c == "x" else "x"}'][0]
        if r['condition'] == 'entity_mention':
            assert f'{r["variables"][0]} mentioned {r["semantic_values"]["initial_x"]} in an unrelated note.' in body
        elif r['condition'] == 'other_attribute':
            assert f'Previously, {r["variables"][0]}’s {r["other_attribute"]} was' in body
            assert r['other_attribute'] != r['attribute']
        elif r['condition'] == 'early_unassigned':
            assert spans['initial_x'][0] < spans['proposed_x'][0]
        elif r['condition'] == 'late_unassigned':
            assert spans['initial_x'][0] > spans['proposed_x'][0]
        elif r['condition'] == 'live':
            assert 'Currently,' in body and 'remains' not in body
        assert body.endswith('Respond with only the value, with no explanation.')


def test_determinism_and_semantic_disjointness_against_prior_files(tmp_path, development):
    histories, rows = development
    assert data.generate('development') == development
    seen = {data.concrete_signature(r) for r in rows}
    for stage in ('frozen_gate', 'confirmatory'):
        h, r = data.generate(stage)
        assert len(h) == data.COUNTS[stage]
        signatures = {data.concrete_signature(x) for x in r}
        assert not seen & signatures
        seen |= signatures
    prior = copy.deepcopy(rows)
    for r in prior:
        r['history_id'] = 'different IDs do not create new semantic histories'
        r['replacement_values'] = {'initial_x': 'coral', 'initial_z': 'amber'}
    path = tmp_path/'earlier.jsonl'
    write_jsonl(prior, path)
    with pytest.raises(ValueError, match='semantic history overlap'):
        data.generate('development', [path])
    # Express the same physical bindings with the opposite analytic orientation.
    one = histories[0]
    opposite = copy.deepcopy(one)
    opposite['variables'] = one['variables'][::-1]
    opposite['matching_values'] = {f'{prefix}_{v}': one['matching_values'][f'{prefix}_{"z" if v == "x" else "x"}']
                                   for prefix in ('initial', 'proposed') for v in ('x', 'z')}
    assert data.concrete_signature(one) == data.concrete_signature(opposite)


class CharTokenizer:
    chat_template = 'exact toy wrapper'
    class Backend:
        def to_str(self): return 'char-offsets-v1'
    backend_tokenizer = Backend()
    def __call__(self, text, **kwargs):
        result = {'input_ids': [ord(c) for c in text]}
        if kwargs.get('return_offsets_mapping'):
            result['offset_mapping'] = [(i, i+1) for i in range(len(text))]
        return result
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False and kwargs['add_generation_prompt'] is True
        return '<user>\n'+messages[0]['content']+'\n</user>\n<assistant>\n'


def test_token_offset_and_shared_surface_edit_audit(development):
    _, rows = development
    rows = [r for r in rows if r['history_index'] == 0]
    tok = CharTokenizer()
    result = validate_tokens(tok, rows, renderer=data.render, positions=data.semantic_positions)
    assert result['edit_audit']['pairs_checked'] == 96
    for r in rows:
        entry = data.span_audit(r, tok)
        for field, value in r['semantic_values'].items():
            span = entry['spans'][field]
            assert span['text'] == value
            assert span['token_length'] == len(value)
            assert span['token_end_exclusive']-span['token_start'] == len(value)
        assert entry['spans']['queried_entity']['text'] == r['query_entity']


def test_nuisance_order_aggregation_and_predefined_contrasts(development):
    _, all_rows = development
    rows = [r for r in all_rows if r['history_index'] == 0]
    scores = synthetic_scores(rows)
    # Controls have source/replacement independent of the answer. Impose known
    # effects; an additive order nuisance cancels in query-specific R.
    amounts = {'superseded': 4., 'early_unassigned': 1., 'entity_mention': 2.,
               'other_attribute': 3., 'late_unassigned': -1.}
    for s in scores:
        if s['condition'] == 'live': continue
        h, c = s['historical_entity_order'], s['current_entity_order']
        effect = 2*h-c + (amounts[s['condition']] if s['query'] == s['edited_variable'] else 0)
        if s['pair_direction']:
            s['semantic_log_mass'][s['replacement_value']] = -10. + effect
        s['semantic_rank'] = strict_rank(s['semantic_log_mass'], s['answer'])
        s['semantic_accuracy'] = int(s['semantic_rank'] == 1)
    out = contrasts(rows, scores)[0]
    assert out['R_superseded'] == pytest.approx(4.)
    assert out['R_superseded_minus_R_early_unassigned'] == pytest.approx(3.)
    assert out['R_superseded_minus_R_entity_mention'] == pytest.approx(2.)
    assert out['R_superseded_minus_R_other_attribute'] == pytest.approx(1.)
    assert out['R_superseded_minus_R_late_unassigned'] == pytest.approx(5.)
    assert out['R_superseded_aligned_minus_reversed'] == pytest.approx(0.)
    with pytest.raises(ValueError):
        contrasts(rows, scores[:-1])


def test_gate_unique_condition_accuracy_and_failed_development_not_a_veto(development):
    _, rows = development
    scores = synthetic_scores(rows, 'competence_only')
    assert design.evaluate(rows, scores)['pass']
    for s in scores:
        if s['condition'] == 'entity_mention':
            s['semantic_log_mass'] = {v: -1. for v in data.VALUES}
            s['semantic_rank'] = 8
            s['semantic_accuracy'] = 0
    failed = design.evaluate(rows, scores)
    assert not failed['pass'] and failed['failed_conditions'] == [{'condition': 'entity_mention', 'reasons': ['semantic_accuracy']}]
    descriptive = design.evaluate(rows, scores, gate=False)
    assert descriptive['pass'] is None and descriptive['failed_conditions'] == []
    assert 'sequence_mass' not in descriptive


def test_config_continuity_including_actual_gemma_provenance():
    for slug in design.MODEL_SETTINGS:
        config = load_config(f'configs/cross_model_relational_v2_factorial_final/{slug}.yaml')
        design.validate_config(config)
        lineage = design.verify_v1_lineage(config)
        assert lineage['scientific_settings']['model_revision'] == config['model']['revision']


def test_history_bootstrap_matches_existing_seeded_v1_algorithm(monkeypatch):
    import src.analysis.metrics as metrics
    monkeypatch.setattr(metrics, 'tqdm', lambda iterable, **kwargs: iterable)
    values = [1., 4., -2., 7.]
    reference = metrics.bootstrap_mean_ci(values, seed=73021)
    assert summary(values)['ci95_cluster_bootstrap'] == pytest.approx([reference['ci_low'], reference['ci_high']])
    assert not normalized([5., 5.], [.5, .5])['available']
