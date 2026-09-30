import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.io import read_jsonl, write_jsonl
from src.data.progress import prepare_jsonl_progress
from src.data.supersession_behavior import (generate_behavior_pairs, audit_behavior_dataset,
                                          audit_candidate_tokens, audit_tokenized_pairs, render_behavior_example)
from src.analysis.supersession_behavior import (audited_effects, matched_edit_effect,
                                               history_contrasts, status_2x2_contrasts, competence)
from scripts.audit_supersession_status import categorize_no_errors, all_no_members_correct_histories

VALUES = ['amber', 'coral', 'denim', 'elm', 'frost', 'grape', 'jade', 'maple', 'navy', 'pearl', 'quartz', 'rust']
IDS = {v: i+1 for i, v in enumerate(VALUES)}


class WordTokenizer:
    chat_template = None
    def __init__(self):
        self.vocab = dict(IDS)
    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = []
        for word in re.findall(r'\w+|[^\w\s]', text):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)+1
            ids.append(self.vocab[word])
        if return_tensors:
            import torch
            return {'input_ids': torch.tensor([ids])}
        return {'input_ids': ids}
    def decode(self, ids):
        return ','.join(str(i) for i in ids)


def scoring_stub():
    import torch
    class ScoringStub:
        def __init__(self):
            self.calls = 0
        def parameters(self):
            return iter([torch.zeros(1)])
        def __call__(self, input_ids, use_cache=False):
            self.calls += 1
            # Every answer is wrong in full vocabulary; primary analyses must still retain it.
            logits = torch.arange(256).float().expand(input_ids.shape[0], input_ids.shape[1], 256)
            return SimpleNamespace(logits=logits)
    return ScoringStub()


def scaffold(rows, relevance=None):
    relevance = relevance or {}
    output = []
    for r in rows:
        score = copy.deepcopy(r)
        logits = {v: 0. for v in VALUES}
        kind, cond, field, variable, query = (r[k] for k in ('experiment_kind', 'condition', 'edited_field', 'edited_variable', 'query'))
        # Independent fixed effects per history/semantic cell; edited member only.
        target = relevance.get((r['history_id'], cond, field), relevance.get((cond, field), 0.))
        if r['pair_direction']:
            logits[r['replacement_value']] = 1.+target if variable == query else 1.
        score.update(candidate_logits=logits, candidate_probabilities={v: 1/len(VALUES) for v in VALUES},
                     full_vocab_rank=40, candidate_rank=2, accuracy=0, full_vocab_next_token_accuracy=0,
                     prompt='opaque prompt deliberately has no assignments, statuses, or query text')
        output.append(score)
    return output


def test_generated_cells_semantics_matching_and_single_value_edits():
    for kind, count in (('controls', 24), ('status', 32)):
        rows = generate_behavior_pairs(kind, 2, VALUES, 7)
        assert len(rows) == count*2
        assert audit_behavior_dataset(rows, kind) == kind
        for base, edit in zip(rows[::2], rows[1::2]):
            changed = [f for f in base['semantic_values'] if base['semantic_values'][f] != edit['semantic_values'][f]]
            assert changed == [base['edited_field']]
            assert base['history_id'] == edit['history_id']
            assert base['replacement_value'] not in base['matching_values'].values()
        audit_tokenized_pairs(rows, WordTokenizer(), IDS, chat=False)
    status = generate_behavior_pairs('status', 1, VALUES, 7)
    yes = next(r for r in status if r['condition'] == 'accepted' and r['pair_direction'] == 0)
    no = next(r for r in status if r['condition'] == 'rejected' and r['edited_field'] == yes['edited_field'] and r['query'] == yes['query'] and r['pair_direction'] == 0)
    assert render_behavior_example(yes).replace('YES', 'STATUS') == render_behavior_example(no).replace('NO', 'STATUS')
    assert yes['replacement_value'] == no['replacement_value']
    assert yes['answer'] == yes['semantic_values']['proposed_x']
    assert no['answer'] == no['semantic_values']['initial_x']


def test_identity_transfer_components_and_fixed_correct_stale_orientation():
    rows = generate_behavior_pairs('controls', 1, VALUES, 7)
    scored = scaffold(rows)
    base = next(r for r in scored if r['condition'] == 'superseded' and r['query'] == 'x' and r['edited_variable'] == 'x' and r['pair_direction'] == 0)
    edit = next(r for r in scored if r['pair_id'] == base['pair_id'] and r['pair_direction'] == 1)
    s, r, c = base['source_value'], base['replacement_value'], base['answer']
    base['candidate_logits'].update({s: 4., r: 1., c: 9.})
    edit['candidate_logits'].update({s: 1., r: 7., c: 8.})
    e = matched_edit_effect(base, edit)
    assert e['identity_transfer'] == pytest.approx(9.)
    assert e['source_logit_change'] == -3 and e['replacement_logit_change'] == 6
    assert e['correct_answer_logit_change'] == -1
    assert e['correct_minus_stale_before'] == 5 and e['correct_minus_stale_after'] == 1
    assert e['stale_value_before'] == s and e['stale_value_after'] == r and e['correct_minus_stale_change'] == -4
    assert e['fixed_baseline_correct_minus_stale_change'] == 2
    assert matched_edit_effect(scored[0], scored[1])['correct_minus_stale_change'] is None


def test_controls_are_symmetric_and_condition_differences_are_history_paired():
    rows = generate_behavior_pairs('controls', 2, VALUES, 7)
    values = {('live', 'initial_x'): 6., ('live', 'initial_z'): 8.,
              ('superseded', 'initial_x'): 2., ('superseded', 'initial_z'): 4.,
              ('irrelevant', 'initial_x'): 1., ('irrelevant', 'initial_z'): -1.,
              ('controls000001', 'live', 'initial_x'): -6., ('controls000001', 'live', 'initial_z'): 2.}
    scored = scaffold(rows, values)
    effects = audited_effects(rows, scored, 'controls')
    histories = history_contrasts(effects, 'controls')
    a, b = histories
    assert a['R_live_x'] == 6 and a['R_live_z'] == 8 and a['R_live'] == 7
    assert a['R_superseded'] == 3 and a['R_irrelevant'] == 0
    assert a['R_live_minus_R_superseded'] == 4
    assert a['R_superseded_minus_R_irrelevant'] == 3
    assert b['R_live'] == -2 and b['R_live_minus_R_superseded'] == -5
    assert len(effects) == 24 and len(histories) == 2
    assert all(r['pair_both_full_vocab_next_token_correct'] == 0 for r in effects)
    diagnostics = competence(scored, effects)
    assert all(r['low_competence'] and r['both_answers_correct_count'] == 0 for r in diagnostics)
    # This analysis succeeds even though prompts carry no usable condition strings and all answers are wrong.


def test_status_keeps_yes_no_and_same_occurrence_comparisons_paired():
    rows = generate_behavior_pairs('status', 1, VALUES, 7)
    values = {('accepted', 'proposed_x'): 6., ('accepted', 'proposed_z'): 4.,
              ('accepted', 'initial_x'): 1., ('accepted', 'initial_z'): 3.,
              ('rejected', 'proposed_x'): 0., ('rejected', 'proposed_z'): 2.,
              ('rejected', 'initial_x'): 8., ('rejected', 'initial_z'): 10.}
    effects = audited_effects(rows, scaffold(rows, values), 'status')
    h = history_contrasts(effects, 'status')[0]
    assert h['R_accepted_current'] == 5 and h['R_rejected_update'] == 1
    assert h['R_superseded_initial'] == 2 and h['R_retained_initial_after_rejection'] == 9
    assert h['R_accepted_current_minus_R_rejected_update'] == 4
    assert h['R_superseded_initial_minus_R_retained_initial_after_rejection'] == -7


def test_counterbalanced_irrelevant_slots_are_independent_and_averaged():
    rows = generate_behavior_pairs('controls_counterbalanced', 1, VALUES, 31)
    assert audit_behavior_dataset(rows, 'controls_counterbalanced') == 'controls_counterbalanced'
    bases = [r for r in rows if r['condition'] == 'irrelevant_counterbalanced' and r['pair_direction'] == 0]
    assert {(r['edited_field'], r['query'], r['unassigned_slot_order']) for r in bases} == {
        (f, q, slot) for f in ('initial_x','initial_z') for q in ('x','z') for slot in ('xz','zx')}
    xz = next(r for r in bases if r['edited_field']=='initial_x' and r['query']=='x' and r['unassigned_slot_order']=='xz')
    zx = next(r for r in bases if r['edited_field']=='initial_x' and r['query']=='x' and r['unassigned_slot_order']=='zx')
    assert xz['semantic_values']==zx['semantic_values']
    lines_xz=render_behavior_example(xz,chat=False).splitlines()
    lines_zx=render_behavior_example(zx,chat=False).splitlines()
    assert lines_xz[2:] == list(reversed(lines_zx[2:4])) + lines_zx[4:]
    effects=audited_effects(rows,scaffold(rows),'controls_counterbalanced')
    for e in effects:
        if e['condition']=='irrelevant_counterbalanced':
            e['identity_transfer'] = {('initial_x','x','xz'):10,('initial_x','x','zx'):0,
                                      ('initial_x','z','xz'):0,('initial_x','z','zx'):2,
                                      ('initial_z','z','xz'):8,('initial_z','z','zx'):4,
                                      ('initial_z','x','xz'):0,('initial_z','x','zx'):2}[(e['edited_field'],e['query'],e['unassigned_slot_order'])]
    h=history_contrasts(effects,'controls_counterbalanced')[0]
    assert h['R_irrelevant_counterbalanced_x']==4
    assert h['R_irrelevant_counterbalanced_z']==5


def test_status_2x2_local_acceptance_effects_are_history_paired():
    rows=generate_behavior_pairs('status_2x2',2,VALUES,41)
    assert audit_behavior_dataset(rows,'status_2x2')=='status_2x2'
    yes=next(r for r in rows if r['condition']=='YY' and r['pair_direction']==0 and r['query']=='x')
    mixed=next(r for r in rows if r['condition']=='NY' and r['edited_field']=='proposed_x' and r['pair_direction']==0 and r['query']=='x')
    assert yes['answer']==yes['semantic_values']['proposed_x']
    assert mixed['answer']==mixed['semantic_values']['initial_x']
    assert yes['semantic_values']==mixed['semantic_values']
    a=render_behavior_example(yes,chat=False).splitlines()
    b=render_behavior_example(mixed,chat=False).splitlines()
    assert [(i,x,y) for i,(x,y) in enumerate(zip(a,b)) if x!=y] == [(3,a[3],b[3])]
    effects=audited_effects(rows,scaffold(rows),'status_2x2')
    contrasts={('YY','x','x'):10,('YY','x','z'):1,('NY','x','x'):4,('NY','x','z'):1,
               ('YN','x','x'):8,('YN','x','z'):0,('NN','x','x'):3,('NN','x','z'):0,
               ('YY','z','z'):9,('YY','z','x'):1,('YN','z','z'):4,('YN','z','x'):1,
               ('NY','z','z'):7,('NY','z','x'):0,('NN','z','z'):2,('NN','z','x'):0}
    for e in effects: e['identity_transfer']=contrasts[(e['condition'],e['edited_variable'],e['query'])]
    h=status_2x2_contrasts(effects)[0]
    assert h['R_x_local_1']==6 and h['R_x_local_2']==5 and h['R_x_acceptance_effect']==5.5
    assert h['R_z_local_1']==5 and h['R_z_local_2']==5 and h['R_z_acceptance_effect']==5
    assert h['R_acceptance_effect_symmetric']==5.25


@pytest.mark.parametrize(('predicted','expected'), [
    ('proposed','rejected_proposed_value'),('initial','retained_initial_value'),
    ('other','other_candidate'),('outside','non_candidate')])
def test_no_condition_error_categories_and_complete_history_sensitivity(predicted,expected):
    rows=generate_behavior_pairs('status',1,VALUES,9)
    scores=scaffold(rows)
    no=[r for r in scores if r['condition']=='rejected']
    for r in no: r['full_vocab_next_token_accuracy']=1
    target=no[0]; source=next(r for r in rows if r['example_id']==target['example_id'])
    query=source['query']
    value={'proposed':source['semantic_values'][f'proposed_{query}'],
           'initial':source['semantic_values'][f'initial_{query}'],
           'other':next(v for v in VALUES if v not in source['semantic_values'].values()),
           'outside':None}[predicted]
    target['full_vocab_next_token_accuracy']=0
    target['greedy_token_id']=IDS[value] if value is not None else 999999
    classified=categorize_no_errors(rows,scores,IDS)
    assert len(classified)==1 and classified[0]['error_category']==expected
    assert all_no_members_correct_histories(scores)==set()
    target['full_vocab_next_token_accuracy']=1
    assert all_no_members_correct_histories(scores)=={source['history_id']}


@pytest.mark.parametrize('mutation', ['direction_missing', 'direction_ambiguous', 'query_cell_missing', 'extra_change', 'missing_status', 'collision', 'mixed', 'duplicate'])
def test_dataset_audits_fail_closed(mutation):
    rows = generate_behavior_pairs('controls', 1, VALUES, 7)
    if mutation == 'direction_missing': rows.pop(0)
    if mutation == 'query_cell_missing': rows = rows[2:]
    if mutation == 'direction_ambiguous': rows[1]['pair_direction'] = True
    if mutation == 'extra_change': rows[1]['variables'] = ['p', 'q']
    if mutation == 'missing_status': del rows[0]['semantic_status']
    if mutation == 'collision': rows[0]['replacement_values']['initial_x'] = rows[0]['source_value']
    if mutation == 'mixed': rows.extend(generate_behavior_pairs('status', 1, VALUES, 7))
    if mutation == 'duplicate': rows.append(rows[0])
    with pytest.raises(ValueError): audit_behavior_dataset(rows)


def test_missing_cells_missing_score_and_changed_scored_metadata_fail():
    rows = generate_behavior_pairs('controls', 1, VALUES, 7)
    effects = audited_effects(rows, scaffold(rows), 'controls')
    with pytest.raises(ValueError, match='missing.*cells'): history_contrasts(effects[:-1], 'controls')
    with pytest.raises(ValueError, match='incomplete'): audited_effects(rows, scaffold(rows)[:-1], 'controls')
    scores = scaffold(rows); scores[0]['edit_status'] = 'invented'
    with pytest.raises(ValueError, match='metadata differs'): audited_effects(rows, scores, 'controls')
    with pytest.raises(ValueError, match='regenerate'): audit_behavior_dataset([{'history_id': 'old', 'condition': 'live'}])


@pytest.mark.parametrize('label', ['source_value', 'replacement_value', 'answer'])
def test_missing_token_map_value_is_explicit(label):
    rows = generate_behavior_pairs('controls', 1, VALUES, 7)
    mapping = IDS.copy(); mapping.pop(rows[0][label])
    with pytest.raises(ValueError, match='missing from validated token map'): audit_candidate_tokens(rows[:1], mapping)


def test_unstable_candidate_continuation_and_prompt_alignment_fail():
    rows = generate_behavior_pairs('controls', 1, VALUES, 7)
    class Unstable(WordTokenizer):
        def __call__(self, text, **kwargs):
            encoded = super().__call__(text, **kwargs)
            if text.endswith(' amber'): encoded['input_ids'].append(255)
            return encoded
    with pytest.raises(ValueError, match='one continuation token'):
        audit_tokenized_pairs(rows, Unstable(), IDS, chat=False)


def test_current_edit_diagnostics_distinguish_fixed_tokens_from_changed_answers():
    rows = generate_behavior_pairs('status', 1, VALUES, 7)
    scores = scaffold(rows)
    base = next(r for r in scores if r['condition'] == 'accepted' and r['edited_field'] == 'proposed_x' and r['query'] == 'x' and r['pair_direction'] == 0)
    edit = next(r for r in scores if r['pair_id'] == base['pair_id'] and r['pair_direction'] == 1)
    source, replacement, old = base['source_value'], base['replacement_value'], base['stale_value']
    base['candidate_logits'].update({source: 4., replacement: 0., old: 1.})
    edit['candidate_logits'].update({source: 2., replacement: 9., old: 1.})
    effect = matched_edit_effect(base, edit)
    assert effect['correct_answer_value_used'] == source and effect['edited_answer_value'] == replacement
    assert effect['correct_answer_logit_change'] == -2 and effect['semantic_target_logit_change'] == 5
    assert effect['correct_minus_stale_change'] == 5 and effect['fixed_baseline_correct_minus_stale_change'] == -2


def test_unmapped_inputs_fail_before_the_model_loader_is_called(tmp_path, monkeypatch):
    import scripts.run_supersession_behavior as runner
    rows = generate_behavior_pairs('controls', 1, VALUES, 7)
    dataset, tokens, config = (tmp_path/name for name in ('data.jsonl', 'tokens.json', 'config.yaml'))
    write_jsonl(rows, dataset)
    ids = IDS.copy(); ids.pop(rows[0]['replacement_value'])
    tokens.write_text(json.dumps({'token_ids': ids})); config.write_text('model:\n  id: unused\n')
    def forbidden_loader(_config):
        pytest.fail('invalid input must be rejected before any model loading')
    monkeypatch.setattr(runner, 'load_model', forbidden_loader)
    monkeypatch.setattr(sys, 'argv', ['score', '--config', str(config), '--dataset', str(dataset), '--token-ids', str(tokens), '--output', str(tmp_path/'scores.jsonl')])
    with pytest.raises(ValueError, match='replacement_value.*missing'):
        runner.main()


def test_scoring_reuses_shared_scorer_preserves_metadata_and_resumes_partial_pair(tmp_path, monkeypatch):
    import scripts.run_supersession_behavior as runner
    rows = generate_behavior_pairs('controls', 1, VALUES, 7)
    dataset, tokens, output = tmp_path/'data.jsonl', tmp_path/'tokens.json', tmp_path/'scores.jsonl'
    write_jsonl(rows, dataset); tokens.write_text(json.dumps({'token_ids': IDS}))
    config = {'model': {'id': 'toy', 'revision': 'fixed'}}
    completed = prepare_jsonl_progress(output, dataset, tokens, config, rows)
    model, tok = scoring_stub(), WordTokenizer()
    shared = runner.score_example
    counter = [0]
    def interrupted(*args, **kwargs):
        counter[0] += 1
        if counter[0] == 2: raise RuntimeError('simulated interruption')
        return shared(*args, **kwargs)
    monkeypatch.setattr(runner, 'score_example', interrupted)
    with pytest.raises(RuntimeError, match='interruption'):
        runner.score_dataset(model, tok, rows, IDS, output, completed, chat=False)
    assert len(read_jsonl(output)) == 1
    with output.open('ab') as f: f.write(b'{"example_id":"partial')
    completed = prepare_jsonl_progress(output, dataset, tokens, config, rows, resume=True)
    assert len(completed) == 1
    monkeypatch.setattr(runner, 'score_example', shared)
    runner.score_dataset(model, tok, rows, IDS, output, completed, chat=False)
    scores = read_jsonl(output)
    for score in scores:
        ex = next(r for r in rows if r['example_id'] == score['example_id'])
        assert all(score[k] == v for k, v in ex.items())
        assert score['prompt'] == render_behavior_example(ex)
        assert 'candidate_logits' in score and 'candidate_probabilities' in score
        if score['pair_direction'] == 1: assert 'matched_edit_effect' in score
    assert len(audited_effects(rows, scores, 'controls')) == 12
    calls = model.calls
    completed = prepare_jsonl_progress(output, dataset, tokens, config, rows, resume=True)
    runner.score_dataset(model, tok, rows, IDS, output, completed, chat=False)
    assert model.calls == calls
    with pytest.raises(FileExistsError): prepare_jsonl_progress(output, dataset, tokens, config, rows)
    with pytest.raises(ValueError, match='fingerprint'):
        prepare_jsonl_progress(output, dataset, tokens, {**config, 'resolved_model_revision': 'changed'}, rows, resume=True)
    tokens.write_text(json.dumps({'token_ids': {**IDS, 'extra': 200}}))
    with pytest.raises(ValueError, match='token_ids_sha256'):
        prepare_jsonl_progress(output, dataset, tokens, config, rows, resume=True)


@pytest.mark.parametrize('kind', ['controls', 'status', 'controls_counterbalanced', 'status_2x2'])
def test_generate_score_analyze_commands_end_to_end_with_scoring_stub(tmp_path, monkeypatch, kind):
    import scripts.generate_supersession_experiments as generator
    import scripts.run_supersession_behavior as runner
    import scripts.analyze_supersession_behavior as analyzer
    values, token_map, dataset, output, config = (tmp_path/name for name in ('values.json', 'tokens.json', 'data.jsonl', 'behavior.jsonl', 'config.yaml'))
    values.write_text(json.dumps(VALUES+['unvalidated']))
    token_map.write_text(json.dumps({'token_ids': IDS, 'model_revision': 'a'*40, 'tokenizer_revision': 'a'*40}))
    config.write_text('seed: 7\nmodel:\n  id: toy\n  revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n  tokenizer_revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n  chat_template: false\n')
    monkeypatch.setattr(sys, 'argv', ['generate', '--kind', kind, '--values', str(values), '--token-ids', str(token_map), '--n', '2', '--output', str(dataset)])
    generator.main()
    rows = read_jsonl(dataset)
    assert all('unvalidated' not in r['candidate_values'] for r in rows)
    model = scoring_stub()
    def load_model(c):
        c['resolved_model_revision'] = 'a'*40; c['resolved_tokenizer_revision'] = 'a'*40
        return model, WordTokenizer()
    monkeypatch.setattr(runner, 'load_model', load_model)
    args = ['score', '--config', str(config), '--dataset', str(dataset), '--token-ids', str(token_map), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', args); runner.main()
    calls = model.calls
    monkeypatch.setattr(sys, 'argv', args+['--resume']); runner.main()
    assert model.calls == calls
    sidecar = json.loads(Path(str(output)+'.provenance.json').read_text())
    for k in ('dataset_sha256', 'config_sha256', 'token_map_sha256', 'model_revision', 'tokenizer_revision', 'git_commit', 'packages', 'dataset_seed'):
        assert k in sidecar
    analysis = tmp_path/'analysis'
    monkeypatch.setattr(sys, 'argv', ['analyze', '--kind', kind, '--dataset', str(dataset), '--behavior', str(output), '--output-dir', str(analysis), '--bootstrap-draws', '20'])
    analyzer.main()
    result = json.loads((analysis/'summary.json').read_text())
    assert result['n_histories'] == 2 and result['primary_filter'] == 'all_valid_trials'
    assert result['all_contrasts_history_paired'] and result['low_competence_conditions']
    assert all(s['n_histories'] == 2 and len(s['ci95_cluster_bootstrap']) == 2 for s in result['results'].values())
    assert (analysis/'matched_edit_effects.csv').exists() and (analysis/'history_relevance.csv').exists()
    with pytest.raises(FileExistsError): analyzer.main()
    config.write_text(config.read_text()+'extra: changed\n')
    monkeypatch.setattr(sys, 'argv', args+['--resume'])
    with pytest.raises(ValueError, match='fingerprint'): runner.main()
