import copy
import json
import runpy
import sys

import pytest

from src.analysis.natural_competence import GATE, GATE_VERSION, evaluate_competence, expected_cells
from src.data.io import write_jsonl, sha256_file
from scripts.generate_supersession_experiments import _recompute_gate_pass


def records():
    dataset, scores = [], []
    for i, key in enumerate(sorted(expected_cells(['nora_v1']))):
        template, condition, query, orientation, variable, status, direction, slot = key
        row = dict(example_id=str(i), prompt_variant=template, condition=condition, query=query,
                   orientation=orientation, edited_variable=variable, edit_status=status,
                   pair_direction=direction, unassigned_slot_order=None if slot=='none' else slot,
                   answer='navy', stale_value='rust' if condition=='superseded' else None, seed=20261024,
                   matching_values={'initial_x':'rust','initial_z':'silver','proposed_x':'navy','proposed_z':'olive'})
        # Baselines have the same prompt across edited-variable pairs.
        prompt = f'{condition}|{query}|{orientation}|{direction}|{slot}' + (f'|{variable}' if direction else '')
        dataset.append(row)
        scores.append({**row, 'prompt':prompt, 'full_vocab_next_token_accuracy':1,
                       'accuracy':1, 'candidate_rank':1, 'candidate_logits':{'navy':4., 'rust':1.}})
    return dataset, scores


def test_duplicate_baselines_cover_both_variables_and_are_order_invariant():
    dataset, scores = records()
    result = evaluate_competence(dataset, scores)
    assert result['expected_cell_count'] == len(result['summary']) == 64
    assert result['n_unique_prompts'] == 48
    assert result == evaluate_competence(dataset[::-1], scores[::-1])
    duplicate = copy.deepcopy(dataset[0]); duplicate['example_id']='duplicate'
    duplicate_score = {**scores[0], 'example_id':'duplicate'}
    assert result == evaluate_competence(dataset+[duplicate], scores+[duplicate_score])


def test_missing_baseline_subcells_fail_even_with_all_conditions_present():
    dataset, scores = records()
    retained = [r for r in dataset if not (r['condition']=='superseded' and r['edited_variable']=='x' and r['pair_direction']==0)]
    with pytest.raises(ValueError, match='incomplete competence cell set'):
        evaluate_competence(retained, [s for s in scores if s['example_id'] in {r['example_id'] for r in retained}])


def test_gate_verifier_uses_same_cell_contract(tmp_path):
    dataset, scores=records()
    d=tmp_path/'data.jsonl'; s=tmp_path/'scores.jsonl'
    write_jsonl(dataset,d); write_jsonl(scores,s)
    result=evaluate_competence(dataset,scores)
    gate={'dataset_path':str(d),'behavior_path':str(s),'gate_version':GATE_VERSION,'preregistered_gate':GATE,
          'competence_code_sha256':sha256_file('src/analysis/natural_competence.py'),
          'dataset_seed':result['dataset_seed'],'expected_cell_count':result['expected_cell_count'],
          'by_template_condition_query_orientation_edit_status_pair_direction_slot':result['summary']}
    assert _recompute_gate_pass(gate)
    gate['dataset_seed']=[20261023]
    assert not _recompute_gate_pass(gate)
    gate.pop('gate_version')
    assert not _recompute_gate_pass(gate)


def test_artifact_seed_comes_from_dataset_not_config(tmp_path, monkeypatch):
    dataset, scores=records()
    d=tmp_path/'data.jsonl'; s=tmp_path/'scores.jsonl'; out=tmp_path/'analysis.json'
    write_jsonl(dataset,d); write_jsonl(scores,s)
    (tmp_path/'scores.jsonl.provenance.json').write_text(json.dumps(
        {'dataset_sha256':sha256_file(d),'dataset_seed':[20261024],'seed':20261023}))
    config=tmp_path/'config.yaml'; config.write_text('seed: 20261023\n')
    tokens=tmp_path/'tokens.json'; tokens.write_text('{}')
    monkeypatch.setattr(sys,'argv',['analyze','--stage','development','--dataset',str(d),'--behavior',str(s),
                                  '--config',str(config),'--token-ids',str(tokens),'--output',str(out)])
    runpy.run_path('scripts/analyze_natural_competence.py',run_name='__main__')
    artifact=json.loads(out.read_text())
    assert artifact['seed']==20261024
    assert artifact['dataset_seed']==[20261024]
    assert artifact['scoring_config_seed']==20261023
