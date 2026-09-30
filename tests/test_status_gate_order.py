import copy
import pytest

from src.data.supersession_behavior import (generate_behavior_pairs, generate_status_2x2_gate,
    audit_status_2x2_gate_dataset, audit_status_2x2_update_order, render_behavior_example)
from scripts.analyze_status_2x2_gate import summarize_gate
from scripts.audit_status_order_orientation import analyze_order

VALUES=['amber','coral','denim','elm','frost','grape','jade','maple','navy','pearl','quartz','rust']
IDS={v:i+1 for i,v in enumerate(VALUES)}


def test_status_2x2_gate_is_fresh_accuracy_only_and_has_balanced_update_order():
    gate=generate_status_2x2_gate(24,VALUES,99117)
    assert len(gate)==192 and audit_status_2x2_gate_dataset(gate)
    assert len({r['history_id'] for r in gate})==24 and {r['seed'] for r in gate}=={99117}
    final=generate_behavior_pairs('status_2x2',96,VALUES,99118)
    assert {r['history_id'] for r in gate}.isdisjoint({r['history_id'] for r in final})
    assert all(not any(k in r for k in ('pair_id','pair_direction','edited_field','source_value','replacement_value')) for r in gate)
    balance=audit_status_2x2_update_order(gate)
    assert all(cells=={'x':12,'z':12} for cells in balance.values())
    assert render_behavior_example(gate[0],chat=False).endswith('Answer:')


def test_competence_gate_summary_is_cellwise_and_fails_at_below_99_percent():
    dataset=generate_status_2x2_gate(24,VALUES,99118)
    scores=[{**r,'prompt':render_behavior_example(r,chat=False),'full_vocab_next_token_accuracy':1,
             'candidate_accuracy':1,'target_rank':1,'candidate_target_rank':1} for r in dataset]
    report=summarize_gate(dataset,scores)
    assert report['gate_pass'] and len(report['status_cells'])==4
    assert all(r['n_histories']==24 and r['n_scored_prompts']==48 for r in report['status_cells'])
    scores[0]['full_vocab_next_token_accuracy']=0
    report=summarize_gate(dataset,scores)
    assert not report['gate_pass'] and report['failure_cells']==['YY']
    scores[0]['identity_transfer']=3
    with pytest.raises(ValueError,match='causal/effect'):
        summarize_gate(dataset,scores)


def test_status_2x2_balances_literal_update_order_in_each_status_cell():
    rows=generate_behavior_pairs('status_2x2',24,VALUES,99119)
    balance=audit_status_2x2_update_order(rows)
    assert set(balance)=={'YY','YN','NY','NN'}
    assert all(counts=={'x':12,'z':12} for counts in balance.values())
    corrupted=copy.deepcopy(rows)
    representative=next(r for r in corrupted if r['condition']=='YY' and r['pair_direction']==0 and r['edited_field']=='initial_x' and r['query']=='x')
    representative['variables']=['z','x']
    with pytest.raises(ValueError,match='order|orientation'):
        audit_status_2x2_update_order(corrupted)


def test_existing_status_order_audit_crosses_literal_name_with_block_position():
    dataset=generate_behavior_pairs('status',2,VALUES,99120)
    scored=[]
    for r in dataset:
        logits={v:float(i) for i,v in enumerate(VALUES)}
        scored.append({**r,'candidate_logits':logits,'candidate_probabilities':{v:1/len(VALUES) for v in VALUES},
            'full_vocab_rank':1,'candidate_rank':1,'accuracy':1,'full_vocab_next_token_accuracy':1,
            'prompt':render_behavior_example(r,chat=False)})
    result=analyze_order(dataset,scored,seed=4,draws=20)
    assert len(result['per_history_components'])==2*4*2
    assert set(result['per_history_components']['literal_variable'])=={'x','z'}
    assert set(result['per_history_components']['block_position'])=={'first','second'}
    assert set(result['per_history_components']['orientation'])=={0,1}
    assert len(result['task_rows'])==2*2*2
    assert set(result['factorial_decomposition']['metric'])=={'R_accepted_current','R_superseded_initial',
        'R_retained_initial_after_rejection','R_rejected_update'}
