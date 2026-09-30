import json
import pytest
from src.data.status_focal import generate,audit,gate_summary,require_passing_frozen_gate
from src.experiments.patching import validity_decision_margin,validity_patch_delta

VALUES=['amber','coral','denim','elm','frost','gray','hazel','indigo']

def test_focal_design_balances_semantic_order_xz_orientation_and_matches_validity():
 rows=generate('confirmatory',96,VALUES,231,('bracketed',));audit(rows,'confirmatory')
 histories={}
 for r in rows:histories.setdefault(r['history_id'],[]).append(r)
 assert len(histories)==96
 for cells in histories.values():
  assert {r['focal_position'] for r in cells}=={'focal_first','focal_second'}
  assert {r['focal_variable'] for r in cells}.__len__()==1
  for role in ('focal','other'):
   for pos in ('focal_first','focal_second'):
    yes=next(r for r in cells if r['query_role']==role and r['focal_position']==pos and r['focal_valid'])
    no=next(r for r in cells if r['query_role']==role and r['focal_position']==pos and not r['focal_valid'])
    assert yes['matching_values']==no['matching_values']
    assert yes['literal_names']==no['literal_names']
    assert yes['update_order']==no['update_order']
 focal={r['focal_variable'] for r in rows if r['focal_valid']}
 assert focal=={'x','z'}
 assert {tuple(r['literal_names'].items()) for r in rows}=={(('x','x'),('z','z')),(('x','z'),('z','x'))}

def test_frozen_gate_completeness_and_accuracy_only_summary():
 rows=generate('frozen_gate',24,VALUES,887,('active_words',))
 scores=[{**r,'full_vocab_next_token_accuracy':1,'target_rank':1,'candidate_accuracy':1,'candidate_target_rank':1} for r in rows]
 result=gate_summary(rows,scores)
 assert result['passed'] and result['complete_cells']
 assert len(result['cells'])==32
 assert result['causal_effects_computed'] is False

def test_confirmatory_fails_closed_without_passing_frozen_gate():
 with pytest.raises(ValueError):require_passing_frozen_gate({'stage':'frozen_gate','passed':False,'selected_variant':'bracketed','causal_effects_computed':False})
 assert require_passing_frozen_gate({'stage':'frozen_gate','passed':True,'selected_variant':'bracketed','causal_effects_computed':False})=='bracketed'

def test_validity_patch_delta_has_current_minus_old_sign():
 ids={'current':0,'old':1}
 assert validity_decision_margin([5.,2.],'current','old',ids)==3.
 assert validity_patch_delta([5.,2.],[3.,4.],'current','old',ids)=={'D_recipient':3.,'D_patched':-1.,'delta_D_patch':-4.}

def test_existing_legacy_artifacts_are_not_opened_or_overwritten(tmp_path):
 legacy=tmp_path/'status_2x2.jsonl';legacy.write_text('{"legacy":true}\n');before=legacy.read_bytes()
 focal=generate('development',2,VALUES,991,('bracketed','active_words'))
 assert audit(focal)=='development'
 assert legacy.read_bytes()==before
