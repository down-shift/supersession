import json
import pytest
from src.data.io import write_jsonl,sha256_file
from src.data.status_focal import (generate,audit,gate_summary,render,audit_validity_prompt_alignment,
    focal_history_signatures,focal_template_hash,verify_competence_artifact,audit_value_edit_alignment)
from src.experiments.patching import focal_status_margin,focal_status_patch_delta
from scripts.analyze_status_focal_behavior import analyze
from src.experiments.patching import partition_history_ids

VALUES=['amber','coral','denim','elm','frost','gray','hazel','indigo']

def test_focal_render_queries_rendered_literal_name():
 r=generate('development',4,VALUES,991,('bracketed','active_words'))[0]
 r.update(literal_names={'x':'z','z':'x'},query='x')
 assert 'What is z?' in render(r)

def test_confirmatory_has_matched_proposed_and_initial_edit_pairs_across_design_cells():
 rows=generate('confirmatory',96,VALUES,231,('bracketed',));audit(rows,'confirmatory')
 assert len(rows)==96*2*2*2*2*2
 groups={}
 for r in rows:groups.setdefault(r['history_id'],[]).append(r)
 assert len(groups)==96
 for cells in groups.values():
  focal=cells[0]['focal_variable']
  assert {r['edited_field'] for r in cells}=={f'proposed_{focal}',f'initial_{focal}'}
  for q in ('focal','other'):
   for pos in ('focal_first','focal_second'):
    for field in (f'proposed_{focal}',f'initial_{focal}'):
     for direction in (0,1):
      yes=next(r for r in cells if r['query_role']==q and r['focal_position']==pos and r['focal_valid'] and r['edited_field']==field and r['pair_direction']==direction)
      no=next(r for r in cells if r['query_role']==q and r['focal_position']==pos and not r['focal_valid'] and r['edited_field']==field and r['pair_direction']==direction)
      assert yes['matching_values']==no['matching_values']
      assert yes['replacement_value']==no['replacement_value']
      assert yes['literal_names']==no['literal_names']

def test_frozen_gate_cells_perfect_accuracy_and_orientation_balance():
 rows=generate('frozen_gate',24,VALUES,887,('active_words',))
 scores=[{**r,'full_vocab_next_token_accuracy':1,'target_rank':1,'candidate_accuracy':1,'candidate_target_rank':1,'validity_alignment_passed':True} for r in rows]
 result=gate_summary(rows,scores)
 assert result['passed'] and result['complete_cells'] and len(result['cells'])==32
 scores[0]['validity_alignment_passed']=False
 assert not gate_summary(rows,scores)['passed']

def test_validity_prompt_alignment_audits_equal_lengths_and_status_only_differences():
 class CharTokenizer:
  def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False,**kwargs):
   d={'input_ids':[ord(ch) for ch in text]}
   if return_offsets_mapping:d['offset_mapping']=[(i,i+1) for i in range(len(text))]
   return d
 rows=generate('frozen_gate',24,VALUES,887,('bracketed',));yes=next(r for r in rows if r['focal_valid']);no=next(r for r in rows if not r['focal_valid'] and r['history_id']==yes['history_id'] and r['query_role']==yes['query_role'] and r['focal_position']==yes['focal_position'])
 report=audit_validity_prompt_alignment(yes,no,CharTokenizer(),False)
 assert report['status_positions'] and set(report['differing_positions'])<=set(report['status_positions'])

def test_value_edit_alignment_is_one_token_at_the_declared_span():
 class CharTokenizer:
  def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False,**kwargs):
   import re
   spans=[m.span() for m in re.finditer(r'\S+|\s+',text)]
   d={'input_ids':[text[x:y] for x,y in spans]}
   if return_offsets_mapping:d['offset_mapping']=spans
   return d
 rows=generate('confirmatory',96,VALUES,44,('bracketed',));base=next(r for r in rows if r['pair_direction']==0);edited=next(r for r in rows if r['pair_id']==base['pair_id'] and r['pair_direction']==1)
 assert audit_value_edit_alignment(base,edited,CharTokenizer(),False)['prompt_length']>0

def test_concrete_histories_are_disjoint_including_orientation_swap():
 dev=generate('development',4,VALUES,1,('bracketed','active_words'))
 seen=focal_history_signatures(dev);new=generate('frozen_gate',24,VALUES,2,('bracketed',),seen)
 assert not (focal_history_signatures(new)&seen)

def test_mechanistic_split_is_24_discovery_and_72_held_out():
 discovery,heldout=partition_history_ids([f'h{i:03d}' for i in range(96)],24,991)
 assert len(discovery)==24 and len(heldout)==72 and not(set(discovery)&set(heldout))
 assert set(discovery)|set(heldout)=={f'h{i:03d}' for i in range(96)}

def test_focal_status_patch_delta_sign_and_donor_orientation():
 ids={'proposed':0,'initial':1}
 assert focal_status_margin([5.,2.],'proposed','initial',ids)==3.
 assert focal_status_patch_delta([5.,2.],[3.,4.],'proposed','initial',ids,True)=={'S_recipient':3.,'S_patched':-1.,'delta_S_patch':-4.,'donor_oriented_delta_S_patch':-4.}
 reverse=focal_status_patch_delta([3.,4.],[5.,2.],'proposed','initial',ids,False)
 assert reverse['delta_S_patch']==4. and reverse['donor_oriented_delta_S_patch']==-4.

def test_behavioral_validity_contrasts_and_symmetric_history_paired_effects():
 rows=[]
 for i in range(96):
  hid,focal=f'h{i:03d}',('x' if i%2==0 else 'z')
  for valid in (True,False):
   for pos in ('focal_first','focal_second'):
    for role in ('focal','other'):
     for field in (f'proposed_{focal}',f'initial_{focal}'):
      R=(5 if valid else 2) if field.startswith('proposed') else (1 if valid else 7)
      E=R if role=='focal' else 0
      rows.append({'history_id':hid,'focal_variable':focal,'focal_valid':valid,'focal_position':pos,'query_role':role,'edited_field':field,'identity_transfer_E':E,
       'baseline_competence':{'full_vocab_next_token_accuracy':1,'candidate_accuracy':1,'target_rank':1,'candidate_target_rank':1},'edited_competence':{},'baseline_candidate_logits':{},'edited_candidate_logits':{}})
 _,_,summary=analyze(rows,n_boot=100)
 got={(x['metric'],x['axis']):x['mean'] for x in summary}
 assert got[('delta_R_proposed','x')]==3 and got[('delta_R_initial','z')]==6
 assert got[('delta_R_proposed','symmetric')]==3 and got[('delta_R_initial','symmetric')]==6

def test_frozen_artifact_recomputes_instead_of_trusting_pass_boolean(tmp_path):
 values=VALUES
 ds=generate('frozen_gate',24,values,887,('active_words',));dataset=tmp_path/'gate.jsonl';scores=tmp_path/'scores.jsonl'
 write_jsonl(ds,dataset)
 ss=[{**r,'full_vocab_next_token_accuracy':1,'target_rank':1,'candidate_accuracy':1,'candidate_target_rank':1,'validity_alignment_passed':True} for r in ds]
 ss[0]['full_vocab_next_token_accuracy']=0;write_jsonl(ss,scores)
 config=tmp_path/'config.yaml';config.write_text('model: {}\n');token=tmp_path/'tokens.json';token.write_text(json.dumps({'token_ids':{v:i for i,v in enumerate(values)},'model_revision':'a'*40,'tokenizer_revision':'a'*40}))
 prov={'purpose':'competence_gate_only','experiment_kind':'status_focal','stage':'frozen_gate','dataset_sha256':sha256_file(dataset),'config_sha256':sha256_file(config),'token_map_sha256':sha256_file(token),'template_sha256':focal_template_hash(),'resolved_model_revision':'a'*40,'resolved_tokenizer_revision':'a'*40,'causal_effects_computed':False}
 pp=tmp_path/'scores.jsonl.provenance.json';pp.write_text(json.dumps(prov))
 summary=gate_summary(ds,ss)
 artifact={'stage':'frozen_gate','template_sha256':focal_template_hash(),'dataset_path':str(dataset),'dataset_sha256':sha256_file(dataset),'scores_path':str(scores),'scores_sha256':sha256_file(scores),'scores_provenance_path':str(pp),'scores_provenance_sha256':sha256_file(pp),'competence_summary':summary,'scoring_provenance':prov,'selected_variant':'active_words','passed':True}
 path=tmp_path/'gate.json';path.write_text(json.dumps(artifact))
 with pytest.raises(ValueError):verify_competence_artifact(path,'frozen_gate',config,token)
