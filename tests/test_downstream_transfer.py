import json,subprocess,sys
from pathlib import Path
import pytest
from src.data.downstream_transfer import generate,audit,render,signature,seal_artifact,verify_sealed_artifact
from src.analysis.downstream_transfer import history_contrasts,evaluate_competence_gate,summarize_histories,score_protocol_record,sequence_logprob

def _scored_rows(rows,effects):
 out=[]
 for r in rows:
  cb=r['codebook']; lp={v:0.0 for v in r['code_vocabulary']}
  if r['pair_direction']:
   e=effects[(r['edited_binding'],r['query_id'])]
   lp[cb[r['replacement_value']]]=e/2;lp[cb[r['source_value']]]=-e/2
  out.append({**r,'candidate_logprobs':lp,'current_code_accuracy':1,'causal_effects_computed':True})
 return out

def test_render_codebook_and_prompt_queries():
 r=generate('development',24)[0];p=render(r,chat=False)
 assert 'Codebook:' in p and '->' in p and "current badge" in p and 'Nora' in p
 assert r['answer_code']==r['codebook'][r['correct_value']]
 assert r['query_id'] in ('current_x','current_z')

def test_random_counterbalanced_code_assignment_and_permutation_invariance():
 rows=generate('development',24); maps=[r['codebook'] for r in rows if r['query_id']=='current_x']
 assert len({tuple(m[v] for v in sorted(m)) for m in maps})>1
 for m in maps: assert len(set(m.values()))==len(m)
 # Code labels have no role-specific assignment: all values can occupy all labels across histories.
 assert set(c for m in maps for c in m.values())==set(rows[0]['code_vocabulary'])

def test_stale_answer_invariant_and_live_answer_changes():
 rows=generate('confirmatory',96);by={(r['history_id'],r['edited_binding'],r['query_id'],r['pair_direction']):r for r in rows}
 h=rows[0]['history_id']
 for b in ('old_x','old_z'):
  for q in ('current_x','current_z'):
   assert by[h,b,q,0]['answer_code']==by[h,b,q,1]['answer_code']
 for b in ('current_x','current_z'):
  b0=by[h,b,b,0];b1=by[h,b,b,1]
  assert b0['answer_code']==b0['codebook'][b0['source_value']]
  assert b1['answer_code']==b1['codebook'][b1['replacement_value']]
  assert b0['answer_code']!=b1['answer_code']
 assert {tuple(r['variables']) for r in rows}=={('Nora','Owen'),('Owen','Nora')}

def test_stale_pair_prompt_changes_only_obsolete_binding():
 rows=generate('confirmatory',96);h=rows[0]['history_id']
 base=next(r for r in rows if r['history_id']==h and r['edited_binding']=='old_x' and r['query_id']=='current_x' and r['pair_direction']==0)
 edit=next(r for r in rows if r['pair_id']==base['pair_id'] and r['pair_direction']==1)
 a=render(base,chat=False).splitlines();b=render(edit,chat=False).splitlines()
 assert len(a)==len(b)
 assert [i for i,(x,y) in enumerate(zip(a,b)) if x!=y]==[0]
 live=next(r for r in rows if r['history_id']==h and r['edited_binding']=='current_x' and r['query_id']=='current_x' and r['pair_direction']==0)
 live_edit=next(r for r in rows if r['pair_id']==live['pair_id'] and r['pair_direction']==1)
 a=render(live,chat=False).splitlines();b=render(live_edit,chat=False).splitlines()
 assert [i for i,(x,y) in enumerate(zip(a,b)) if x!=y]==[2]

def test_source_replacement_code_lookup_and_exact_pair_checks():
 rows=generate('confirmatory',96);r=rows[0];assert r['codebook'][r['source_value']]!=r['codebook'][r['replacement_value']]
 scored=_scored_rows(rows,{(b,q):0 for b in ('old_x','old_z','current_x','current_z') for q in ('current_x','current_z')})
 scored[1]['codebook']=dict(scored[1]['codebook'], amber='BAD')
 with pytest.raises(ValueError):history_contrasts(scored)

def test_primary_estimand_orientation_synthetic_transfer_and_irrelevant_subtraction():
 rows=generate('confirmatory',96); effects={}
 for b in ('old_x','old_z','current_x','current_z'):
  for q in ('current_x','current_z'):
   effects[b,q]=1.0 if (b.endswith('_x') and q=='current_x') or (b.endswith('_z') and q=='current_z') else 0.0
 hist=history_contrasts(_scored_rows(rows,effects))
 assert all(h['R_stale_derived']==pytest.approx(1.) for h in hist)
 assert all(h['R_live_derived']==pytest.approx(1.) for h in hist)
 effects={(b,q):1.0 for b in ('old_x','old_z','current_x','current_z') for q in ('current_x','current_z')}
 hist=history_contrasts(_scored_rows(rows,effects))
 assert all(h['R_stale_derived']==pytest.approx(0.) for h in hist)

def test_xz_symmetry_history_aggregation_and_deterministic_bootstrap():
 rows=generate('confirmatory',96);eff={(b,q):float(b[-1]==q[-1]) for b in ('old_x','old_z','current_x','current_z') for q in ('current_x','current_z')}
 hist=history_contrasts(_scored_rows(rows,eff));assert hist[0]['R_stale_derived']==pytest.approx(1)
 assert len(hist)==96
 assert summarize_histories([h['R_stale_derived'] for h in hist],seed=19,n_boot=300)==summarize_histories([h['R_stale_derived'] for h in hist],seed=19,n_boot=300)

def test_gate_threshold_and_effect_separation():
 rows=generate('frozen_gate',24)
 scores=[{**r,'current_code_accuracy':int(i>0),'causal_effects_computed':False} for i,r in enumerate(rows)]
 assert evaluate_competence_gate(rows,scores)['pass'] is True
 bad=[{**r,'current_code_accuracy':0,'causal_effects_computed':False} for r in rows]
 assert not evaluate_competence_gate(rows,bad)['pass']
 badscores=[{**s,'derived_transfer':0.1} for s in scores]
 with pytest.raises(ValueError):evaluate_competence_gate(rows,badscores)
 assert all(not {'pair_id','edited_binding','replacement_value'}&r.keys() for r in rows)

def test_multitoken_complete_sequence_score():
 torch=pytest.importorskip('torch')
 class Tok:
  def __call__(self,s,add_special_tokens=False):return {'input_ids':[0] if s=='P' else ([0,1,2] if s=='PAB' else [0,1])}
 class Model:
  def parameters(self):yield torch.nn.Parameter(torch.zeros(1))
  def __call__(self,input_ids,use_cache=False):
   logits=torch.zeros((1,input_ids.shape[1],3));logits[0,0,1]=2;logits[0,1,2]=3
   return type('O',(),{'logits':logits})()
 score=sequence_logprob(Model(),Tok(),'P','AB')
 assert score==pytest.approx(float(torch.log_softmax(torch.tensor([0.,2.,0.]),0)[1]+torch.log_softmax(torch.tensor([0.,0.,3.]),0)[2]))

def test_dataset_freshness_and_stage_audit():
 d=generate('development',24);g=generate('frozen_gate',24,excluded=d)
 assert {signature(x) for x in d}.isdisjoint({signature(x) for x in g})
 with pytest.raises(ValueError):generate('development',24,excluded=d)
 assert audit(generate('confirmatory',96))

def test_gate_seal_and_create_only_generation(tmp_path):
 doc=seal_artifact({'pass':True});assert verify_sealed_artifact(doc)=={'pass':True};doc['pass']=False
 with pytest.raises(ValueError):verify_sealed_artifact(doc)
 out=tmp_path/'existing.jsonl';out.write_text('retain\n')
 run=subprocess.run([sys.executable,'scripts/generate_downstream_transfer.py','--stage','development','--output',str(out)],capture_output=True,text=True)
 assert run.returncode!=0 and out.read_text()=='retain\n'

def test_frozen_vocabulary_mutation_rejected_before_confirmatory_generation(tmp_path):
 from src.data.downstream_transfer import template_hash
 from src.data.io import sha256_file
 cfg=tmp_path/'cfg.yaml';cfg.write_text(Path('configs/downstream_transfer_v1.yaml').read_text())
 vals=Path('configs/downstream_transfer_values.json');codes=Path('configs/downstream_transfer_codes.json')
 dataset=tmp_path/'gate.jsonl';dataset.write_text('{}\n');scores=tmp_path/'scores.jsonl';scores.write_text('{}\n')
 doc={'protocol':'downstream_transfer_v1','stage':'frozen_gate','pass':True,'dataset_path':str(dataset),'dataset_sha256':sha256_file(dataset),'scores_path':str(scores),'scores_sha256':sha256_file(scores),'template_sha256':template_hash(),'config_sha256':sha256_file(cfg),'values_sha256':sha256_file(vals),'codes_sha256':sha256_file(codes)}
 gate=tmp_path/'gate.json';gate.write_text(json.dumps(seal_artifact(doc)))
 cfg.write_text(cfg.read_text()+'# mutation\n')
 proc=subprocess.run([sys.executable,'scripts/generate_downstream_transfer.py','--stage','confirmatory','--gate',str(gate),'--config',str(cfg),'--output',str(tmp_path/'confirmation.jsonl')],capture_output=True,text=True)
 assert proc.returncode!=0 and 'frozen config_sha256 changed' in proc.stderr

def test_checkpoint_resumability_fingerprint(tmp_path):
 from src.data.progress import prepare_jsonl_progress,append_jsonl_record
 data=tmp_path/'data.jsonl';data.write_text('{}\n');tokens=tmp_path/'codes.json';tokens.write_text('[]')
 output=tmp_path/'scores.jsonl';rows=[{'example_id':'a'}]
 assert prepare_jsonl_progress(output,data,tokens,{'frozen':1},rows)==set()
 append_jsonl_record(output,{'example_id':'a'})
 assert prepare_jsonl_progress(output,data,tokens,{'frozen':1},rows,resume=True)=={'a'}
 with pytest.raises(ValueError):prepare_jsonl_progress(output,data,tokens,{'frozen':2},rows,resume=True)

def test_scoring_stage_strips_effect_fields():
 r=generate('development',24)[0]
 got=score_protocol_record(r,lambda x:{**x,'candidate_logprobs':{'K7':1},'derived_transfer':3}, {},'development')
 assert 'derived_transfer' not in got and got['causal_effects_computed'] is False
