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
  out.append({**r,'candidate_logprobs':lp,'current_code_accuracy':1,
             'unrestricted_generated_text':r['answer_code'],'unrestricted_generated_code':r['answer_code'],
             'unrestricted_code_accuracy':1,'causal_effects_computed':True})
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
 scores=[{**r,'current_code_accuracy':int(i>0),'unrestricted_generated_text':r['answer_code'] if i>0 else 'wrong',
          'unrestricted_generated_code':r['answer_code'] if i>0 else None,'unrestricted_code_accuracy':int(i>0),
          'causal_effects_computed':False} for i,r in enumerate(rows)]
 assert evaluate_competence_gate(rows,scores)['pass'] is True
 bad=[{**r,'current_code_accuracy':0,'unrestricted_generated_text':'wrong',
       'unrestricted_generated_code':None,'unrestricted_code_accuracy':0,
       'causal_effects_computed':False} for r in rows]
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
  def get_input_embeddings(self):return torch.nn.Embedding(3,1)
  def __call__(self,input_ids,use_cache=False):
   logits=torch.zeros((1,input_ids.shape[1],3));logits[0,0,1]=2
   if input_ids.shape[1]>1:logits[0,1,2]=3
   return type('O',(),{'logits':logits})()
 score=sequence_logprob(Model(),Tok(),'P','AB')
 assert score==pytest.approx(float(torch.log_softmax(torch.tensor([0.,2.,0.]),0)[1]+torch.log_softmax(torch.tensor([0.,0.,3.]),0)[2]))

def test_unrestricted_generation_accepts_only_an_exact_code_label():
 torch=pytest.importorskip('torch')
 from src.experiments.downstream_transfer import generate_unrestricted_code
 class Tokenizer:
  def __init__(self,text):self.text=text
  def __call__(self,prompt,return_tensors,add_special_tokens):
   return {'input_ids':torch.tensor([[1,2]])}
  def decode(self,tokens,skip_special_tokens):return self.text
 class Model:
  def __init__(self):self.embedding=torch.nn.Embedding(3,2)
  def get_input_embeddings(self):return self.embedding
  def generate(self,input_ids,**kwargs):return torch.cat([input_ids,torch.tensor([[2]])],dim=1)
 audit={'codes':{'M2':{'n_tokens':1},'K7':{'n_tokens':1}}}
 result=generate_unrestricted_code(Model(),Tokenizer('M2'), 'prompt',['M2','K7'],audit)
 assert result['unrestricted_generated_code']=='M2'
 result=generate_unrestricted_code(Model(),Tokenizer('The code is M2.'),'prompt',['M2','K7'],audit)
 assert result['unrestricted_generated_code'] is None

def test_dataset_freshness_and_stage_audit():
 d=generate('development',24);g=generate('frozen_gate',24,excluded=d)
 assert {signature(x) for x in d}.isdisjoint({signature(x) for x in g})
 with pytest.raises(ValueError):generate('development',24,excluded=d)
 assert audit(generate('confirmatory',96))

def test_model_panel_seed_profile_is_shared_and_fresh_from_qwen():
 from src.data.downstream_transfer import SEED_PROFILES
 assert SEED_PROFILES['gemma_phi_panel_v1']=={'development':20261040,'frozen_gate':20261041,'confirmatory':20261042}
 assert 'seed_profile' not in generate('development',24)[0]  # Preserve the established Qwen row schema.
 legacy_all=[]
 for stage,count in (('development',24),('frozen_gate',24),('confirmatory',96)):
  legacy_all.extend(generate(stage,count))
 for stage,count in (('development',24),('frozen_gate',24),('confirmatory',96)):
  panel_a=generate(stage,count,excluded=legacy_all,seed_profile='gemma_phi_panel_v1')
  panel_b=generate(stage,count,excluded=legacy_all,seed_profile='gemma_phi_panel_v1')
  assert panel_a==panel_b  # Shared concrete histories/codebooks for paired models.
  assert audit(panel_a)
  old={(signature(r)) for r in legacy_all}
  fresh={(signature(r)) for r in panel_a}
  assert old.isdisjoint(fresh)
  assert {r['seed_profile'] for r in panel_a}=={'gemma_phi_panel_v1'}

def test_model_configs_are_pinned_and_use_shared_panel_profile():
 from src.utils import load_config
 expected={
  'gemma3_4b':('google/gemma-3-4b-it','093f9f388b31de276ce2de164bdc2081324b9767'),
  'phi4_mini':('microsoft/Phi-4-mini-instruct','cfbefacb99257ffa30c83adab238a50856ac3083'),
 }
 for name,(model_id,revision) in expected.items():
  config=load_config(f'configs/downstream_transfer_v1/{name}.yaml')
  assert config['model']['id']==model_id
  assert config['model']['revision']==revision
  assert config['model']['tokenizer_revision']==revision
  assert config['data_seed_profile']=='gemma_phi_panel_v1'
  assert config['thresholds']['gate_accuracy']==.97

def test_prior_exclusion_validation_allows_other_model_provenance(tmp_path):
 from scripts.generate_downstream_transfer import validate_exclusion_dataset
 from src.data.io import sha256_file
 rows=generate('development',24)
 path=tmp_path/'prior.jsonl'
 path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
 from src.data.downstream_transfer import template_hash
 provenance={'protocol':'downstream_transfer_v1','stage':'development',
             'dataset_sha256':sha256_file(path),'template_sha256':template_hash(),
             'model_revision':'different-model-revision'}
 Path(str(path)+'.provenance.json').write_text(json.dumps(provenance))
 assert validate_exclusion_dataset(path,rows[0]['value_vocabulary'],rows[0]['code_vocabulary'])==rows

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
 proc=subprocess.run([sys.executable,'scripts/generate_downstream_transfer.py','--stage','confirmatory','--gate',str(gate),'--token-audit',str(tmp_path/'audit.json'),'--config',str(cfg),'--output',str(tmp_path/'confirmation.jsonl')],capture_output=True,text=True)
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


def test_reject_duplicate_pairs_nonfinite_statistics_and_wrong_sources():
 import copy
 rows=generate('confirmatory',96)
 effects={(b,q):0.0 for b in ('old_x','old_z','current_x','current_z') for q in ('current_x','current_z')}
 scores=_scored_rows(rows,effects)
 with pytest.raises(ValueError,match='duplicate pair direction'):
  history_contrasts(scores+[scores[0]])
 with pytest.raises(ValueError,match='nonfinite'):
  summarize_histories([1.0,float('nan')])
 broken=copy.deepcopy(rows)
 broken[0]['source_value']=broken[0]['matching_values']['current_x']
 with pytest.raises(ValueError,match='source/replacement'):
  audit(broken)


def test_gate_46_of_48_fails():
 rows=generate('frozen_gate',24)
 scores=[{**r,'current_code_accuracy':int(i>=2),'unrestricted_generated_text':r['answer_code'],
          'unrestricted_generated_code':r['answer_code'],'unrestricted_code_accuracy':1,
          'causal_effects_computed':False} for i,r in enumerate(rows)]
 assert evaluate_competence_gate(rows,scores)['pass'] is False

def test_gate_unrestricted_generation_is_diagnostic_not_a_veto():
 rows=generate('frozen_gate',24)
 scores=[{**r,'current_code_accuracy':1,'unrestricted_generated_text':r['answer_code'] if i>=2 else 'wrong',
          'unrestricted_generated_code':r['answer_code'] if i>=2 else None,'unrestricted_code_accuracy':int(i>=2),
          'causal_effects_computed':False} for i,r in enumerate(rows)]
 result=evaluate_competence_gate(rows,scores)
 assert result['current_derived_code_accuracy']==1.0
 assert result['current_unrestricted_generation_accuracy']==pytest.approx(46/48)
 assert result['pass'] is True


@pytest.fixture
def synthetic_workflow(tmp_path,monkeypatch):
 import re
 import scripts.generate_downstream_transfer as generator
 import scripts.run_downstream_transfer as runner
 from src.experiments.downstream_transfer import frozen_metadata,write_json_create
 from src.utils import load_config
 config=tmp_path/'config.yaml';config.write_text(Path('configs/downstream_transfer_v1.yaml').read_text())
 values=tmp_path/'values.json';values.write_text(Path('configs/downstream_transfer_values.json').read_text())
 codes=tmp_path/'codes.json';codes.write_text(Path('configs/downstream_transfer_codes.json').read_text())
 common=['--config',str(config),'--values-json',str(values),'--codes-json',str(codes)]
 metadata=frozen_metadata(load_config(config),config,values,codes)
 code_list=json.loads(codes.read_text());audit_path=tmp_path/'token_audit.json'
 doc={'protocol':'downstream_transfer_v1','audit':'tokenizer_only_no_model_inference',**metadata,
      'tokenizer_id':load_config(config)['model']['id'],'chat_template':True,
      'continuation_prefix_stable':True,'prompts_checked':64,
      'tokenizer_sha256':'synthetic','chat_template_sha256':'synthetic',
      'codes':{code:{'token_ids':[i+1,20],'n_tokens':2} for i,code in enumerate(code_list)}}
 write_json_create(audit_path,seal_artifact(doc))
 calls=[]
 monkeypatch.setattr(runner,'load_model',lambda c:(calls.append('load') or object(),None))
 monkeypatch.setattr(runner,'check_tokenizer',lambda *args:None)
 def fake_scores(model,tokenizer,prompt,candidates,audit,progress_callback=None):
  # Synthetic oracle reads the current value and codebook; no pretrained model.
  entity=re.search(r"Which code corresponds to (\w+)'s",prompt).group(1)
  current=re.search(r"Later, "+entity+r"'s badge was changed to (\w+)\.",prompt).group(1)
  codebook=dict(re.findall(r'(\w+) -> (\w+)',prompt))
  if progress_callback:progress_callback()
  return {code:(-1.0 if code==codebook[current] else -10.0) for code in candidates}
 monkeypatch.setattr(runner,'score_codes',fake_scores)
 def fake_generation(model,tokenizer,prompt,candidates,audit,progress_callback=None):
  entity=re.search(r"Which code corresponds to (\w+)'s",prompt).group(1)
  current=re.search(r"Later, "+entity+r"'s badge was changed to (\w+)\.",prompt).group(1)
  codebook=dict(re.findall(r'(\w+) -> (\w+)',prompt)); code=codebook[current]
  if progress_callback:progress_callback()
  return {'unrestricted_generated_text':code,'unrestricted_generated_code':code}
 monkeypatch.setattr(runner,'generate_unrestricted_code',fake_generation)
 dev=tmp_path/'development.jsonl';gate_data=tmp_path/'gate.jsonl';gate_scores=tmp_path/'gate_scores.jsonl'
 generator.main(common+['--stage','development','--output',str(dev)])
 generator.main(common+['--stage','frozen_gate','--output',str(gate_data),'--prior-dataset',str(dev)])
 score_args=common+['--dataset',str(gate_data),'--token-audit',str(audit_path),'--output',str(gate_scores)]
 runner.main(score_args)
 return dict(root=tmp_path,common=common,config=config,values=values,codes=codes,
             metadata=metadata,audit=audit_path,dev=dev,gate_data=gate_data,gate_scores=gate_scores,
             gate=Path(str(gate_scores)+'.gate.json'),runner=runner,generator=generator,
             score_args=score_args,calls=calls,fake_scores=fake_scores)


def test_interrupted_pair_resume_complete_checkpoint_and_analysis(synthetic_workflow,monkeypatch):
 from src.data.io import read_jsonl
 from scripts.analyze_downstream_transfer import main as analyze
 w=synthetic_workflow
 initial_loads=len(w['calls'])
 w['runner'].main(w['score_args']+['--resume'])
 assert len(w['calls'])==initial_loads
 confirmation=w['root']/'confirmatory.jsonl'
 w['generator'].main(w['common']+['--stage','confirmatory','--gate',str(w['gate']),
                      '--token-audit',str(w['audit']),'--output',str(confirmation)])
 # No explicit development argument: its exclusion is inherited from the gate.
 prov=json.loads(Path(str(confirmation)+'.provenance.json').read_text())
 assert str(w['dev']) in prov['prior_dataset_paths']
 assert prov['model_revision']==w['metadata']['model_revision']
 scores=w['root']/'confirmatory_scores.jsonl'
 args=w['common']+['--dataset',str(confirmation),'--gate',str(w['gate']),
                   '--token-audit',str(w['audit']),'--output',str(scores)]
 counts=[]
 def interrupted(*a,**k):
  counts.append(1)
  if len(counts)==2:raise RuntimeError('synthetic interruption')
  return w['fake_scores'](*a,**k)
 monkeypatch.setattr(w['runner'],'score_codes',interrupted)
 with pytest.raises(RuntimeError,match='synthetic interruption'):w['runner'].main(args)
 assert len(read_jsonl(scores))==1
 monkeypatch.setattr(w['runner'],'score_codes',w['fake_scores'])
 w['runner'].main(args+['--resume'])
 assert len(read_jsonl(scores))==1536
 assert Path(str(scores)+'.complete.json').exists()
 loads=len(w['calls']);w['runner'].main(args+['--resume'])
 assert len(w['calls'])==loads
 summary=analyze(w['common']+['--dataset',str(confirmation),'--scores',str(scores),
                            '--gate',str(w['gate']),'--output-dir',str(w['root']/'analysis'),
                            '--bootstrap-draws','20'])
 assert summary['n_histories']==96
 assert summary['estimands']['R_stale_derived']['mean']==pytest.approx(0)
 assert summary['estimands']['R_live_derived']['mean']==pytest.approx(18)
 assert summary['current_answer_stability']['delta_correct_logprob']['n_histories']==96
 assert summary['current_answer_stability']['edited_accuracy']['mean']==pytest.approx(1)

def test_prior_artifact_path_resolves_after_same_directory_relocation(tmp_path):
 from src.experiments.downstream_transfer import resolve_artifact_path
 original=tmp_path/'old-host'/'development.jsonl'
 local=tmp_path/'new-host'/'development.jsonl'
 local.parent.mkdir();local.write_text('artifact\n')
 assert resolve_artifact_path(original,local)==local


@pytest.mark.parametrize('artifact',['config','values','codes','audit','gate_data','gate_scores','dataset_provenance','score_provenance'])
def test_any_frozen_file_mutation_rejected(synthetic_workflow,artifact):
 from src.experiments.downstream_transfer import validate_gate
 w=synthetic_workflow
 path=(Path(str(w['gate_data'])+'.provenance.json') if artifact=='dataset_provenance' else
       Path(str(w['gate_scores'])+'.provenance.json') if artifact=='score_provenance' else w[artifact])
 path.write_text(path.read_text()+'\n')
 metadata=w['metadata']
 if artifact in ('config','values','codes'):
  from src.experiments.downstream_transfer import frozen_metadata
  from src.utils import load_config
  metadata=frozen_metadata(load_config(w['config']),w['config'],w['values'],w['codes'])
 with pytest.raises(ValueError):validate_gate(w['gate'],metadata,w['audit'])


def test_changed_scoring_hash_rejected(synthetic_workflow):
 from src.experiments.downstream_transfer import validate_gate
 w=synthetic_workflow
 with pytest.raises(ValueError,match='scoring_code_sha256'):
  validate_gate(w['gate'],{**w['metadata'],'scoring_code_sha256':'changed'},w['audit'])


def test_create_only_score_output_rejected_before_model_loading(synthetic_workflow):
 w=synthetic_workflow;loads=len(w['calls'])
 with pytest.raises(FileExistsError):w['runner'].main(w['score_args'])
 assert len(w['calls'])==loads


def test_gate_requires_development_exclusion(tmp_path):
 from scripts.generate_downstream_transfer import main
 with pytest.raises(ValueError,match='development --prior-dataset'):
  main(['--stage','frozen_gate','--output',str(tmp_path/'gate.jsonl')])
 assert not (tmp_path/'gate.jsonl').exists()


def test_code_label_permutation_preserves_history_estimands():
 rows=generate('confirmatory',96)
 effects={(b,q):float(b[-1]==q[-1]) for b in ('old_x','old_z','current_x','current_z') for q in ('current_x','current_z')}
 scores=_scored_rows(rows,effects)
 codes=rows[0]['code_vocabulary'];permutation=dict(zip(codes,codes[::-1]))
 permuted=[{**r,'codebook':{v:permutation[c] for v,c in r['codebook'].items()},
            'candidate_logprobs':{permutation[c]:v for c,v in r['candidate_logprobs'].items()}}
           for r in scores]
 assert history_contrasts(scores)==history_contrasts(permuted)
