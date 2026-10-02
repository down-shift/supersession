import json
import hashlib
import subprocess
import sys
import pytest
import runpy
import re
from src.data.query_reactivation import VALUES, generate, audit, signature, render, template_hash, validate_dataset_provenance, seal_artifact, verify_sealed_artifact, validated_stage, candidate_values_digest, verify_gate_values
from src.analysis.query_reactivation import history_contrasts, summarize_histories, score_protocol_record, evaluate_competence_gate, resume_baseline_cache

def test_fresh_histories_and_semantic_cells_are_matched_and_disjoint():
    dev=generate('development',24); gate=generate('frozen_gate',24,excluded=dev)
    assert {signature(r) for r in dev}.isdisjoint({signature(r) for r in gate})
    assert len({signature(r) for r in dev})==24
    h=dev[0]; cells={r['query_id']:r for r in dev if r['history_id']==h['history_id']}
    assert cells['current_x']['answer']==cells['initial_x']['matching_values']['current_x']
    assert cells['initial_x']['answer']==cells['initial_x']['matching_values']['initial_x']
    assert 'current color' in render(cells['current_x'],chat=False)
    assert 'originally' in render(cells['initial_x'],chat=False)

def test_confirmatory_old_edits_semantics_and_orientation():
    rows=generate('confirmatory',96)
    audit(rows)
    for h in {r['history_id'] for r in rows}:
        cells={(r['edited_binding'],r['query_id'],r['pair_direction']):r for r in rows if r['history_id']==h}
        for b in ('old_x','old_z'):
            assert cells[b,f'initial_{b[-1]}',0]['answer']==cells[b,f'initial_{b[-1]}',0]['source_value']
            assert cells[b,f'initial_{b[-1]}',1]['answer']==cells[b,f'initial_{b[-1]}',1]['replacement_value']
            assert cells[b,f'current_{b[-1]}',1]['answer']==cells[b,f'current_{b[-1]}',1]['matching_values'][f'current_{b[-1]}']

def test_scorer_routes_synthetic_confirmatory_pair_and_computes_identity_transfer():
    rows=generate('confirmatory',96)
    base=next(r for r in rows if r['edited_binding']=='old_x' and r['query_id']=='current_x' and r['pair_direction']==0)
    edit=next(r for r in rows if r['pair_id']==base['pair_id'] and r['pair_direction']==1)
    stage=validated_stage(rows)
    source,replacement=base['source_value'],base['replacement_value']
    cache={}
    def fake_score(row):
        logits={source:0.,replacement:0.}
        if row['pair_direction']==1: logits.update({source:-1.,replacement:2.})
        return {**row,'candidate_logits':logits,'candidate_probabilities':{source:.5,replacement:.5}}
    first=score_protocol_record(base,fake_score,cache,stage)
    second=score_protocol_record(edit,fake_score,cache,stage)
    assert 'identity_transfer' not in first
    assert second['identity_transfer']==pytest.approx(3.)
    assert second['causal_effects_computed'] is True

def test_aggregate_competence_gate_keeps_orientation_cells_diagnostic():
    rows=generate('frozen_gate',24)
    scores=[{**r,'full_vocab_next_token_accuracy':int(i!=0),'causal_effects_computed':False} for i,r in enumerate(rows)]
    result=evaluate_competence_gate(rows,scores)
    assert result['pass'] is True
    assert result['aggregate_task_competence']['current']['n']==48
    assert result['aggregate_task_competence']['current']['full_vocab_accuracy']==pytest.approx(47/48)
    assert min(c['full_vocab_accuracy'] for c in result['query_variable_orientation_diagnostics'].values())<.97
    bad=[{**r,'full_vocab_next_token_accuracy':0,'causal_effects_computed':False} for r in rows]
    assert evaluate_competence_gate(rows,bad)['pass'] is False

def test_frozen_vocabulary_and_threshold_have_one_source():
    assert list(VALUES)==json.loads(open('configs/query_reactivation_values.json').read())
    config=__import__('yaml').safe_load(open('configs/query_reactivation_v1.yaml'))
    assert config['thresholds']['gate_accuracy']==.97
    with pytest.raises(ValueError,match='0.97'):
        evaluate_competence_gate(generate('frozen_gate',24),[],threshold=.99)

def test_resume_cache_uses_baseline_pair_ids_and_rejects_orphan_edit():
    rows=generate('confirmatory',96)
    base=next(r for r in rows if r['pair_direction']==0)
    edit=next(r for r in rows if r['pair_id']==base['pair_id'] and r['pair_direction']==1)
    cache=resume_baseline_cache([base],'confirmatory')
    assert cache=={base['pair_id']:base}
    with pytest.raises(ValueError,match='lacks baseline'):
        resume_baseline_cache([edit],'confirmatory')

def test_query_reactivation_rejects_duplicate_pairs_and_nonfinite_history_values():
    rows=generate('confirmatory',96)
    base=next(r for r in rows if r['pair_direction']==0)
    with pytest.raises(ValueError,match='duplicate pair direction'):
        history_contrasts([base,base])
    with pytest.raises(ValueError,match='nonfinite'):
        summarize_histories([1.0,float('nan')])

def test_cli_development_and_frozen_gate_work_without_confirmatory_gate_block(tmp_path,monkeypatch):
    from src.models import loader
    from src.experiments import behavior
    from src.data.io import read_jsonl
    class FakeTokenizer:
        chat_template=None
        def __init__(self): self.ids={value:index+1 for index,value in enumerate(VALUES)}
        def __call__(self,text,add_special_tokens=False,**kwargs):
            ids=[]
            for token in re.findall(r'\S+',text):
                self.ids.setdefault(token,len(self.ids)+100); ids.append(self.ids[token])
            return {'input_ids':ids}
    tokenizer=FakeTokenizer()
    monkeypatch.setattr(loader,'load_model',lambda config:(object(),tokenizer))
    def fake_score(model,tok,row,ids,**kwargs):
        return {**row,'full_vocab_next_token_accuracy':1,'full_vocab_rank':1,'accuracy':1,
                'candidate_logits':{v:float(v==row['answer']) for v in ids},
                'candidate_probabilities':{v:float(v==row['answer']) for v in ids},
                'generated_first_token':' '+row['answer']}
    monkeypatch.setattr(behavior,'score_example',fake_score)
    value_map={value:index+1 for index,value in enumerate(VALUES)}
    token_path=tmp_path/'tokens.json'
    token_path.write_text(json.dumps({'tokenizer_id':'Qwen/Qwen3-8B',
        'revision':'b968826d9c46dd6066d109eabc6255188de91218',
        'tokenizer_revision':'b968826d9c46dd6066d109eabc6255188de91218',
        'chat_template':True,'token_ids':value_map}))
    config='configs/query_reactivation_v1.yaml'
    dev=tmp_path/'development.jsonl'; gate=tmp_path/'gate.jsonl'
    for stage,out,prior in [('development',dev,[]),('frozen_gate',gate,[dev])]:
        command=[sys.executable,'scripts/generate_query_reactivation.py','--stage',stage,
                 '--output',str(out),'--values-json','configs/query_reactivation_values.json']
        for path in prior: command += ['--prior-dataset',str(path)]
        assert subprocess.run(command,capture_output=True,text=True).returncode==0
        scores=tmp_path/f'{stage}_scores.jsonl'
        monkeypatch.setattr(sys,'argv',['scripts/run_query_reactivation.py','--dataset',str(out),
            '--token-ids',str(token_path),'--config',config,'--output',str(scores)])
        runpy.run_path('scripts/run_query_reactivation.py',run_name='__main__')
        assert len((tmp_path/f'{stage}_scores.jsonl').read_text().splitlines())==len(read_jsonl(out))

def test_primary_contrast_xz_symmetry_and_history_not_pair_aggregation():
    rows=[]
    # Per history old-query selectivity: current=.2; historical=.8; Delta=.6.
    for hid in ('h1','h2'):
        cells={('old_x','current_x'):.2,('old_x','current_z'):0.,
          ('old_z','current_z'):0.,('old_z','current_x'):-.2,
          ('old_x','initial_x'):.9,('old_x','initial_z'):.1,
          ('old_z','initial_z'):.7,('old_z','initial_x'):-.1}
        for (b,q),e in cells.items():
            base={'history_id':hid,'pair_id':f'{hid}:{b}:{q}','edited_binding':b,'query_id':q,'source_value':'old','replacement_value':'new'}
            # Difference of replacement/source log odds is exactly e.
            rows.extend([{**base,'pair_direction':0,'candidate_logits':{'new':0.,'old':0.}},
                         {**base,'pair_direction':1,'candidate_logits':{'new':e/2,'old':-e/2}}])
    out=history_contrasts(rows)
    assert len(out)==2
    assert out[0]['R_old_current']==pytest.approx(.2)
    assert out[0]['R_old_historical']==pytest.approx(.8)
    assert out[0]['Delta_reactivate']==pytest.approx(.6)
    assert summarize_histories([.6,.6],seed=42,n_boot=100)==summarize_histories([.6,.6],seed=42,n_boot=100)

def test_confirmatory_audit_rejects_bad_history_match_and_effect_leak_to_gate():
    rows=generate('frozen_gate',24)
    rows[1]['matching_values']['initial_x']='mutated'
    with pytest.raises(ValueError): audit(rows)
    assert all(not any(k in r for k in ('pair_id','identity_transfer','replacement_value')) for r in generate('frozen_gate',24))

def test_hash_is_stable_nonempty():
    assert len(template_hash())==64

def test_dataset_provenance_hash_fails_closed(tmp_path):
    data=tmp_path/'stage.jsonl';data.write_text('{"x":1}\n')
    side=tmp_path/'stage.jsonl.provenance.json'
    side.write_text(json.dumps({'protocol':'query_reactivation_v1','dataset_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'template_sha256':template_hash()}))
    assert validate_dataset_provenance(data,side)['protocol']=='query_reactivation_v1'
    data.write_text('{"x":2}\n')
    with pytest.raises(ValueError,match='hash mismatch'): validate_dataset_provenance(data,side)

def test_gate_artifact_seal_rejects_mutation():
    sealed=seal_artifact({'stage':'frozen_gate','pass':True,'threshold':.99})
    assert verify_sealed_artifact(sealed)=={'stage':'frozen_gate','pass':True,'threshold':.99}
    sealed['pass']=False
    with pytest.raises(ValueError,match='seal mismatch'): verify_sealed_artifact(sealed)

def test_frozen_gate_binds_candidate_values():
    values=['amber','coral','denim','elm','frost','grape']
    gate={'candidate_values':values,'candidate_values_sha256':candidate_values_digest(values)}
    assert verify_gate_values(values,gate)
    with pytest.raises(ValueError,match='candidate values'): verify_gate_values(values[:-1]+['new'],gate)

def test_generator_preserves_existing_artifact(tmp_path):
    out=tmp_path/'already.jsonl';out.write_text('keep\n')
    proc=subprocess.run([sys.executable,'scripts/generate_query_reactivation.py','--stage','development',
      '--output',str(out),'--values-json','configs/query_reactivation_values.json'],capture_output=True,text=True)
    assert proc.returncode!=0 and 'FileExistsError' in proc.stderr
    assert out.read_text()=='keep\n'
