import json
import hashlib
import subprocess
import sys
import pytest
from src.data.query_reactivation import generate, audit, signature, render, template_hash, validate_dataset_provenance, seal_artifact, verify_sealed_artifact
from src.analysis.query_reactivation import history_contrasts, summarize_histories

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

def test_generator_preserves_existing_artifact(tmp_path):
    out=tmp_path/'already.jsonl';out.write_text('keep\n')
    proc=subprocess.run([sys.executable,'scripts/generate_query_reactivation.py','--stage','development',
      '--output',str(out),'--values-json','configs/query_reactivation_values.json'],capture_output=True,text=True)
    assert proc.returncode!=0 and 'FileExistsError' in proc.stderr
    assert out.read_text()=='keep\n'
