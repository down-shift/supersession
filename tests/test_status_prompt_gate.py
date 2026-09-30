"""CPU-only scaffolds: no pretrained model is loaded by these tests."""
import copy
import json
import sys
from pathlib import Path

import pytest

from src.data.io import read_jsonl, write_jsonl, sha256_file
from src.data.status_prompt_gate import (VARIANTS, generate_prompt_stage, audit_prompt_dataset,
    competence_summary, checked_artifact, history_signatures, template_hash, audit_final_dataset)
from src.data.supersession_behavior import render_behavior_example
from scripts.generate_status_prompt_stage import generate_stage
from scripts.analyze_status_prompt_gate import analyze

VALUES = ['amber','coral','denim','elm','frost','grape','jade','maple','navy','pearl','quartz','rust']


def scores_for(rows):
    return [{**r,'prompt':render_behavior_example(r,chat=False),
             'full_vocab_next_token_accuracy':1,'target_rank':1} for r in rows]


def inputs(tmp_path):
    values, tokens = tmp_path/'values.json', tmp_path/'tokens.json'
    values.write_text(json.dumps(VALUES))
    tokens.write_text(json.dumps({'token_ids':{v:i for i,v in enumerate(VALUES)}}))
    return values, tokens


def scaffold_scoring(dataset, token_path, failure=False):
    rows = read_jsonl(dataset)
    scores = scores_for(rows)
    if failure:
        scores[0].update(full_vocab_next_token_accuracy=0,target_rank=2)
    output = Path(str(dataset)+'.scores.jsonl')
    write_jsonl(scores,output)
    Path(str(output)+'.provenance.json').write_text(json.dumps({
        'dataset_sha256':sha256_file(dataset),'token_map_sha256':sha256_file(token_path),
        'config_sha256':'b'*64,'model_id':'synthetic_fixture', 'model_revision':'a'*40,
        'tokenizer_revision':'a'*40,'causal_effects_computed':False}))
    return output


def development(tmp_path, select='bracketed'):
    values, tokens = inputs(tmp_path)
    dataset = tmp_path/'development.jsonl'
    generate_stage('development',values,tokens,dataset,n=3,seed=81701)
    scores = scaffold_scoring(dataset,tokens)
    report = tmp_path/'development_analysis'
    assert analyze(dataset,scores,report,select)
    return values,tokens,dataset,report/'selection.json'


def frozen(tmp_path, failure=False):
    values,tokens,dev,selection = development(tmp_path)
    gate = tmp_path/'frozen.jsonl'
    generate_stage('frozen_gate',values,tokens,gate,seed=81702,selection_path=selection)
    scores = scaffold_scoring(gate,tokens,failure)
    report = tmp_path/'frozen_analysis'
    assert analyze(gate,scores,report) is not failure
    return values,tokens,dev,gate,report/'frozen_gate.json'


def test_update_order_is_independent_and_all_cells_are_balanced():
    rows = generate_prompt_stage('development',3,VALUES,701)
    assert len(rows) == 3*3*4*2*2
    assert audit_prompt_dataset(rows) == 'development'
    assert {r['variables'][0] for r in rows} == {'x'}
    for hid in {r['history_id'] for r in rows}:
        cells = [r for r in rows if r['history_id']==hid]
        assert len(cells)==48
        for variant in VARIANTS:
            for condition in ('YY','YN','NY','NN'):
                for query in 'xz':
                    pair = [r for r in cells if (r['prompt_variant'],r['condition'],r['query']) == (variant,condition,query)]
                    assert {r['update_order'] for r in pair} == {'xz','zx'}
                    a,b = pair
                    assert a['semantic_values'] == b['semantic_values']
                    assert a['answer'] == b['answer']
                    text = render_behavior_example(a,chat=False)
                    updates = text.split('Updates:\n')[1].splitlines()[:2]
                    assert a['update_order'][0] in updates[0]
    summary = competence_summary(rows,scores_for(rows))
    assert len(summary['cells'])==48 and summary['gate_pass']
    assert all(c['n_histories']==3 and c['target_rank_max']==1 for c in summary['cells'])


def test_variant_isolation_missing_cells_and_causal_fields():
    rows = generate_prompt_stage('frozen_gate',24,VALUES,702,('inline_status',))
    assert len(rows)==384 and {r['prompt_variant'] for r in rows}=={'inline_status'}
    with pytest.raises(ValueError,match='isolate'):
        generate_prompt_stage('frozen_gate',24,VALUES,702,VARIANTS)
    with pytest.raises(ValueError,match='missing required'):
        audit_prompt_dataset(rows[:-1])
    bad = copy.deepcopy(rows); bad[0]['prompt_variant']='action_labels'
    with pytest.raises(ValueError,match='one isolated'):
        audit_prompt_dataset(bad)
    scores = scores_for(rows); scores[0]['identity_transfer']=3
    with pytest.raises(ValueError,match='causal'):
        competence_summary(rows,scores)


def test_variants_do_not_disclose_final_binding_and_answers_use_status():
    rows = generate_prompt_stage('development',1,VALUES,703)
    for row in rows:
        selected = 'proposed' if row['update_accepted_by_variable'][row['query']] else 'initial'
        assert row['answer']==row['semantic_values'][f'{selected}_{row["query"]}']
        text = render_behavior_example(row,chat=False)
        assert 'current value' not in text and 'Final state' not in text
        # Every value occurs once; no answer is supplied as a final-state annotation.
        assert all(text.count(v)==1 for v in row['semantic_values'].values())
    for variant in VARIANTS:
        assert any(r['prompt_variant']==variant for r in rows)


def test_one_error_in_one_cell_fails_even_if_overall_accuracy_exceeds_99_percent():
    rows = generate_prompt_stage('frozen_gate',24,VALUES,704,('bracketed',))
    scores = scores_for(rows)
    scores[0].update(full_vocab_next_token_accuracy=0,target_rank=2)
    report = competence_summary(rows,scores)
    assert not report['gate_pass'] and report['threshold']==1.0
    assert len([c for c in report['cells'] if c['full_vocab_accuracy']<1])==1
    assert sum(s['full_vocab_next_token_accuracy'] for s in scores)/len(scores)>.99


def test_fresh_seeds_and_concrete_histories():
    dev = generate_prompt_stage('development',24,VALUES,705)
    gate = generate_prompt_stage('frozen_gate',24,VALUES,706,('bracketed',),dev)
    assert not history_signatures(dev) & history_signatures(gate)
    with pytest.raises(ValueError,match='separate seed'):
        generate_prompt_stage('frozen_gate',24,VALUES,705,('bracketed',),dev)


def test_missing_or_development_gate_cannot_generate_confirmatory(tmp_path):
    values,tokens,dev,selection = development(tmp_path)
    output = tmp_path/'final.jsonl'
    with pytest.raises(ValueError,match='explicit passing'):
        generate_stage('confirmatory',values,tokens,output)
    with pytest.raises(ValueError,match='wrong stage'):
        generate_stage('confirmatory',values,tokens,output,frozen_gate_path=selection)
    assert not output.exists()
    with pytest.raises(ValueError,match='explicit passing'):
        generate_prompt_stage('confirmatory',96,VALUES,81703,('bracketed',))


def test_development_report_does_not_implicitly_select_wording(tmp_path):
    values,tokens=inputs(tmp_path)
    dataset=tmp_path/'development.jsonl'
    generate_stage('development',values,tokens,dataset,n=2,seed=81705)
    scores=scaffold_scoring(dataset,tokens)
    report=tmp_path/'report'
    assert analyze(dataset,scores,report)
    assert (report/'development_report.json').exists() and not (report/'selection.json').exists()
    with pytest.raises(ValueError,match='selected variant'):
        generate_stage('frozen_gate',values,tokens,tmp_path/'gate.jsonl',seed=81706,
                       selection_path=report/'development_report.json')


def test_failed_gate_is_preserved_and_blocks_final_even_if_boolean_is_changed(tmp_path):
    values,tokens,dev,gate,artifact = frozen(tmp_path,failure=True)
    output = tmp_path/'final.jsonl'
    with pytest.raises(ValueError,match='failed'):
        generate_stage('confirmatory',values,tokens,output,frozen_gate_path=artifact)
    data = json.loads(artifact.read_text()); data['gate_pass']=True
    artifact.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='recomputed'):
        generate_stage('confirmatory',values,tokens,output,frozen_gate_path=artifact)
    assert not output.exists() and gate.exists()


def test_passing_gate_binds_template_tokenmap_and_fresh_final(tmp_path):
    values,tokens,dev,gate,artifact = frozen(tmp_path)
    assert checked_artifact(artifact,'frozen_gate')['gate_pass']
    output = tmp_path/'final.jsonl'
    generate_stage('confirmatory',values,tokens,output,frozen_gate_path=artifact,seed=81703)
    rows = read_jsonl(output)
    assert len(rows)==96*4*4*2*2*2
    assert audit_final_dataset(rows)=='status_2x2'
    assert not history_signatures(rows) & history_signatures(read_jsonl(dev)+read_jsonl(gate))
    with pytest.raises(FileExistsError):
        generate_stage('confirmatory',values,tokens,output,frozen_gate_path=artifact)
    tokens.write_text(tokens.read_text()+'\n')
    with pytest.raises(ValueError,match='token map differs'):
        generate_stage('confirmatory',values,tokens,tmp_path/'changed.jsonl',frozen_gate_path=artifact)


def test_changed_gate_inputs_and_templates_fail_closed(tmp_path):
    values,tokens,dev,gate,artifact = frozen(tmp_path)
    data = json.loads(artifact.read_text()); data['template_sha256']='changed'
    artifact.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='changed prompt'):
        checked_artifact(artifact,'frozen_gate')
    data['template_sha256']=template_hash(); artifact.write_text(json.dumps(data))
    gate.write_text(gate.read_text()+'\n')
    with pytest.raises(ValueError,match='input hash'):
        checked_artifact(artifact,'frozen_gate')


def test_legacy_generation_cli_is_fail_closed_for_any_status_2x2_request(tmp_path,monkeypatch):
    from scripts.generate_supersession_experiments import main
    values,tokens=inputs(tmp_path)
    monkeypatch.setattr(sys,'argv',['generate','--kind','status_2x2','--values',str(values),
        '--token-ids',str(tokens),'--output',str(tmp_path/'final.jsonl'),'--n','96'])
    with pytest.raises(ValueError,match='requires --frozen-gate'):
        main()
    assert not (tmp_path/'final.jsonl').exists()


def test_accuracy_runner_preserves_new_metadata_and_resume_without_model_loading(tmp_path):
    import scripts.run_status_2x2_gate as runner
    from tests.test_supersession_behavior import WordTokenizer, scoring_stub
    from src.data.progress import prepare_jsonl_progress
    rows = generate_prompt_stage('development',1,VALUES,707)
    values,tokens=inputs(tmp_path)
    dataset,output=tmp_path/'rows.jsonl',tmp_path/'scores.jsonl'
    write_jsonl(rows,dataset)
    ids={v:i+1 for i,v in enumerate(VALUES)}
    model=scoring_stub(); tok=WordTokenizer()
    completed=prepare_jsonl_progress(output,dataset,tokens,{'test':True},rows)
    runner.score_gate(model,tok,rows,ids,output,completed,False)
    scores=read_jsonl(output)
    assert len(scores)==len(rows)
    assert all(all(s[k]==v for k,v in r.items()) for r,s in zip(rows,scores))
    assert all(not set(s)&{'candidate_logits','candidate_accuracy','candidate_target_rank','identity_transfer'} for s in scores)
    calls=model.calls
    completed=prepare_jsonl_progress(output,dataset,tokens,{'test':True},rows,resume=True)
    runner.score_gate(model,tok,rows,ids,output,completed,False)
    assert model.calls==calls
    with pytest.raises(ValueError,match='fingerprint'):
        prepare_jsonl_progress(output,dataset,tokens,{'test':False},rows,resume=True)
