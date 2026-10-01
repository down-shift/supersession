import copy
import hashlib
import json
import re
from collections import Counter
import pytest
from src.data.single_deep_chain import generate,audit,audit_tokens,render,signature
from src.analysis.single_deep_chain import relevance,competence,summarize,depth_one_sanity

VALUES = [f'value{i:02d}' for i in range(25)]


class Tokenizer:
    chat_template = None
    def __init__(self):
        self.ids = {v:i for i,v in enumerate(VALUES)}
    def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
        ms = list(re.finditer(r'\S+',text))
        for m in ms:
            self.ids.setdefault(m.group(),len(self.ids))
        result = {'input_ids':[self.ids[m.group()] for m in ms]}
        if return_offsets_mapping:
            result['offset_mapping'] = [m.span() for m in ms]
        return result


def scored(rows):
    result = []
    for r in rows:
        logits = {v:0. for v in r['candidate_values']}; logits[r['answer']] = 5.
        result.append({**copy.deepcopy(r),'candidate_logits':logits,'full_vocab_next_token_accuracy':1,
                       'accuracy':1,'full_vocab_rank':1,'candidate_rank':1})
    return result


def test_design_versions_balance_and_stable_variable_no_updates():
    rows = generate('pilot',8,(2,3,4),VALUES,20261102)
    assert len(rows)==480 and audit(rows)=='pilot'
    refs = {r['history_id']:r for r in rows}
    for depth in (2,3,4):
        assert Counter((r['focal_variable'],r['orientation']) for r in refs.values() if r['depth']==depth)=={
            ('x',0):2,('x',1):2,('z',0):2,('z',1):2}
    for r in rows:
        f,d = r['focal_variable'],r['distractor_variable']
        assert len(r['versions'][f])==r['depth']+1 and len(r['versions'][d])==1
        updates = render(r,chat=False).split('Updates:\n')[1].split('\n\n')[0]
        assert all(line.startswith(r['literal_names'][f]+' = ') for line in updates.splitlines())
        changes = [(v,i) for v in ('x','z') for i in range(len(r['versions'][v]))
                   if r['versions'][v][i]!=r['baseline_versions'][v][i]]
        assert changes==([(r['edited_variable'],r['version_index'])] if r['pair_direction'] else [])
    with pytest.raises(ValueError,match='incomplete'):
        audit(rows[:-1])
    wrong = copy.deepcopy(rows); wrong[0]['versions'][wrong[0]['distractor_variable']].append('wrong')
    with pytest.raises(ValueError,match='exactly'):
        audit(wrong)


def test_one_token_audit_all_versions_and_stable_control():
    rows = generate('pilot',4,(2,3,4),VALUES,20261103)
    assert audit_tokens(rows,Tokenizer(),{v:i for i,v in enumerate(VALUES)},False)['pairs_checked']==120


def test_relevance_sign_controls_and_pooled_history_order():
    rows = generate('pilot',4,(2,3,4),VALUES,20261104)
    effects,R,controls = relevance(rows,scored(rows))
    assert all(r['R']==(10. if r['age_from_current']==0 else 0.) for r in R)
    assert all(r['R_stable']==10. for r in controls)
    assert len(controls)==12
    summary,contrasts = summarize(R,draws=10,seed=1)
    assert sorted((r['age_from_current'],r['version_index']) for r in summary if r['depth']==3 and r['axis']=='pooled')==[(0,3),(1,2),(2,1),(3,0)]
    assert all(r['mean']==10. for r in contrasts if r['contrast']=='current_minus_previous')
    assert all(r['mean']==0. for r in contrasts if r['contrast'].startswith('obsolete_'))
    assert all(r['n_histories']==4 for r in summary if r['axis']=='pooled')


def test_competence_is_per_depth_per_version_and_keeps_98_percent_bar():
    rows = generate('pilot',8,(2,3,4),VALUES,20261105); scores = scored(rows)
    report = competence(rows,scores)
    assert report['minimum_cell_accuracy']==.98 and report['qualified_depths']==[2,3,4]
    for d in report['depths']:
        assert d['baseline_focal_n']==8 and d['baseline_distractor_n']==8
        assert d['baseline_focal_current_minus_previous_mean']==5.
    bad = next(s for s in scores if s['depth']==3 and s['pair_direction']==1 and s['edit_role']=='focal_chain' and s['query_role']=='focal')
    bad.update(full_vocab_next_token_accuracy=0,accuracy=0,full_vocab_rank=2,candidate_rank=2)
    assert competence(rows,scores)['qualified_depths']==[2,4]
    assert len(relevance(rows,scores)[1])==len(relevance(rows,scored(rows))[1])


def test_fresh_histories_and_duplicate_rejection():
    old = generate('pilot',4,(2,3,4),VALUES,20261106)
    fresh = generate('full',4,(2,3,4),VALUES,20261107,{signature(r) for r in old})
    assert not {signature(r) for r in old}&{signature(r) for r in fresh}
    duplicate = [copy.deepcopy(r) for r in old if r['history_id']==old[0]['history_id']]
    for r in duplicate:
        r['history_id'] += '_copy'; r['pair_id'] += '_copy'; r['example_id'] += '_copy'
    with pytest.raises(ValueError,match='duplicate concrete'):
        audit(old+duplicate)


def test_depth_one_is_only_sanity_and_does_not_qualify_new_depths(tmp_path):
    assert not depth_one_sanity(tmp_path)['available']
    (tmp_path/'analysis.json').write_text(json.dumps({'stage':'full'}))
    (tmp_path/'R_summary.csv').write_text('depth,axis,version_index,mean\n1,symmetric,0,1\n1,symmetric,1,20\n')
    (tmp_path/'paired_contrasts.csv').write_text('depth,axis,contrast,mean,ci_low\n1,symmetric,current_minus_previous,19,17\n')
    result = depth_one_sanity(tmp_path)
    assert result['available'] and result['expected_direction'] and result['gap_ci_excludes_zero']
    assert 'qualified_depths' not in result


def test_cli_checkpoint_analysis_and_full_generation_require_recomputed_pilot(tmp_path,monkeypatch):
    from src.data.io import read_jsonl, sha256_file
    from src.data.single_deep_chain import template_hash, render
    from scripts.generate_single_deep_chain import main as generate_main
    from src.analysis.single_deep_chain_gate import verified_pilot
    import scripts.run_single_deep_chain as runner
    from scripts.analyze_single_deep_chain import main as analyze_main
    config = tmp_path/'config.yaml'
    config.write_text('model:\n  id: test\n  revision: '+ 'a'*40 + '\n  tokenizer_revision: '+ 'a'*40 + '\n  chat_template: false\n')
    class Tokenizer:
        chat_template = None
        def __init__(self):
            self.ids = {v:i for i,v in enumerate(VALUES)}
        def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
            ms = list(re.finditer(r'\S+',text))
            for m in ms:
                self.ids.setdefault(m.group(),len(self.ids))
            result = {'input_ids':[self.ids[m.group()] for m in ms]}
            if return_offsets_mapping:
                result['offset_mapping'] = [m.span() for m in ms]
            return result
    tok = Tokenizer()
    tokens = tmp_path/'tokens.json'
    tokens.write_text(json.dumps({'token_ids':{v:i for i,v in enumerate(VALUES)},
        'config_sha256':sha256_file(config),'template_sha256':template_hash(),
        'model_revision':'a'*40,'tokenizer_revision':'a'*40,
        'chat_template_sha256':hashlib.sha256(str(None).encode()).hexdigest()}))
    dataset = tmp_path/'single_deep_chain_pilot.jsonl'
    score_path = tmp_path/'single_deep_chain_pilot_scores.jsonl'
    def invoke(main,args):
        monkeypatch.setattr('sys.argv',['script']+args)
        main()
    common = ['--config',str(config),'--token-ids',str(tokens)]
    invoke(generate_main,common+['--stage','pilot','--depths','2,3','--n','4','--seed','110','--output',str(dataset)])
    calls = []
    monkeypatch.setattr(runner,'load_model',lambda cfg:(None,tok))
    def fake_score(model,tokenizer,row,ids,**kwargs):
        calls.append(row['example_id'])
        return {**scored([row])[0],'prompt':render(row,tokenizer,False)}
    monkeypatch.setattr(runner,'score_example',fake_score)
    run_args = common+['--dataset',str(dataset),'--output',str(score_path)]
    invoke(runner.main,run_args)
    count = len(calls)
    invoke(runner.main,run_args+['--resume'])
    assert len(calls)==count and len(read_jsonl(score_path))==len(read_jsonl(dataset))
    analysis_dir = tmp_path/'pilot_analysis'
    invoke(analyze_main,['--dataset',str(dataset),'--scores',str(score_path),
                        '--output-dir',str(analysis_dir),'--bootstrap-draws','10'])
    report = analysis_dir/'analysis.json'
    assert json.loads(report.read_text())['competence']['qualified_depths']==[2,3]
    assert (analysis_dir/'R_by_age.png').exists()
    full = tmp_path/'single_deep_chain_full.jsonl'
    full_args = common+['--stage','full','--n','4','--seed','111','--output',str(full)]
    with pytest.raises(ValueError,match='requires --pilot-audit'):
        invoke(generate_main,full_args)
    invoke(generate_main,full_args+['--pilot-audit',str(report)])
    assert {r['depth'] for r in read_jsonl(full)}=={2,3}
    assert not {signature(r) for r in read_jsonl(full)} & {signature(r) for r in read_jsonl(dataset)}
    forged = json.loads(report.read_text())
    forged['competence']['qualified_depths'].append(8)
    report.write_text(json.dumps(forged))
    with pytest.raises(ValueError,match='recomputation'):
        verified_pilot(report,config,tokens)
    # Direct CLI execution must reach the full-run guard without importing scripts as a package.
    import os
    import subprocess
    import sys
    result = subprocess.run([sys.executable,'scripts/run_single_deep_chain.py',
        '--config',str(config),'--dataset',str(full),'--token-ids',str(tokens),
        '--output',str(tmp_path/'full_scores.jsonl')],
        env={**os.environ,'PYTHONPATH':''},capture_output=True,text=True)
    assert result.returncode != 0
    assert 'pilot artifact changed or is missing' in result.stderr
    assert "No module named 'scripts'" not in result.stderr
