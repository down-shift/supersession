import copy
import re
import json
import hashlib
import pytest
from src.data.version_chain import generate, audit, audit_tokens, signature
from src.analysis.version_chain import identity_transfer, relevance, competence, summarize

VALUES = [f'value{i:02d}' for i in range(25)]


def scored(rows):
    scores = []
    for r in rows:
        logits = {v:0. for v in r['candidate_values']}
        logits[r['answer']] = 5.
        scores.append({**copy.deepcopy(r),'candidate_logits':logits,
            'full_vocab_next_token_accuracy':1,'accuracy':1,'full_vocab_rank':1,'candidate_rank':1})
    return scores


def test_versions_all_depths_have_exactly_one_edit_and_complete_query_cells():
    rows = generate('pilot',4,(1,2,4,8),VALUES,101)
    assert audit(rows) == 'pilot'
    for r in rows:
        assert len(r['versions']['x']) == r['depth']+1
        assert r['age_from_current'] == r['depth']-r['version_index']
        changed = [(v,i) for v in ('x','z') for i in range(r['depth']+1)
                   if r['versions'][v][i] != r['baseline_versions'][v][i]]
        assert changed == ([(r['edited_variable'],r['version_index'])] if r['pair_direction'] else [])
    with pytest.raises(ValueError,match='incomplete'):
        audit(rows[:-1])
    assert len({r['history_id'] for r in rows}) == 16
    refs = {r['history_id']:r for r in rows}
    assert len({signature(r) for r in refs.values()}) == 16
    altered = copy.deepcopy(rows)
    altered[0]['versions']['x'][0] = 'changed'
    with pytest.raises(ValueError,match='edit changes'):
        audit(altered)


def test_new_histories_exclude_previous_histories_and_require_distinct_deep_values():
    pilot = generate('pilot',4,(1,2,4,8),VALUES,102)
    seen = {signature(r) for r in pilot}
    full = generate('full',4,(1,2,4,8),VALUES,103,seen)
    assert not seen & {signature(r) for r in full}
    with pytest.raises(ValueError,match='19'):
        generate('pilot',4,(8,),VALUES[:12],104)


def test_token_audit_locates_declared_single_assignment_token():
    class Tokenizer:
        chat_template = None
        def __init__(self):
            self.ids = {v:i for i,v in enumerate(VALUES)}
        def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
            matches = list(re.finditer(r'\S+',text))
            for m in matches:
                self.ids.setdefault(m.group(),len(self.ids))
            out = {'input_ids':[self.ids[m.group()] for m in matches]}
            if return_offsets_mapping:
                out['offset_mapping'] = [m.span() for m in matches]
            return out
    tok = Tokenizer()
    rows = generate('pilot',4,(1,),VALUES,105)
    report = audit_tokens(rows,tok,{v:i for i,v in enumerate(VALUES)},False)
    assert report['exact_one_input_token_edit'] and report['pairs_checked'] == 32


def test_R_sign_symmetric_means_and_age_ordering():
    rows = generate('pilot',4,(1,2),VALUES,106)
    scores = scored(rows)
    effects, rs = relevance(rows,scores)
    assert all(r['R'] == (10. if r['age_from_current'] == 0 else 0.) for r in rs)
    assert {r['axis'] for r in rs} == {'x','z','symmetric'}
    summary, contrasts = summarize(rs,draws=20,seed=7)
    assert all(r['mean'] == 10. for r in contrasts if r['contrast']=='current_minus_previous')
    assert sorted((r['age_from_current'],r['version_index']) for r in summary if r['depth']==2 and r['axis']=='symmetric') == [(0,2),(1,1),(2,0)]
    base = next(s for s in scores if s['pair_direction']==0 and s['version_index']==s['depth'] and s['query']==s['edited_variable'])
    edit = next(s for s in scores if s['pair_id']==base['pair_id'] and s['pair_direction']==1)
    assert identity_transfer(base,edit) == 10.
    with pytest.raises(ValueError):
        identity_transfer(edit,base)


def test_depth_specific_competence_caps_at_first_failure_without_filtering_scores():
    rows = generate('pilot',4,(1,2,4),VALUES,107)
    scores = scored(rows)
    report = competence(rows,scores)
    assert report['qualified_depths'] == [1,2,4]
    assert len(report['depths']) == 3
    assert all(d['baseline_query_count']==8 for d in report['depths'])
    bad = next(s for s in scores if s['depth']==2 and s['pair_direction']==1)
    bad.update(full_vocab_next_token_accuracy=0,accuracy=0,full_vocab_rank=2,candidate_rank=2)
    assert competence(rows,scores)['qualified_depths'] == [1]
    assert len(relevance(rows,scores)[1]) == len(relevance(rows,scored(rows))[1])


def test_duplicated_concrete_history_rejected():
    rows = generate('pilot',4,(1,),VALUES,108)
    first = [copy.deepcopy(r) for r in rows if r['history_index']==0]
    for r in first:
        r['history_id'] += '_duplicate'
        r['pair_id'] += '_duplicate'
        r['example_id'] += '_duplicate'
    with pytest.raises(ValueError,match='duplicated concrete'):
        audit(rows+first)


def test_cli_checkpoint_analysis_and_full_generation_require_recomputed_pilot(tmp_path,monkeypatch):
    from src.data.io import read_jsonl, sha256_file
    from src.data.version_chain import template_hash, render
    from scripts.generate_version_chain import main as generate_main, verified_pilot
    import scripts.run_version_chain as runner
    from scripts.analyze_version_chain import main as analyze_main
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
    dataset = tmp_path/'version_chain_pilot.jsonl'
    score_path = tmp_path/'version_chain_pilot_scores.jsonl'
    def invoke(main,args):
        monkeypatch.setattr('sys.argv',['script']+args)
        main()
    common = ['--config',str(config),'--token-ids',str(tokens)]
    invoke(generate_main,common+['--stage','pilot','--depths','1,2','--n','4','--seed','110','--output',str(dataset)])
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
    assert json.loads(report.read_text())['competence']['qualified_depths']==[1,2]
    assert (analysis_dir/'R_by_age.png').exists()
    full = tmp_path/'version_chain_full.jsonl'
    full_args = common+['--stage','full','--n','4','--seed','111','--output',str(full)]
    with pytest.raises(ValueError,match='requires --pilot-audit'):
        invoke(generate_main,full_args)
    invoke(generate_main,full_args+['--pilot-audit',str(report)])
    assert {r['depth'] for r in read_jsonl(full)}=={1,2}
    assert not {signature(r) for r in read_jsonl(full)} & {signature(r) for r in read_jsonl(dataset)}
    forged = json.loads(report.read_text())
    forged['competence']['qualified_depths'].append(8)
    report.write_text(json.dumps(forged))
    with pytest.raises(ValueError,match='recomputation'):
        verified_pilot(report,config,tokens)
