"""Cheap protocol, likelihood, provenance, stage, and real tiny-architecture tests."""
import copy
import json
import math
from types import SimpleNamespace

import pytest

from src.cross_model.dataset import generate
from src.cross_model.protocol import (CONTRACT, VALUES, digest, disjoint, evaluate, read_sealed,
                                       sealed, validate_dataset, write_new)
from src.cross_model.scoring import logsumexp
from src.cross_model.tokens import continuations


def scores_for(rows):
    result = []
    for r in rows:
        masses = {v: (-.1 if v == r['answer'] else -10.) for v in VALUES}
        # Prompt signature here is a deterministic semantic rendering independent of metadata.
        from src.data.supersession_behavior import render_behavior_example
        result.append({**r,'prompt':render_behavior_example(r, None, False),
            'score_kind':'competence_only','semantic_log_mass':masses,
            'full_vocab_next_token_accuracy':0, 'generated_first_token':r['answer'].title(), 'full_vocab_rank':2})
    return result


def test_semantic_gate_preserves_surface_failure_as_diagnostic():
    rows = generate('frozen_gate',24); validate_dataset(rows,'frozen_gate')
    result = evaluate(rows,scores_for(rows))
    assert result['pass'] and result['expected_cell_count']==64
    assert all(v['exact_token_accuracy_diagnostic']==0 for v in result['cells'].values())


@pytest.mark.parametrize('failure',['missing','duplicate','metadata','nan','causal','tie','margin'])
def test_gate_fails_closed(failure):
    rows = generate('frozen_gate',24); scores = scores_for(rows)
    if failure=='missing': scores.pop()
    elif failure=='duplicate': scores.append(scores[0])
    elif failure=='metadata': scores[0]['answer']='bad'
    elif failure=='nan': scores[0]['semantic_log_mass'][VALUES[0]]=math.nan
    elif failure=='causal': scores[0]['matched_edit_effect']={}
    elif failure=='tie':
        for s in scores: s['semantic_log_mass']={v:0. for v in VALUES}
    elif failure=='margin':
        for s in scores:
            if s['stale_value']: s['semantic_log_mass'][s['stale_value']]=1.
    if failure in ('tie','margin'): assert not evaluate(rows,scores)['pass']
    else:
        with pytest.raises(ValueError): evaluate(rows,scores)


def test_missing_required_cell():
    rows = [r for r in generate('frozen_gate',24) if r['orientation']==0]
    with pytest.raises(ValueError,match='incomplete'):evaluate(rows,scores_for(rows))


def test_seal_and_no_overwrite(tmp_path):
    p=tmp_path/'artifact.json';write_new(p,sealed({'pass':True}))
    assert read_sealed(p)=={'pass':True}
    with pytest.raises(FileExistsError):write_new(p,{})
    p.write_text(json.dumps({'pass':False,'seal':'bogus'}))
    with pytest.raises(ValueError,match='seal'):read_sealed(p)


def test_history_overlap_includes_values_and_entities(tmp_path):
    rows=generate('development',24);p=tmp_path/'dev.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError,match='overlap'):disjoint(rows,[p])
    assert disjoint(generate('frozen_gate',24),[p])


def test_fixed_stage_count_seed_vocabulary():
    for kind,change in [('count',lambda r:r[:40]),('seed',lambda r:[{**x,'seed':0} for x in r]),
                        ('vocabulary',lambda r:[{**x,'candidate_values':list(reversed(VALUES))} for x in r])]:
        with pytest.raises(ValueError):validate_dataset(change(generate('development',24)),'development')


def test_logsumexp_is_mass_not_mean_or_max():
    assert logsumexp([-2.,-2.])==pytest.approx(-2+math.log(2))
    with pytest.raises(ValueError):logsumexp([math.nan])


class CharTokenizer:
    def __call__(self,text,**kwargs):return {'input_ids':[ord(c) for c in text]}


def test_surface_events_multi_token_and_case():
    result=continuations(CharTokenizer(),'Answer:')
    assert [e['text'] for e in result['jade']]==[' jade',' Jade','jade','Jade']
    assert result['jade'][0]['ids']==list(map(ord,' jade'))


def test_collisions_rejected():
    class Colliding:
        def __call__(self,text,**kw):return {'input_ids':[0] if text=='Answer:' else [0,1]}
    with pytest.raises(ValueError,match='collision'):continuations(Colliding(),'Answer:')


def test_complete_sequence_likelihood_and_teacher_forcing():
    torch=pytest.importorskip('torch')
    from src.cross_model.scoring import score_prompt
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.embedding=torch.nn.Embedding(128,4)
        def get_input_embeddings(self):return self.embedding
        def forward(self,input_ids,use_cache=False):
            # Uniform logits make exact probability product analytically known.
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape,128))
    model=Tiny();events={'rust':[{'text':' rust','ids':[32,114,117,115,116]}]}
    masses,details,_,_=score_prompt(model,CharTokenizer(),'A:',events)
    assert masses['rust']==pytest.approx(-5*math.log(128),abs=1e-5)
    assert details['rust'][0]['ids']==[32,114,117,115,116]


@pytest.mark.parametrize('family',['qwen3','llama','mistral','phi3'])
def test_native_tiny_architecture_hooks(family):
    torch=pytest.importorskip('torch');transformers=pytest.importorskip('transformers')
    from src.cross_model.adapters import ActivationHook, get_decoder_blocks, normalized_depth, head_dimensions
    classes={'qwen3':('Qwen3Config','Qwen3ForCausalLM'),'llama':('LlamaConfig','LlamaForCausalLM'),
             'mistral':('MistralConfig','MistralForCausalLM'),'phi3':('Phi3Config','Phi3ForCausalLM')}
    cfg_cls,model_cls=classes[family]
    cfg=getattr(transformers,cfg_cls)(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        max_position_embeddings=64,original_max_position_embeddings=64,pad_token_id=0)
    cfg._attn_implementation='eager';torch.manual_seed(123)
    model=getattr(transformers,model_cls)(cfg).eval();inp=torch.tensor([[1,2,3,4]])
    assert len(get_decoder_blocks(model))==3
    assert head_dimensions(model,0)==(4,2,8)
    assert normalized_depth(1,3)==.5
    baseline=model(input_ids=inp,use_cache=False).logits.detach()
    for component in ['block_output','attention_output','mlp_output','query_head']:
        kwargs={'head':1} if component=='query_head' else {}
        with ActivationHook(model,1,component,[1,2],**kwargs) as capture:
            out=model(input_ids=inp,use_cache=False).logits.detach()
        assert capture.activation is not None
        assert torch.equal(out,baseline)
        with ActivationHook(model,1,component,[1,2],source=capture.activation,**kwargs):
            same=model(input_ids=inp,use_cache=False).logits.detach()
        assert torch.equal(same,baseline)
        with ActivationHook(model,1,component,[1,2],source=torch.zeros_like(capture.activation),**kwargs):
            changed=model(input_ids=inp,use_cache=False).logits.detach()
        assert not torch.equal(changed,baseline)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_gate_recomputation_rejects_forged_pass(tmp_path,monkeypatch):
    from src.cross_model import workflow
    path=tmp_path/'gate.json'
    gate={'stage':'frozen_gate','contract':CONTRACT,'evaluation':{'pass':True},
          'development_report_path':str(tmp_path/'dev.json'),'development_report_sha256':'devhash',
          'dataset_path':'data','scores_path':'scores','provenance':{}}
    write_new(path,sealed(gate))
    monkeypatch.setattr(workflow,'sha256_file',lambda _: 'devhash')
    monkeypatch.setattr(workflow,'check_manifest',lambda *args:None)
    monkeypatch.setattr(workflow,'gate_report',lambda *args:{**gate,'evaluation':{'pass':False}})
    with pytest.raises(ValueError,match='failed'):workflow.verify_gate(path,{},'config','candidates')


def test_resume_fingerprint_and_duplicate_ids(tmp_path):
    from src.data.progress import prepare_jsonl_progress
    d=tmp_path/'data';v=tmp_path/'map';o=tmp_path/'out';d.write_text('data');v.write_text('map')
    rows=[{'example_id':'one'}]
    prepare_jsonl_progress(o,d,v,{'code':'one'},rows)
    with pytest.raises(FileExistsError):prepare_jsonl_progress(o,d,v,{},rows)
    with pytest.raises(ValueError,match='fingerprint'):prepare_jsonl_progress(o,d,v,{'code':'two'},rows,resume=True)
    o.write_text('{"example_id":"one"}\n{"example_id":"one"}\n')
    with pytest.raises(ValueError,match='duplicate'):prepare_jsonl_progress(o,d,v,{'code':'one'},rows,resume=True)


def test_normalized_effect_unavailable_at_small_denominator():
    from src.cross_model.analysis import summarize
    rows=[{'history_id':str(i),'R_live':.1,'R_live_minus_R_irrelevant_counterbalanced':.1,'R_superseded_minus_R_irrelevant_counterbalanced':1.} for i in range(24)]
    assert not summarize(rows)['normalized_primary_relative_to_live_minus_irrelevant']['available']


def test_hash_bound_stage_chain_and_changed_raw_scores(tmp_path):
    from src.cross_model.protocol import manifest, history_signatures
    from src.cross_model.workflow import gate_report, verify_gate
    from src.data.io import sha256_file
    from src.utils import load_config
    config_path='configs/cross_model_v1/qwen3_8b.yaml';config=load_config(config_path)
    candidate=tmp_path/'candidate.json';write_new(candidate,sealed({'events':{}}))
    def stage_files(stage,development=None):
        rows=generate(stage,24);data=tmp_path/(stage+'.jsonl');scores=tmp_path/(stage+'_scores.jsonl')
        data.write_text(''.join(json.dumps(r)+'\n' for r in rows));scored=scores_for(rows)
        if stage=='development':
            for s in scored:s['semantic_log_mass']={v:0. for v in VALUES}
        scores.write_text(''.join(json.dumps(s)+'\n' for s in scored))
        info={'stage':stage,'history_signatures':sorted(history_signatures(rows)),
              'prior_datasets':[], 'provenance':manifest(config,config_path,candidate,data)}
        if development:
            info['development_report_sha256']=sha256_file(development)
            info['prior_datasets']=[{'path':str(tmp_path/'development.jsonl'),'sha256':sha256_file(tmp_path/'development.jsonl')}]
        write_new(str(data)+'.provenance.json',sealed(info))
        write_new(str(scores)+'.provenance.json',sealed({'stage':stage,'scores_sha256':sha256_file(scores),
                  'provenance':manifest(config,config_path,candidate,data)}))
        return data,scores,rows,scored
    dd,ds,dr,dsc=stage_files('development');dev=tmp_path/'dev_report.json'
    write_new(dev,sealed({'stage':'development','dataset_path':str(dd),'scores_path':str(ds),
        'dataset_sha256':sha256_file(dd),'scores_sha256':sha256_file(ds),'evaluation':evaluate(dr,dsc,gate=False)}))
    gd,gs,_,_=stage_files('frozen_gate',dev);g=tmp_path/'gate.json'
    report=gate_report(config,config_path,candidate,gd,gs,dev);write_new(g,sealed(report))
    assert verify_gate(g,config,config_path,candidate)['evaluation']['pass']
    changed=json.loads(gs.read_text().splitlines()[0]);changed['semantic_log_mass'][changed['answer']]=-100
    gs.write_text(json.dumps(changed)+'\n'+'\n'.join(gs.read_text().splitlines()[1:])+'\n')
    with pytest.raises(ValueError,match='hash'):verify_gate(g,config,config_path,candidate)


def test_prompt_edit_audit_rejects_nonlocal_changes():
    from src.cross_model.tokens import audit_pairs
    rows=generate('validation',4)
    class BadTokenizer(CharTokenizer):
        def __call__(self,text,**kwargs):
            result={'input_ids':[ord(c) for c in text]}
            if kwargs.get('return_offsets_mapping'):result['offset_mapping']=[(i,i+1) for i in range(len(text))]
            # Introduce a context token change when a counterfactual value occurs.
            if rows[1]['replacement_value'] in text:result['input_ids'][0]=999
            return result
    # Isolate a pair where replacement is absent in the baseline.
    with pytest.raises(ValueError,match='outside'):audit_pairs(rows[:2],BadTokenizer())


def test_mechanistic_contrasts_require_complete_queries():
    from src.cross_model.mechanism import relevance
    r={'history_id':'h','condition':'superseded','edited_field':'initial_x','edited_variable':'x',
       'query':'x','slot_order':None,'component':'block_output','site':'final_preanswer','layer':0,
       'head':None,'donor_oriented_patch_delta':1.}
    with pytest.raises(ValueError,match='incomplete'):relevance([r])


def test_stage_claim_prevents_retry_in_fresh_path(tmp_path,monkeypatch):
    from scripts.cross_model import claim_stage
    monkeypatch.chdir(tmp_path)
    c={'model':{'id':'family/model','revision':'a'*40}}
    claim_stage(c,'development','first.jsonl')
    with pytest.raises(FileExistsError):claim_stage(c,'development','second.jsonl')
    claim_stage(c,'frozen_gate','gate.jsonl')


def test_provenance_rejects_changed_config_candidate_and_renderer(tmp_path):
    from src.cross_model.protocol import manifest,check_manifest
    from src.utils import load_config
    config_path='configs/cross_model_v1/qwen3_8b.yaml';c=load_config(config_path)
    candidate=tmp_path/'candidate';candidate.write_text('one')
    saved=manifest(c,config_path,candidate)
    check_manifest(saved,c,config_path,candidate)
    candidate.write_text('two')
    with pytest.raises(ValueError,match='candidate_map'):check_manifest(saved,c,config_path,candidate)
    candidate.write_text('one');changed={**saved,'renderer_sha256':'bad'}
    with pytest.raises(ValueError,match='renderer'):check_manifest(changed,c,config_path,candidate)
    changed={**saved,'config_sha256':'bad'}
    with pytest.raises(ValueError,match='config_sha256'):check_manifest(changed,c,config_path,candidate)


@pytest.mark.parametrize('family',['qwen3','llama','mistral','phi3'])
def test_runtime_smoke_on_tiny_native_model(family):
    transformers=pytest.importorskip('transformers')
    from src.cross_model.runtime import hook_smoke
    names={'qwen3':('Qwen3Config','Qwen3ForCausalLM'),'llama':('LlamaConfig','LlamaForCausalLM'),
           'mistral':('MistralConfig','MistralForCausalLM'),'phi3':('Phi3Config','Phi3ForCausalLM')}
    cfg_name,model_name=names[family]
    cfg=getattr(transformers,cfg_name)(vocab_size=128,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        max_position_embeddings=64,original_max_position_embeddings=64,pad_token_id=0)
    cfg._attn_implementation='eager'
    result=hook_smoke(getattr(transformers,model_name)(cfg).eval(),CharTokenizer(),'A:')
    assert result['status']=='passed'
    assert result['within_block_order']==['input_norm','attention','post_attention_norm','mlp','block']
    assert set(result['activation_dtypes'].values())=={'torch.float32'}


def test_fresh_run_rejects_orphan_analysis_before_model_load(tmp_path):
    from scripts.cross_model import fresh_bundle
    output=tmp_path/'mechanism.jsonl'
    (tmp_path/'mechanism.jsonl.analysis.json').write_text('{}')
    with pytest.raises(FileExistsError):fresh_bundle(output)


def test_one_semantic_error_does_not_fail_tiny_cells_gate():
    rows=generate('frozen_gate',24);scores=scores_for(rows)
    prompt=next(s['prompt'] for s in scores if s['condition']=='superseded' and s['pair_direction']==0)
    for s in scores:
        if s['prompt']==prompt:
            wrong=next(v for v in VALUES if v!=s['answer'])
            s['semantic_log_mass'][s['answer']]=-20.  # Rank eight: one error still fits the 99% aggregate rule.
    result=evaluate(rows,scores)
    assert result['pass']
    assert result['condition_summary']['superseded']['semantic_accuracy']<1
    assert any(c['semantic_accuracy']<.99 for c in result['cells'].values())


def test_development_is_descriptive_even_when_all_semantic_answers_tie():
    rows=generate('development',24);scores=scores_for(rows)
    for s in scores:s['semantic_log_mass']={v:0. for v in VALUES}
    result=evaluate(rows,scores,gate=False)
    assert result['pass'] is None
    assert result['eligibility_decision']=='descriptive_only'
    assert not result['failed_conditions']
    assert result['condition_summary']['live']['semantic_accuracy']==0


def test_surface_geometry_preserves_deduplicated_aliases_and_unequal_classes():
    from src.cross_model.tokens import surface_geometry_audit
    class AliasTokenizer(CharTokenizer):
        def __call__(self,text,**kw):
            if text.startswith('Answer:'):
                suffix=text[len('Answer:'):]
                if suffix in (' jade','jade'):text='Answer:jade'
            return super().__call__(text,**kw)
    events=continuations(AliasTokenizer(),'Answer:')
    table=surface_geometry_audit(events)
    assert table['by_value']['jade']['distinct_event_count']==3
    assert table['by_value']['coral']['distinct_event_count']==4
    assert not table['equal_event_counts']
    jade=next(e for e in table['by_value']['jade']['events'] if ' jade' in e['surface_strings'])
    assert jade['surface_strings']==[' jade','jade']
    assert jade['token_length']==4


def test_restricted_mass_is_identical_and_uses_fewer_patched_forwards():
    torch=pytest.importorskip('torch')
    from src.cross_model.scoring import score_prompt
    class Tiny(torch.nn.Module):
        def __init__(self):super().__init__();self.embedding=torch.nn.Embedding(128,4);self.calls=0
        def get_input_embeddings(self):return self.embedding
        def forward(self,input_ids,use_cache=False):
            self.calls+=1
            # Nonuniform conditional logits depend on the previous input token.
            grid=torch.arange(128).reshape(1,1,128)
            logits=-((grid-input_ids.unsqueeze(-1))**2).float()/1000
            return SimpleNamespace(logits=logits)
    model=Tiny();events=continuations(CharTokenizer(),'A:')
    full=score_prompt(model,CharTokenizer(),'A:',events)[0];all_calls=model.calls
    model.calls=0
    small=score_prompt(model,CharTokenizer(),'A:',events,values=['jade','pearl'])[0]
    assert set(small)=={'jade','pearl'}
    assert small=={v:full[v] for v in small}
    assert model.calls<all_calls/3
    with pytest.raises(ValueError):score_prompt(model,CharTokenizer(),'A:',events,values=['missing'])


def test_normalization_uses_irrelevant_corrected_live_and_guard():
    from src.cross_model.analysis import summarize
    rows=[{'history_id':str(i),'R_live':10.,'R_live_minus_R_irrelevant_counterbalanced':8.,
           'R_superseded_minus_R_irrelevant_counterbalanced':4.} for i in range(24)]
    norm=summarize(rows)['normalized_primary_relative_to_live_minus_irrelevant']
    assert norm['ratio_of_means']==.5
    # A large raw live signal must not bypass the corrected denominator guard.
    for r in rows:r['R_live_minus_R_irrelevant_counterbalanced']=.5
    assert not summarize(rows)['normalized_primary_relative_to_live_minus_irrelevant']['available']


def test_core_grid_is_narrow_and_controls_complete(monkeypatch):
    from src.cross_model import mechanism
    rows=generate('confirmatory',96);ids,pairs=mechanism.selected_pairs(rows,include_current=False)
    monkeypatch.setattr(mechanism,'site_map',lambda *args:{s:[0] for s in
        ['historical_value_span','final_preanswer','current_value_span','distractor_value_span','queried_entity']})
    tasks=mechanism.build_tasks(ids,pairs,None,36)
    assert len(tasks)==15552
    all_layer=[t for t in tasks if t[2]=='block_output' and t[3] in ['historical_value_span','final_preanswer']]
    assert len(all_layer)==12*16*36*2
    assert {pairs[t[0]][0]['condition'] for t in all_layer}=={'live','superseded','irrelevant_counterbalanced'}
    assert all(pairs[t[0]][0]['edited_field'].startswith('initial_') for t in tasks)
    assert not any(pairs[t[0]][0]['condition']=='irrelevant' for t in tasks)
    secondary=[t for t in tasks if t[3] in ['current_value_span','distractor_value_span','queried_entity']]
    assert {pairs[t[0]][0]['history_id'] for t in secondary}==set(ids[:4])
    assert {t[1] for t in secondary}=={26,31,35}


def test_patched_pair_only_scoring_matches_full_class_reference(monkeypatch):
    torch=pytest.importorskip('torch');transformers=pytest.importorskip('transformers')
    from src.cross_model import mechanism
    from src.cross_model.scoring import score_prompt
    cfg=transformers.Qwen3Config(vocab_size=128,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    cfg._attn_implementation='eager';torch.manual_seed(77)
    model=transformers.Qwen3ForCausalLM(cfg).eval();tok=CharTokenizer()
    base={'pair_id':'p','history_id':'h','condition':'superseded','edited_field':'initial_x',
          'edited_variable':'x','query':'x','source_value':'jade','replacement_value':'pearl'}
    members={d:{**base,'pair_direction':d} for d in (0,1)}
    monkeypatch.setattr(mechanism,'semantic_positions',lambda r,t:('A:' if r['pair_direction']==0 else 'B:',{}))
    monkeypatch.setattr(mechanism,'site_map',lambda *args:{'edited_value_span':[0]})
    candidate={'events':continuations(tok,'A:')};requested=[]
    def restricted(*args,**kw):
        requested.append(set(kw.get('values',[])))
        return score_prompt(*args,**kw)
    monkeypatch.setattr(mechanism,'score_prompt',restricted)
    small=mechanism.patch_pair(model,tok,members,candidate,0,'block_output','edited_value_span',cache={})
    assert requested and all(v=={'jade','pearl'} for v in requested)
    def full_reference(*args,**kw):
        kw.pop('values',None)
        return score_prompt(*args,**kw)
    monkeypatch.setattr(mechanism,'score_prompt',full_reference)
    full=mechanism.patch_pair(model,tok,members,candidate,0,'block_output','edited_value_span',cache={})
    assert small['directions']==pytest.approx(full['directions'],abs=1e-7)
