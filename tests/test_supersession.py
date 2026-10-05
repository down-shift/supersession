import json
import pytest
from src.data.supersession import (make_control_history, render_control_example, make_status_history,
    render_status_example, status_counterfactual_pair, make_version_chain, version_edit_pair)

VALUES=['amber','birch','coral','denim','elm','frost','grape','hazel','indigo','jade','khaki','lilac','maple','navy','ochre','pearl','quartz','ruby']

def test_control_roles_are_explicit_and_rendered():
    live=make_control_history('a',VALUES,'live',3); stale=make_control_history('b',VALUES,'superseded',3); irrelevant=make_control_history('c',VALUES,'irrelevant',3)
    assert live['semantic_status']=='live_current' and live['answer']==live['source_value']
    assert stale['semantic_status']=='superseded_initial' and stale['old_x']==stale['source_value'] and stale['answer']!=stale['old_x']
    assert irrelevant['semantic_status']=='irrelevant_occurrence' and irrelevant['source_value'] in irrelevant['context_lines'][-2]
    assert 'what is x?' in render_control_example(live).lower()

def test_yes_no_semantics_and_matched_prompt_structure():
    yes,no=make_status_history('h',VALUES,8)
    assert yes['update_accepted'] and yes['current_x']==yes['proposed_x'] and yes['semantic_status']['proposed_x']=='accepted_current'
    assert not no['update_accepted'] and no['current_x']==no['initial_x'] and no['semantic_status']['proposed_x']=='rejected_update'
    py,pn=render_status_example(yes),render_status_example(no)
    assert py.replace('YES','STATUS')==pn.replace('NO','STATUS')
    for history,field in ((yes,'initial_x'),(yes,'proposed_x'),(no,'initial_x'),(no,'proposed_x')):
        base,edit=status_counterfactual_pair(history,field)
        assert [k for k in ('initial_x','initial_z','proposed_x','proposed_z') if base[k]!=edit[k]]==[field]

def test_version_depth_metadata_and_single_version_edits():
    h=make_version_chain('chain',VALUES,4,11)
    assert len(h['x_versions'])==len(h['z_versions'])==5
    assert h['current_x']==h['x_versions'][-1]
    assert h['version_metadata'][h['x_versions'][0]]['distance_from_current']==4
    replacement=next(v for v in VALUES if v not in h['x_versions']+h['z_versions'])
    edit=version_edit_pair(h,'x',2,replacement)
    assert [i for i,(a,b) in enumerate(zip(h['x_versions'],edit['edited']['x_versions'])) if a!=b]==[2]
    assert edit['edited']['version_metadata'][replacement]['version_index']==2
    with pytest.raises(ValueError): make_version_chain('bad',VALUES,3)

def test_reserve_split_is_deterministic_exact_cover_and_immutable(tmp_path):
    from scripts.freeze_head_reserve import build_document
    ids=[f'h{i}' for i in range(168)]
    parent={'history_ids':{'unused_mechanistic_reserve':ids}}
    one=build_document(parent); two=build_document(parent)
    assert one==two and one['counts']=={'head_confirmation':72,'path_confirmation':48,'final_validation':48}
    sets=[set(v) for v in one['history_ids'].values()]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3)) and set.union(*sets)==set(ids)
    from scripts.freeze_head_reserve import main

def test_gqa_query_to_kv_mapping():
    pytest.importorskip("torch")  # the module imports torch at load time
    from src.experiments.qkv_interventions import query_to_kv_head
    assert [query_to_kv_head(h,8,2) for h in range(8)]==[0,0,0,0,1,1,1,1]
    with pytest.raises(ValueError): query_to_kv_head(8,8,2)

def test_identity_transfer_and_current_stale_margin_orientation():
    torch = pytest.importorskip("torch")
    from src.experiments.patching import identity_transfer_effect,version_selection_diagnostics
    ids={'amber':0,'birch':1,'coral':2}; base=torch.tensor([4.,0.,0.]); edit=torch.tensor([1.,0.,5.])
    assert identity_transfer_effect(base,edit,'amber','coral',ids)['identity_transfer_E']==pytest.approx(8.)
    diag=version_selection_diagnostics(base,edit,'amber','coral',ids)
    assert diag['baseline_current_minus_old']==pytest.approx(4.)
    assert diag['patched_current_minus_old']==pytest.approx(-4.)
    assert diag['patch_delta_current_minus_old']==pytest.approx(-8.)

def test_projection_hook_changes_only_requested_head_and_position():
    torch=pytest.importorskip('torch')
    from src.experiments.qkv_interventions import ProjectionHook
    proj=torch.nn.Linear(2,4,bias=False); x=torch.zeros(1,3,2); source=torch.ones(1,1,2,2)
    baseline=proj(x).detach().reshape(1,3,2,2); hook=ProjectionHook(proj,2,2,1,source=source,head_ids=[1]); observed={}
    handle=proj.register_forward_hook(lambda m,i,o: observed.setdefault('out',o.detach().clone()))
    try: proj(x)
    finally: hook.close(); handle.remove()
    after=observed['out'].reshape(1,3,2,2)
    assert torch.equal(after[:,0],baseline[:,0]) and torch.equal(after[:,2],baseline[:,2])
    assert torch.equal(after[:,1,0],baseline[:,1,0]) and torch.equal(after[:,1,1],source[:,0,1])
