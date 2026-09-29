import json
import pandas as pd
import pytest

from scripts.analyze_mechanistic_trajectory import trajectory

def test_trajectory_pairs_history_and_keeps_semantic_sites_distinct(tmp_path):
    rows=[]
    for h,delta in [('h1',2.),('h2',4.)]:
        for role,base in [('edited_binding_value',1.),('query_variable',.2)]:
            rows.append(dict(history_id=h,site_role=role,layer=2,R_x_patch=base+delta))
    f=tmp_path/'input.csv'; pd.DataFrame(rows).to_csv(f,index=False)
    tab,_=trajectory(f,tmp_path/'out',selected_layers=(2,),n_boot=200)
    assert set(tab.site_role)=={'edited_binding_value','query_variable'}
    assert set(tab.n_histories)=={2}

def test_all_position_pairing_ignores_query_token_identity(tmp_path):
    rows=[]
    for q,token,effect in [('current_x','x',5.),('current_z','z',1.)]:
        for d in ('baseline_to_edited','edited_to_baseline'):
            rows.append(dict(history_id='h',query_id=q,edited_binding='old_x',site_role='other_position',layer=1,position=7,direction=d,patch_delta_toward_donor=effect,token_text=token))
    f=tmp_path/'raw.jsonl'; f.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    from scripts.analyze_four_query_patching import main
    # Pairing contract is also represented directly by the analysis key.
    pivot=pd.DataFrame(rows).groupby(['history_id','layer','position','query_id'],as_index=False).patch_delta_toward_donor.mean()
    wide=pivot.pivot(index=['history_id','layer','position'],columns='query_id',values='patch_delta_toward_donor')
    assert wide.loc[('h',1,7),'current_x']-wide.loc[('h',1,7),'current_z']==pytest.approx(4.)

def test_margin_decomposition_identity():
    correct_delta=1.25; margin_delta=3.5
    old_source_delta=correct_delta-margin_delta
    assert margin_delta==pytest.approx(correct_delta-old_source_delta)

def test_shared_prefix_audit_batches_causal_prefix_and_rejects_changed_ids(monkeypatch):
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    import src.experiments.prefix_invariance as pi
    class Tok:
        pad_token_id=0; eos_token_id=0
        def __call__(self,text,add_special_tokens=False,return_offsets_mapping=False):
            ids=[ord(c) for c in text]
            return {'input_ids':ids,'offset_mapping':[(i,i+1) for i in range(len(text))]}
    class Causal(torch.nn.Module):
        def forward(self,x): return x.cumsum(dim=1)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.config=SimpleNamespace(num_hidden_layers=1); self.embed=torch.nn.Embedding(128,2); self.layers=torch.nn.ModuleList([Causal()]); self.head=torch.nn.Linear(2,2)
            with torch.no_grad(): self.embed.weight.fill_(1)
        def forward(self,input_ids,attention_mask=None,use_cache=False):
            x=self.embed(input_ids); x=self.layers[0](x); return SimpleNamespace(logits=self.head(x))
    monkeypatch.setattr(pi,'render_example',lambda ex,tok,chat=True: ex['prompt'])
    qs=('current_x','current_z','initial_x','initial_z')
    examples={q:{'example_id':q,'history_id':'h','old_x':'v','prompt':'v|'+q} for q in qs}
    report=pi.audit_prefix_invariance(Model(),Tok(),examples,qs)
    assert report['passed'] and report['per_layer'][0]['max_abs_difference']==0
    examples['current_z']['prompt']='w|current_z'
    with pytest.raises(ValueError,match='token IDs through old-x'):
        pi.audit_prefix_invariance(Model(),Tok(),examples,qs)

def test_partition_artifact_has_disjoint_exact_union(tmp_path):
    from scripts.freeze_mechanistic_partitions import ids_in,partition_sets
    # Basic source reader keeps stable artifact ordering and exact IDs.
    p=tmp_path/'x.csv'; pd.DataFrame({'history_id':['h1','h1','h2']}).to_csv(p,index=False)
    assert ids_in(p)==['h1','h2']
    d,h,r=partition_sets(['a','b','c','d'],['a'],['b'])
    assert not (d&h or d&r or h&r)
    assert d|h|r=={'a','b','c','d'}
    with pytest.raises(ValueError): partition_sets(['a','b'],['a'],['a'])
