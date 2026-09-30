import pytest


def _toy_qwen3(torch,model_type='qwen3'):
    from types import SimpleNamespace
    from src.experiments.attention_head_patching import capture_head_layers
    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.head_dim=2; self.num_key_value_groups=2
            self.q_proj=torch.nn.Linear(4,8,bias=False); self.k_proj=torch.nn.Linear(4,4,bias=False)
            self.o_proj=torch.nn.Linear(8,4,bias=True)
        def forward(self,x):
            heads=self.q_proj(x).reshape(x.shape[0],x.shape[1],4,2)
            return (self.o_proj(heads.reshape(x.shape[0],x.shape[1],8)),None)
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.self_attn=Attention(); self.mlp=torch.nn.Identity(); self.input_layernorm=torch.nn.Identity(); self.post_attention_layernorm=torch.nn.Identity()
        def forward(self,hidden_states,**kwargs):
            residual=hidden_states; contribution=self.self_attn(self.input_layernorm(hidden_states))[0]
            hidden_states=residual+contribution; return hidden_states+self.mlp(self.post_attention_layernorm(hidden_states))
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.config=SimpleNamespace(model_type=model_type,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2)
            self.embed=torch.nn.Embedding(3,4); self.layers=torch.nn.ModuleList([Block()]); self.head=torch.nn.Linear(4,3,bias=False)
        def forward(self,input_ids,use_cache=False):
            x=self.embed(input_ids)
            for layer in self.layers: x=layer(x)
            return SimpleNamespace(logits=self.head(x))
    return Model().eval()


def test_single_head_hook_changes_only_selected_query_head_and_position():
    torch=pytest.importorskip('torch')
    from src.experiments.attention_head_patching import (HeadInputHook,capture_head_layers,
        validate_qwen3_attention_heads)
    torch.manual_seed(17); model=_toy_qwen3(torch)
    source={'input_ids':torch.tensor([[0,1]])}; target={'input_ids':torch.tensor([[0,2]])}; pos=1; layer=0; head=3
    _,source_heads=capture_head_layers(model,source,[layer]); _,target_heads=capture_head_layers(model,target,[layer])
    _,dims=validate_qwen3_attention_heads(model); qheads,_,head_dim=dims[layer]
    observed={}
    patch=HeadInputHook(model.layers[layer].self_attn,qheads,head_dim,source_heads[layer],pos,[(head,)])
    def record_input(_module,args): observed['input']=args[0].detach().clone()
    monitor=model.layers[layer].self_attn.o_proj.register_forward_pre_hook(record_input)
    try: model(**target)
    finally: patch.close(); monitor.remove()
    before=target_heads[layer]; after=observed['input'].reshape_as(before)
    changed=(after-before).abs().sum(dim=-1)>1e-7
    expected=torch.zeros_like(changed); expected[:,pos,head]=True
    assert torch.equal(changed,expected)
    assert torch.equal(after[:,pos,head],source_heads[layer][:,pos,head])


def test_joint_all_heads_equals_whole_attention_output_patch():
    torch=pytest.importorskip('torch')
    from src.experiments.attention_head_patching import capture_head_layers,patch_head_set_logits
    from src.experiments.component_patching import capture_component,patch_component_logits
    torch.manual_seed(23); model=_toy_qwen3(torch)
    source={'input_ids':torch.tensor([[0,1]])}; target={'input_ids':torch.tensor([[0,2]])}; position=1
    _,head_source=capture_head_layers(model,source,[0])
    _,attention_source=capture_component(model,source,0,'attention_output')
    joint=patch_head_set_logits(model,target,{0:head_source[0]},{0:[0,1,2,3]},position)
    whole=patch_component_logits(model,target,0,'attention_output',attention_source,position)
    assert torch.allclose(joint,whole,atol=1e-6,rtol=1e-6)


def test_head_indexing_uses_query_heads_under_gqa_and_rejects_unsupported_models():
    torch=pytest.importorskip('torch')
    from src.experiments.attention_head_patching import validate_qwen3_attention_heads,validate_head_sets
    model=_toy_qwen3(torch); _,dims=validate_qwen3_attention_heads(model)
    assert dims==[(4,2,2)]
    assert validate_head_sets({0:[3]},dims)=={0:(3,)}
    with pytest.raises(ValueError,match='outside 0..3'):
        validate_head_sets({0:[4]},dims)
    unsupported=_toy_qwen3(torch,model_type='llama')
    with pytest.raises(RuntimeError,match="expected config.model_type='qwen3'"):
        validate_qwen3_attention_heads(unsupported)


def test_pair_direction_examples_never_unpacks_mapping_keys():
    from src.experiments.attention_head_patching import pair_direction_examples
    baseline={'example_id':'baseline'}; edited={'example_id':'edited'}
    assert pair_direction_examples({0:baseline,1:edited})==(baseline,edited)
    with pytest.raises(ValueError,match='pair_direction 0 and 1'):
        pair_direction_examples({0:baseline,1:edited,2:{}})


def test_donor_oriented_patch_effect_is_positive_toward_donor_in_both_directions():
    import numpy as np
    from src.experiments.attention_head_patching import donor_oriented_margin_effect
    ids={'amber':0,'birch':1}
    forward=donor_oriented_margin_effect(np.array([4.,0.]),np.array([0.,4.]),np.array([3.,1.]),'amber','birch',ids)
    reverse=donor_oriented_margin_effect(np.array([0.,4.]),np.array([4.,0.]),np.array([1.,3.]),'birch','amber',ids)
    assert forward['patch_delta_toward_donor']==pytest.approx(6.)
    assert reverse['patch_delta_toward_donor']==pytest.approx(6.)


def test_reserve_head_set_reports_x_z_and_symmetric_history_contrasts():
    from scripts.analyze_attention_head_patching import reserve_summary
    rows=[]
    cells={('old_x','current_x'):3.,('old_x','current_z'):1.,('old_z','current_z'):4.,('old_z','current_x'):0.}
    for history in ('h1','h2'):
        for (binding,query),effect in cells.items():
            for direction in ('baseline_to_edited','edited_to_baseline'):
                rows.append({'history_id':history,'head_set':[[32,4],[34,7]],'edited_binding':binding,'query_id':query,'direction':direction,'patch_delta_toward_donor':effect})
    per_history,summary=reserve_summary(rows,n_boot=1000)
    assert set(per_history.R_x)=={2.}
    assert set(per_history.R_z)=={4.}
    assert set(per_history.symmetric_R)=={3.}
    assert summary.iloc[0].symmetric_R_mean==pytest.approx(3.)
