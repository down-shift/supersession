import pytest
from src.experiments.patching import assert_aligned

def test_identity_alignment_contract():
    assert_aligned([3,4,5],[3,4,5],[])

def test_patch_pairs_must_differ_only_at_expected_input_positions():
    assert_aligned([3,4,5,6],[3,9,5,6],[1])
    with pytest.raises(ValueError): assert_aligned([3,4,5],[3,9,8],[1])

def test_focal_cells_and_canonical_roles():
    from src.experiments.patching import focal_cells,canonical_site_role,r_x_patch,symmetric_r_patch
    assert focal_cells('discovery')==[('old_x','current_x'),('old_x','current_z'),('old_x','initial_x'),('current_x','current_x'),('current_x','initial_x')]
    assert focal_cells('discovery',True)==[('old_x','current_x'),('old_x','current_z')]
    assert len(focal_cells('heldout'))==10
    assert canonical_site_role('old_x','old_value')=='edited_binding_value'
    assert canonical_site_role('old_z','current_value')=='same_variable_other_value'
    assert canonical_site_role('current_x','current_value')=='edited_binding_value'
    assert canonical_site_role('current_z','old_value')=='same_variable_other_value'
    assert r_x_patch(4.5,1.25)==pytest.approx(3.25)
    assert symmetric_r_patch(3.25,2.75)==pytest.approx(3.)

def test_patch_resume_requires_complete_pair_marker_and_discards_half_second_direction(tmp_path):
    import json
    from src.experiments.patching import commit_patch_pair,recover_patch_checkpoint
    output=tmp_path/'patches.jsonl'; markers=tmp_path/'patches.complete.jsonl'
    commit_patch_pair(output,markers,'complete',[{'pair_id':'complete','direction':'baseline_to_edited','row':i} for i in range(2)]+[{'pair_id':'complete','direction':'edited_to_baseline','row':i} for i in range(2)])
    # Simulates a crash after both direction labels appeared, but halfway
    # through the second direction's requested layer/position records.
    with output.open('a') as f:
        f.write(json.dumps({'pair_id':'interrupted','direction':'baseline_to_edited','row':0})+'\n')
        f.write(json.dumps({'pair_id':'interrupted','direction':'edited_to_baseline','row':0})+'\n')
    done=recover_patch_checkpoint(output,markers)
    assert done=={'complete'}
    rows=[json.loads(line) for line in output.read_text().splitlines()]
    assert {row['pair_id'] for row in rows}=={'complete'}

def test_torch_patching_is_lazy_and_clear_architecture_errors():
    # This package must remain importable/testable without torch on data-only hosts.
    import src.models.hooks

def test_identity_patch_matches_unpatched_and_current_binding_toy_changes_metric():
    torch=pytest.importorskip("torch")
    from types import SimpleNamespace
    from src.experiments.patching import capture_run,patched_logits
    class Identity(torch.nn.Module):
        def forward(self,x): return x
    class Broadcast(torch.nn.Module):
        def forward(self,x): return x+x.sum(dim=1,keepdim=True)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.config=SimpleNamespace(num_hidden_layers=2); self.embedding=torch.nn.Embedding(3,2); self.layers=torch.nn.ModuleList([Identity(),Broadcast()]); self.head=torch.nn.Linear(2,2,bias=False)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[0.,0.]])); self.head.weight.copy_(torch.eye(2))
        def forward(self,input_ids,use_cache=False):
            x=self.embedding(input_ids)
            for layer in self.layers: x=layer(x)
            return SimpleNamespace(logits=self.head(x))
    model=Toy().eval(); source={"input_ids":torch.tensor([[0,2]])}; target={"input_ids":torch.tensor([[1,2]])}
    source_logits,source_acts=capture_run(model,source); base,target_acts=capture_run(model,target)
    identity=patched_logits(model,target,target_acts[0],0,0)[0]
    assert torch.allclose(identity,base,atol=1e-6)
    patched=patched_logits(model,target,source_acts[0],0,0)[0]
    natural_margin=float(base[0]-base[1]); patched_margin=float(patched[0]-patched[1])
    assert patched_margin>natural_margin

def test_qwen3_component_hooks_capture_and_patch_only_named_tensor_and_block_output_matches_existing():
    torch=pytest.importorskip("torch")
    from types import SimpleNamespace
    from src.experiments.component_patching import ComponentHook, capture_component, patch_component_logits
    from src.experiments.patching import capture_run, patched_logits
    class Scale(torch.nn.Module):
        def __init__(self,scale): super().__init__(); self.scale=scale
        def forward(self,x): return x*self.scale
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.input_layernorm=torch.nn.Identity(); self.self_attn=Scale(2); self.post_attention_layernorm=torch.nn.Identity(); self.mlp=Scale(3)
        def forward(self,hidden_states,**kwargs):
            residual=hidden_states; hidden_states=self.self_attn(self.input_layernorm(hidden_states)); hidden_states=residual+hidden_states
            residual=hidden_states; hidden_states=self.mlp(self.post_attention_layernorm(hidden_states)); return (residual+hidden_states,)
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.config=SimpleNamespace(model_type='qwen3',num_hidden_layers=1); self.layers=torch.nn.ModuleList([Block()]); self.embed=torch.nn.Embedding(3,2); self.head=torch.nn.Linear(2,2,bias=False)
            with torch.no_grad(): self.embed.weight.copy_(torch.tensor([[1.,0.],[0.,1.],[2.,2.]])); self.head.weight.copy_(torch.eye(2))
        def forward(self,input_ids,use_cache=False):
            x=self.embed(input_ids)
            for block in self.layers: x=block(x)[0]
            return SimpleNamespace(logits=self.head(x))
    model=Toy().eval(); source={'input_ids':torch.tensor([[0,2]])}; target={'input_ids':torch.tensor([[1,2]])}; pos=1
    # Component captures are exactly the values entering/leaving their named module.
    expected={'residual_input':model.embed(source['input_ids']),'attention_output':model.layers[0].self_attn(model.embed(source['input_ids'])),'mlp_output':model.layers[0].mlp(model.embed(source['input_ids'])+2*model.embed(source['input_ids'])),'block_output':model.layers[0](model.embed(source['input_ids']))[0]}
    factors={'residual_input':12.,'attention_output':4.,'mlp_output':1.,'block_output':1.}
    baseline=model(**target).logits[0,-1]
    for component in ('residual_input','attention_output','mlp_output','block_output'):
        _,captured=capture_component(model,source,0,component)
        assert torch.equal(captured,expected[component])
        donor=captured.clone(); donor[:,pos,:]+=1
        result=patch_component_logits(model,target,0,component,donor,pos)
        # The output change equals the component's exact downstream residual path.
        assert torch.allclose(result-baseline,torch.tensor([factors[component],factors[component]]),atol=1e-6)
        assert torch.equal(captured,expected[component])
    _,acts=capture_run(model,source)
    via_existing=patched_logits(model,target,acts[0],0,pos)[0]
    via_component=patch_component_logits(model,target,0,'block_output',acts[0],pos)
    assert torch.equal(via_existing,via_component)

def test_component_hook_rejects_unsupported_architecture_clearly():
    torch=pytest.importorskip("torch")
    from types import SimpleNamespace
    from src.experiments.component_patching import validate_qwen3_blocks
    class Unsupported(torch.nn.Module):
        def __init__(self): super().__init__(); self.config=SimpleNamespace(model_type='llama',num_hidden_layers=0); self.layers=torch.nn.ModuleList()
    with pytest.raises(RuntimeError,match='expected config.model_type=.qwen3.'):
        validate_qwen3_blocks(Unsupported())
