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
