import torch
from torch import nn

from src.models.loader import _materialize_phi4_rope_buffers


def test_phi_rope_meta_buffers_are_rebuilt_on_embedding_device():
    class Phi3RotaryEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('inv_freq', torch.empty(2, device='meta'), persistent=False)
            self.original_inv_freq = self.inv_freq
            self.config = object()
            self.rope_init_fn = lambda config, device: (torch.ones(2, device=device), 0.75)
            self.attention_scaling = 1.0

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(4, 2)
            self.rotary = Phi3RotaryEmbedding()

        def get_input_embeddings(self):
            return self.embedding

    model = FakeModel()
    _materialize_phi4_rope_buffers(model)

    assert not model.rotary.inv_freq.is_meta
    assert not model.rotary.original_inv_freq.is_meta
    assert torch.equal(model.rotary.inv_freq, torch.ones(2))
    assert model.rotary.attention_scaling == 0.75
