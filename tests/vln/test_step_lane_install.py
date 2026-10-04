from types import SimpleNamespace
from dataclasses import replace
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class Native(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.chunk_gated_delta_rule = lambda *a, **k: None
        self.recurrent_gated_delta_rule = self.chunk_gated_delta_rule
    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        assert not any(k.startswith('step_lane_') for k in kwargs)
        return self.proj(hidden_states)


class Attention(nn.Module):
    def forward(self, x, **kwargs):
        assert not any(k.startswith('step_lane_') for k in kwargs)
        return x


class FakeLane(nn.Module):
    def __init__(self, hidden_size, spec, **kwargs):
        super().__init__()
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.out_proj.weight)
    def forward(self, x, roles, initial_state=None, *, return_final_state=False):
        if roles is None:
            raise ValueError('roles required')
        return self.out_proj(x * roles.unsqueeze(-1)), initial_state


def fixture_model(monkeypatch):
    import qwen_vl.models.install_step_lane as install
    from qwen_vl.research.lane_contract import StepLaneSpec
    monkeypatch.setattr(install, 'ParallelStepLane', FakeLane)
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.model = nn.Module()
    model.backbone.model.language_model = nn.Module()
    layers = nn.ModuleList()
    for i in range(32):
        layer = nn.Module()
        if i % 4 == 3:
            layer.self_attn = Attention()
        else:
            layer.linear_attn = Native()
        layers.append(layer)
    model.backbone.model.language_model.layers = layers
    model.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4, layer_types=['full_attention' if i%4==3 else 'linear_attention' for i in range(32)]))
    return model, StepLaneSpec(), install


def test_install_keeps_parameters_rng_and_native_off_callable(monkeypatch):
    model, spec, install = fixture_model(monkeypatch)
    original = dict(model.named_parameters())
    torch.manual_seed(194)
    before = torch.get_rng_state().clone()
    install.install_step_lanes(model, spec, init_seed=72)
    assert torch.equal(before, torch.get_rng_state())
    after = dict(model.named_parameters())
    assert all(after[n] is p for n, p in original.items())
    assert all('.step_lane.' in n for n in after.keys() - original.keys())
    install.install_step_lanes(model, spec, init_seed=72)
    assert {n: id(p) for n,p in after.items()} == {n: id(p) for n,p in model.named_parameters()}
    with pytest.raises(ValueError):
        install.install_step_lanes(model, replace(spec, beta_init=0.2), init_seed=72)
    m = model.backbone.model.language_model.layers[16].linear_attn
    x = torch.randn(1, 8, 4)
    roles = torch.tensor([[0,1,1,3,1,1,2,3]], dtype=torch.uint8)
    expected = m._step_lane_native_forward(m, x)
    torch.testing.assert_close(m(x, step_lane_roles=roles), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match='roles'):
        m(x)
    model.backbone.model.language_model.layers[3].self_attn(x, step_lane_roles=roles)


def test_checkpoint_captures_explicit_roles_and_preserves_gradients(monkeypatch):
    model, spec, install = fixture_model(monkeypatch)
    install.install_step_lanes(model, spec, init_seed=7)
    m = model.backbone.model.language_model.layers[16].linear_attn
    nn.init.normal_(m.step_lane.out_proj.weight)
    roles = torch.tensor([[0,1,1,3,1,1,2,3]], dtype=torch.uint8)
    x = torch.randn(1,8,4, requires_grad=True)
    direct = m(x, step_lane_roles=roles)
    reference = torch.autograd.grad(direct.square().sum(), (x,*m.parameters()))
    result = checkpoint(m, x, step_lane_roles=roles, use_reentrant=False)
    # An unrelated outer name can be reassigned, but recomputation consumes the passed value.
    roles = torch.zeros_like(roles)
    actual = torch.autograd.grad(result.square().sum(), (x,*m.parameters()))
    for a,b in zip(actual, reference):
        torch.testing.assert_close(a,b, rtol=0, atol=0)
