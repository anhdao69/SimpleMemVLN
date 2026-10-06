"""Real cache/position and attention tests; GPU tests require the H100 stack."""
import torch
import pytest
from transformers import Qwen3_5TextConfig


def test_cache_preserves_fp32_after_bf16_conv():
    from qwen_vl.stream.cache import make_stream_cache_fp32, evict_kv

    cfg = Qwen3_5TextConfig(
        num_hidden_layers=2, layer_types=["linear_attention", "full_attention"]
    )
    cache = make_stream_cache_fp32(cfg)
    cache.update_conv_state(torch.ones(1, 8, 4, dtype=torch.bfloat16), 0)
    state = torch.full((1, 2, 4, 4), 1.0001)
    cache.update_recurrent_state(state, 0)
    assert cache.layers[0].recurrent_states.dtype == torch.float32
    torch.testing.assert_close(cache.layers[0].recurrent_states, state, atol=0, rtol=0)
    kv = torch.arange(10.0).reshape(1, 1, 10, 1)
    cache.update(kv, kv, 1)
    evict_kv(cache, prefix_length=2, remove_tokens=3)
    assert cache.layers[1].keys.flatten().tolist() == [0, 1, 5, 6, 7, 8, 9]
    torch.testing.assert_close(cache.layers[0].recurrent_states, state, atol=0, rtol=0)


def test_window_flash_matches_dense_values_and_gradients():
    if not torch.cuda.is_available():
        pytest.skip("H100 FlashAttention test")
    from qwen_vl.stream.window_attention import step_flash_attention

    torch.manual_seed(5)
    # Variable step sizes; step 8 must lose step 0, but keep prefix.
    spans = (
        (2, 5),
        (5, 6),
        (6, 10),
        (10, 12),
        (12, 15),
        (15, 17),
        (17, 18),
        (18, 20),
        (20, 23),
        (23, 25),
    )
    q = torch.randn(
        1, 4, 25, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        1, 2, 25, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn_like(k, requires_grad=True)
    actual, _ = step_flash_attention(
        None, q, k, v, None, scaling=32**-0.5, step_plan=(2, spans)
    )
    group = torch.tensor(
        [-1] * 2 + [t for t, (s, e) in enumerate(spans) for _ in range(e - s)],
        device="cuda",
    )
    pos = torch.arange(25, device="cuda")
    mask = (pos[None, :] <= pos[:, None]) & (
        (group[None, :] == -1) | (group[None, :] >= group[:, None] - 7)
    )
    logits = (
        q.float() @ k.float().repeat_interleave(2, 1).transpose(-1, -2)
    ) * 32**-0.5
    expected = (
        logits.masked_fill(~mask, -torch.inf).softmax(-1)
        @ v.float().repeat_interleave(2, 1)
    ).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, atol=0.025, rtol=0.025)
    grad = torch.randn_like(expected)
    ga = torch.autograd.grad(
        (actual.float() * grad).sum(), (q, k, v), retain_graph=True
    )
    ge = torch.autograd.grad((expected * grad).sum(), (q, k, v))
    for a, e in zip(ga, ge):
        torch.testing.assert_close(a.float(), e.float(), atol=0.04, rtol=0.04)


def test_action_token_loss_is_action_mean_not_token_mean():
    from qwen_vl.models.nav_model import per_action_token_loss

    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]], requires_grad=True)
    targets = torch.tensor([0, 0, 0])
    result = per_action_token_loss(logits, targets, torch.tensor([0, 1, 1]), 2)
    expected = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    torch.testing.assert_close(result, torch.stack([expected[0], expected[1:].mean()]))
    result.sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_fla_hopper_backward_matches_fp32_recurrent_oracle():
    if not torch.cuda.is_available():
        pytest.skip("H100 FLA backward gate")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        torch_recurrent_gated_delta_rule,
    )

    torch.manual_seed(19)
    q = torch.randn(
        1, 137, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    g = (-torch.rand(1, 137, 4, device="cuda") * 0.1).requires_grad_()
    beta = torch.rand(
        1, 137, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    actual, _ = chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True
    )
    refs = [x.detach().float().requires_grad_() for x in (q, k, v, g, beta)]
    expected, _ = torch_recurrent_gated_delta_rule(
        *refs[:3],
        g=refs[3],
        beta=refs[4],
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True
    )
    torch.testing.assert_close(actual.float(), expected, atol=0.012, rtol=0.04)
    grad = torch.randn_like(expected) * 0.01
    ga = torch.autograd.grad((actual.float() * grad).sum(), (q, k, v, g, beta))
    ge = torch.autograd.grad((expected * grad).sum(), refs)
    for a, e in zip(ga, ge):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.float(), e, atol=0.004, rtol=0.07)
    from fla.ops.backends import BackendRegistry

    registry = BackendRegistry._registries["common"]
    if torch.cuda.get_device_capability()[0] == 9:
        assert "common:chunk_bwd_dqkwg:tilelang" in registry._logged
