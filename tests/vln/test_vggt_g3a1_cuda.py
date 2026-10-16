"""Pinned H100 FLA qualification of the fused G3+A1 decoder24 against both single arms."""
import copy
import functools
from types import SimpleNamespace
import pytest
import torch
from torch.utils.checkpoint import checkpoint
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='H100 FLA qualification')


def relative(actual, expected):
    return float((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def build(active):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from qwen_vl.models.geoaddress_runtime import install_geoaddress
    from qwen_vl.models.vggt_g3_heads import NativeStateHead
    cfg = Qwen3_5TextConfig(hidden_size=2560, linear_num_value_heads=32, linear_num_key_heads=16,
                            linear_key_head_dim=128, linear_value_head_dim=128, dtype=torch.bfloat16)
    torch.manual_seed(11)
    gdn = Qwen3_5GatedDeltaNet(cfg, 24).cuda().bfloat16()
    with torch.no_grad():
        gdn.A_log.zero_(); gdn.dt_bias.zero_()
    plain = copy.deepcopy(gdn)  # same native weights, no adapter: the G3-only layer
    install_geoaddress(SimpleNamespace(layers=[SimpleNamespace(linear_attn=gdn)]), layer_index=0)
    if active:
        with torch.no_grad():
            gdn.geoaddress.q_up.weight.normal_(0, .05); gdn.geoaddress.k_up.weight.normal_(0, .05)
    head = NativeStateHead().cuda().bfloat16()
    # Episode layout: source frame image rows 0..299, text 300..319, read frame 320..619, text 620..699.
    x = torch.randn(1, 700, 2560, device='cuda', dtype=torch.bfloat16)
    mask = torch.zeros(1, 700, dtype=torch.bool, device='cuda'); mask[:, :300] = True; mask[:, 320:620] = True
    anchors = torch.tensor([650, 690], device='cuda'); lags = torch.tensor([8, 20], device='cuda')
    targets = torch.randn(2, 5, 2048, device='cuda')
    from qwen_vl.models.geoaddress_loss import teacher_relations, address_relation_loss
    qpos = (320 + torch.randperm(300, device='cuda')[:32]).repeat(2, 1); kpos = torch.arange(300, device='cuda').repeat(2, 1)
    probs = teacher_relations(torch.randn(2, 32, 2048, device='cuda'), torch.randn(2, 300, 2048, device='cuda'))
    relation = lambda q, k: address_relation_loss(q[0, qpos, :2], k[0, kpos, :2], probs)
    return SimpleNamespace(gdn=gdn, plain=plain, head=head, x=x, mask=mask, anchors=anchors, lags=lags,
                           targets=targets, relation=relation)


def test_zero_adapter_fused_equals_g3_and_a1_exactly():
    from qwen_vl.models.vggt_g3a1_forward import address_state_gdn
    from qwen_vl.models.vggt_g3_forward import native_state_gdn
    from qwen_vl.models.geoaddress_runtime import geoaddress_gdn
    s = build(active=False)
    with torch.no_grad():
        out, losses = address_state_gdn(s.gdn, s.x, s.head, s.anchors, s.lags, s.targets, s.mask, s.relation)
        g3_out, g3_loss = native_state_gdn(s.plain, s.x, s.head, s.anchors, s.lags, s.targets)
        _, a1_loss = geoaddress_gdn(s.gdn, s.x, s.mask, relation=s.relation)
        # No pairs: fused real output is exactly A1's layer output (both without probe rows).
        empty_out, empty = address_state_gdn(s.gdn, s.x, s.head, s.anchors[:0], s.lags[:0], s.targets[:0], s.mask, s.relation)
        a1_out, _ = geoaddress_gdn(s.gdn, s.x, s.mask, relation=s.relation)
    assert torch.equal(out, g3_out) and torch.equal(losses[0], g3_loss.float())
    assert torch.equal(losses[1], a1_loss.float()) and float(losses[1]) > 0
    assert torch.equal(empty_out, a1_out) and float(empty[0]) == 0
    print('G3A1_ZERO_ADAPTER_EXACT', {'g3': float(losses[0]), 'a1': float(losses[1])})


def test_active_adapter_relation_exact_reads_reference_and_gradients():
    import fla.ops.gated_delta_rule as fla_gdn
    from qwen_vl.models.vggt_g3a1_forward import address_state_gdn
    from qwen_vl.models.geoaddress_runtime import geoaddress_gdn
    from test_vggt_g3 import recurrence
    s = build(active=True)
    real_kernel = s.gdn.chunk_gated_delta_rule
    captured = {}

    def capturing(q, k, v, **kwargs):  # identical kernel; records the probe-expanded inputs
        captured['inputs'] = (q, k, v, kwargs['g'], kwargs['beta'])
        out = real_kernel(q, k, v, **kwargs); captured['out'] = out[0]; return out
    capturing.__module__, capturing.__name__ = real_kernel.__module__, real_kernel.__name__
    with torch.no_grad():
        a1_out, a1_loss = geoaddress_gdn(s.gdn, s.x, s.mask, relation=s.relation)
        s.gdn.chunk_gated_delta_rule = capturing
        out, losses = address_state_gdn(s.gdn, s.x, s.head, s.anchors, s.lags, s.targets, s.mask, s.relation)
        s.gdn.chunk_gated_delta_rule = real_kernel
    # A1 objective is computed before the kernel on identical adapted raw heads: exact.
    assert torch.equal(losses[1], a1_loss.float())
    # Probe insertion only changes FLA BF16 chunk boundaries for real rows (<2%, as G3 gate).
    assert relative(out, a1_out) < .02, relative(out, a1_out)
    # Probe reads equal an independent FP64 recurrence over the adapted expanded inputs.
    reference, _ = recurrence(*[t.detach().double() for t in captured['inputs']])
    probe_rows = torch.cat([a + 1 + torch.arange(5, device='cuda') + 5 * i for i, a in enumerate(s.anchors.tolist())])
    assert relative(captured['out'][:, probe_rows], reference[:, probe_rows]) < .02
    # Adapted K enters the probed state: probe keys are unit vectors, real image keys are adapted.
    q, k, v, g, beta = captured['inputs']
    assert torch.count_nonzero(g[:, probe_rows]) == 0 and torch.count_nonzero(beta[:, probe_rows]) == 0
    results = []
    for enabled in (False, True):
        leaf = s.x.clone().requires_grad_()
        s.gdn.zero_grad(set_to_none=True); s.head.zero_grad(set_to_none=True)
        fn = lambda h: address_state_gdn(s.gdn, h, s.head, s.anchors, s.lags, s.targets, s.mask, s.relation)
        o, l = checkpoint(fn, leaf, use_reentrant=False) if enabled else fn(leaf)
        l.sum().backward()
        # Auxiliary losses never depend on rows after the last anchor.
        assert leaf.grad[:, :691].abs().sum() > 0 and leaf.grad[:, 691:].abs().max() < 1e-7
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in s.gdn.geoaddress.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in s.head.parameters())
        results.append((o.detach(), l.detach(), leaf.grad.clone(), s.gdn.geoaddress.k_up.weight.grad.clone(),
                        s.head.region_embedding.grad.clone()))
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    # The G3 state objective alone trains the adapters through adapted image keys in the state.
    s.gdn.zero_grad(set_to_none=True); s.head.zero_grad(set_to_none=True)
    _, l = address_state_gdn(s.gdn, s.x, s.head, s.anchors, s.lags, s.targets, s.mask, s.relation)
    l[0].backward()
    assert s.gdn.geoaddress.k_up.weight.grad.abs().sum() > 0
    print('G3A1_ACTIVE_PASS', {'a1': float(losses[1]), 'g3': float(losses[0]), 'real_vs_a1': relative(out, a1_out),
                               'probe_vs_fp64': relative(captured['out'][:, probe_rows], reference[:, probe_rows])})
