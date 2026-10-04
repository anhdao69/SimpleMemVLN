"""Release gates requiring the installed production CUDA FLA kernels.

No CPU/Torch fallback is accepted. Explicit tolerances measure BF16 production
error independently from the exact FP32 sequential arithmetic tests.
"""
import math
import pytest
import torch
from qwen_vl.models.parallel_step_lane import ParallelStepLane
from qwen_vl.research.lane_contract import StepLaneSpec
from step_lane_test_utils import reference_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Production CUDA/FLA release gate')


def kernels():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    return chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def prepared(length=137):
    generator = torch.Generator(device='cuda').manual_seed(195)
    q = torch.randn(1, length, 4, 128, device='cuda', dtype=torch.bfloat16, generator=generator) * .3
    k = torch.randn(q.shape, device='cuda', dtype=torch.bfloat16, generator=generator) * .3
    v = torch.randn(q.shape, device='cuda', dtype=torch.bfloat16, generator=generator)
    roles = torch.ones((1, length), device='cuda', dtype=torch.uint8)
    roles[:, :70] = 0
    for index in (71, 104, 136):
        if index < length:
            roles[:, index] = 3
    if length == 1:
        roles[:, 0] = 3
    return q, k, v, roles


@pytest.mark.parametrize('kind,length', [('chunk', 137), ('recurrent', 1)])
def test_installed_fla_matches_masked_reference_and_preserves_initial_state(kind, length):
    chunk, recurrent = kernels()
    kernel = chunk if kind == 'chunk' else recurrent
    q, k, v, roles = prepared(length)
    g = torch.where(roles.eq(3)[..., None], torch.full((1, length, 4), -.03, device='cuda'), 0.)
    beta = torch.where(roles.eq(3)[..., None], torch.full((1, length, 4), .1, device='cuda', dtype=torch.bfloat16), 0.)
    state = torch.randn(1, 4, 128, 128, device='cuda') * .02
    before = state.clone()
    kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, scale=1/math.sqrt(128),
                  initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)
    expected, expected_state = reference_kernel(**kwargs)
    actual, actual_state = kernel(**kwargs)
    assert torch.equal(state, before), 'Installed FLA mutates its input; adapter clones must remain enabled'
    assert torch.isfinite(actual).all() and torch.isfinite(actual_state).all()
    print(f'{kind}: max_abs_output={(actual.float()-expected.float()).abs().max().item():.8g}, '
          f'max_abs_state={(actual_state-expected_state).abs().max().item():.8g}')
    torch.testing.assert_close(actual, expected, rtol=.03, atol=.003)
    torch.testing.assert_close(actual_state, expected_state, rtol=.03, atol=.003)
    # All read-only tokens must inherit the initial state, including a full chunk
    # and its boundary. Their query-dependent reads need not be equal.
    if length > 1:
        assert g[:, :70].eq(0).all() and beta[:, :70].eq(0).all()


def test_installed_chunk_gradients_with_exact_zero_mask_crossing_chunk_boundary():
    chunk, _ = kernels()
    q, k, v, roles = prepared()
    q, k, v = [tensor.requires_grad_() for tensor in (q, k, v)]
    raw_g = torch.full((1, 137, 4), -.03, device='cuda', requires_grad=True)
    raw_beta = torch.full((1, 137, 4), .1, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    g = torch.where(roles.eq(3)[..., None], raw_g, 0.)
    beta = torch.where(roles.eq(3)[..., None], raw_beta, 0.)
    inputs = (q, k, v, raw_g, raw_beta)
    kwargs = dict(q=q, k=k, v=v, g=g, beta=beta, scale=1/math.sqrt(128),
                  initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
    expected, _ = reference_kernel(**kwargs)
    actual, _ = chunk(**kwargs)
    probe = torch.randn_like(actual)
    expected_grads = torch.autograd.grad((expected.float() * probe.float()).sum(), inputs, retain_graph=True)
    actual_grads = torch.autograd.grad((actual.float() * probe.float()).sum(), inputs)
    for name, expected_grad, actual_grad in zip(('q', 'k', 'v', 'g', 'beta'), expected_grads, actual_grads):
        assert torch.isfinite(actual_grad).all(), name
        print(f'{name}: max_abs_gradient={(actual_grad.float()-expected_grad.float()).abs().max().item():.8g}')
        torch.testing.assert_close(actual_grad, expected_grad, rtol=.07, atol=.03)
    assert actual_grads[3][:, :70].eq(0).all()
    assert actual_grads[4][:, :70].eq(0).all()


def test_production_lane_bf16_reference_outputs_and_full_bptt():
    chunk, recurrent = kernels()
    spec = StepLaneSpec(layers=(16,))
    m = ParallelStepLane(2560, spec, device='cuda', dtype=torch.bfloat16,
                         chunk_kernel=chunk, recurrent_kernel=recurrent, lane_seed=45)
    with torch.no_grad():
        m.out_proj.weight.normal_(std=.02)
    x = torch.randn(1, 137, 2560, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    roles = prepared()[3]
    output, _ = m(x, roles)
    output[:, 135].float().square().sum().backward()
    assert torch.equal(output[:, :71], torch.zeros_like(output[:, :71]))
    assert x.grad[:, 71].abs().sum() > 0, 'Later loss must train the earlier writer through full BPTT'
    assert x.grad[:, 136].eq(0).all(), 'Post-decision final writer must not affect earlier loss'
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_installed_lane_streaming_irregular_blocks_match_full_sequence():
    chunk, recurrent = kernels()
    before_cpu = torch.random.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone()
    m = ParallelStepLane(2560, StepLaneSpec(layers=(16,)), device='cuda', dtype=torch.bfloat16,
                         chunk_kernel=chunk, recurrent_kernel=recurrent, lane_seed=87).eval()
    assert torch.equal(torch.random.get_rng_state(), before_cpu)
    assert torch.equal(torch.cuda.get_rng_state(), before_cuda)
    x = torch.randn(1, 137, 2560, device='cuda', dtype=torch.bfloat16)
    roles = prepared()[3]
    with torch.no_grad():
        m.out_proj.weight.normal_(std=.02)
        whole, whole_state = m(x, roles, return_final_state=True)
        cursor, state, pieces = 0, None, []
        for length in (70, 1, 1, 32, 1, 31, 1):
            part, state = m(x[:, cursor:cursor+length], roles[:, cursor:cursor+length],
                            state, return_final_state=True)
            pieces.append(part)
            cursor += length
        assert cursor == 137
        streamed = torch.cat(pieces, 1)
    assert torch.isfinite(streamed).all()
    print(f'streamed lane: max_abs_output={(streamed.float()-whole.float()).abs().max().item():.8g}, '
          f'max_abs_state={(state-whole_state).abs().max().item():.8g}')
    torch.testing.assert_close(streamed, whole, rtol=.05, atol=.05)
    torch.testing.assert_close(state, whole_state, rtol=.03, atol=.003)


def test_installed_zero_projection_has_exact_zero_input_and_upstream_gradients():
    chunk, recurrent = kernels()
    m = ParallelStepLane(2560, StepLaneSpec(layers=(16,)), device='cuda', dtype=torch.bfloat16,
                         chunk_kernel=chunk, recurrent_kernel=recurrent, lane_seed=34)
    x = torch.randn(1, 137, 2560, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    roles = prepared()[3]
    output, _ = m(x, roles)
    (output.float() * torch.randn_like(output).float()).sum().backward()
    assert x.grad is not None and x.grad.eq(0).all(), 'Zero projection must transmit exactly zero gradient to native input'
    for name, parameter in m.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        if name != 'out_proj.weight':
            assert parameter.grad.eq(0).all(), name
    assert m.out_proj.weight.grad.abs().sum() > 0


def test_production_single_token_functional_scan_has_backward():
    chunk,recurrent=kernels()
    m=ParallelStepLane(2560,StepLaneSpec(layers=(16,)),device='cuda',dtype=torch.bfloat16,
                       chunk_kernel=chunk,recurrent_kernel=recurrent,lane_seed=73)
    with torch.no_grad():
        m.out_proj.weight.normal_(std=.002)
    x=torch.randn(1,1,2560,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    output,state=m(x,torch.tensor([[3]],device='cuda',dtype=torch.uint8),return_final_state=True)
    (output.float().square().sum()+state.square().sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
