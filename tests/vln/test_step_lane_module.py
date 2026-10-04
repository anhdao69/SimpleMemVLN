from dataclasses import replace
import math
import pytest
import torch
from qwen_vl.models.parallel_step_lane import ParallelStepLane
from qwen_vl.research.lane_contract import StepLaneSpec
from step_lane_test_utils import reference_kernel, tiny_spec


def lane(spec=None, dtype=torch.float32, hidden=7):
    return ParallelStepLane(hidden, spec or tiny_spec(), device='cpu', dtype=dtype,
                            chunk_kernel=reference_kernel, recurrent_kernel=reference_kernel,
                            test_only=True, lane_seed=91)


def sample():
    torch.manual_seed(82)
    x = torch.randn(1, 9, 7)
    roles = torch.tensor([[0, 1, 1, 3, 1, 2, 3, 1, 3]], dtype=torch.uint8)
    return x, roles


def test_production_size_and_zero_projection_preserve_rng():
    torch.manual_seed(56)
    before = torch.random.get_rng_state().clone()
    m = lane(StepLaneSpec(), hidden=2560)
    assert torch.equal(before, torch.random.get_rng_state())
    assert sum(p.numel() for p in m.parameters()) == 5263500
    out, state = m(torch.randn(1, 3, 2560), torch.tensor([[0, 1, 3]], dtype=torch.uint8),
                   return_final_state=True)
    assert torch.equal(out, torch.zeros_like(out))
    assert state.dtype == torch.float32
    assert not any('state' in key for key in m.state_dict())


@pytest.mark.parametrize('dtype,tolerance', [(torch.float32, 1e-5), (torch.bfloat16, .035)])
def test_initial_half_lives_and_calibration_after_cast(dtype, tolerance):
    for spec in (tiny_spec(), tiny_spec(clock_mode='token_rate_calibrated', token_rate_reference=320)):
        m = lane(spec, dtype)
        decay = -m.A_log.float().exp() * torch.nn.functional.softplus(m.dt_bias.float())
        actual = math.log(2) / -decay
        target = torch.tensor(spec.half_lives_steps) * (320 if spec.token_rate_reference else 1)
        torch.testing.assert_close(actual, target, rtol=tolerance, atol=0)
        expected_beta = 1 - .9 ** (1/320) if spec.token_rate_reference else .1
        torch.testing.assert_close(m.in_proj_b.bias.float().sigmoid(), torch.full((2,), expected_beta),
                                   rtol=tolerance, atol=0)


def test_first_decision_is_zero_with_open_projection():
    m = lane()
    torch.nn.init.normal_(m.out_proj.weight)
    x, roles = sample()
    out, _ = m(x, roles)
    assert torch.equal(out[:, :3], torch.zeros_like(out[:, :3]))
    assert out[:, 4].abs().sum() > 0


def test_multi_step_opens_output_then_trains_earlier_writers():
    m = lane()
    x, roles = sample()
    y, _ = m(x, roles)
    y[:, 7].sum().backward()
    assert m.out_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert torch.equal(m.in_proj_qkv.weight.grad, torch.zeros_like(m.in_proj_qkv.weight))
    with torch.no_grad():
        m.out_proj.weight.add_(m.out_proj.weight.grad, alpha=-.05)
    m.zero_grad(set_to_none=True)
    x.requires_grad_()
    y, _ = m(x, roles)
    y[:, 7].square().sum().backward()
    assert m.in_proj_qkv.weight.grad.abs().sum() > 0
    assert m.in_proj_b.weight.grad.abs().sum() > 0
    assert m.in_proj_a.weight.grad.abs().sum() > 0
    assert x.grad[:, 3].abs().sum() > 0
    assert torch.equal(x.grad[:, 8], torch.zeros_like(x.grad[:, 8]))


def test_stop_only_keeps_all_parameters_in_graph_with_finite_zero_gradients():
    m = lane()
    y, _ = m(torch.randn(1, 3, 7), torch.tensor([[0, 1, 3]], dtype=torch.uint8))
    y[:, 1].sum().backward()
    for p in m.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert torch.equal(p.grad, torch.zeros_like(p.grad))


@pytest.mark.parametrize('blocks', [[1] * 9, [2, 1, 4, 2]])
def test_functional_blocks_match_whole_sequence_and_gradients_without_detach(blocks):
    m = lane()
    torch.nn.init.normal_(m.out_proj.weight, std=.1)
    x, roles = sample()
    x.requires_grad_()
    y, state = m(x, roles, return_final_state=True)
    parameters = (x, *m.parameters())
    whole_grad = torch.autograd.grad(y.square().sum() + state.square().sum(), parameters)
    cursor, state, pieces = 0, None, []
    for length in blocks:
        part, state = m(x[:, cursor:cursor + length], roles[:, cursor:cursor + length],
                        state, return_final_state=True)
        pieces.append(part)
        cursor += length
    chunked = torch.cat(pieces, 1)
    grad = torch.autograd.grad(chunked.square().sum() + state.square().sum(), parameters)
    torch.testing.assert_close(chunked, y)
    for expected, actual in zip(whole_grad, grad):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)


def test_masks_both_gates_and_protects_kernel_state_aliasing():
    m = lane()
    state = torch.randn(1, 2, 3, 5)
    before = state.clone()
    y, final = m(torch.randn(1, 2, 7), torch.tensor([[1, 2]], dtype=torch.uint8),
                  state, return_final_state=True)
    assert torch.equal(before, state)
    assert torch.equal(final, state)
    assert torch.isfinite(y).all()
    # The adapter clones even if a supplied backend mutates its argument.
    def mutating(**kwargs):
        kwargs['initial_state'].add_(1)
        return reference_kernel(**kwargs)
    m._chunk_kernel = mutating
    m(torch.randn(1, 2, 7), torch.tensor([[1, 2]], dtype=torch.uint8), state, return_final_state=True)
    assert torch.equal(state, before)


@pytest.mark.parametrize('bad', ['input', 'state', 'role', 'dtype'])
def test_invalid_or_nonfinite_input_rejected_before_zero_output(bad):
    m = lane()
    x, roles = sample()
    state = torch.zeros(1, 2, 3, 5)
    if bad == 'input':
        x[0, 0, 0] = float('nan')
    elif bad == 'state':
        state[0, 0, 0, 0] = float('inf')
    elif bad == 'role':
        roles[0, 0] = 4
    else:
        roles = roles.float()
    with pytest.raises((ValueError, FloatingPointError)):
        m(x, roles, state)


def test_production_refuses_test_reference_or_native_torch_fallback():
    with pytest.raises(RuntimeError, match='FLA'):
        ParallelStepLane(7, tiny_spec(), device='cpu', dtype=torch.float32,
                         chunk_kernel=reference_kernel, recurrent_kernel=reference_kernel)


def test_output_normalizes_then_gates_with_shared_norm_and_fp32_statistics():
    m = lane()
    read = torch.tensor([[[[1., 2., 3., 4., 5.], [5., 4., 3., 2., 1.]],
                          [[2., 3., 4., 5., 6.], [6., 5., 4., 3., 2.]]]])
    def fixed_read(**kwargs):
        return read, None
    m._chunk_kernel = fixed_read
    z = torch.tensor([[-2., -1., 0., 1., 2.], [2., 1., 0., -1., -2.]])
    weights = torch.tensor([1., 2., 3., 4., 5.])
    with torch.no_grad():
        m.in_proj_z.weight.zero_()
        m.in_proj_z.weight[:, 0] = z.flatten()
        m.norm.weight.copy_(weights)
        m.out_proj.weight.zero_()
        m.out_proj.weight[:, :7] = torch.eye(7)
    x = torch.zeros(1, 2, 7)
    x[:, :, 0] = 1
    output, _ = m(x, torch.tensor([[1, 1]], dtype=torch.uint8))
    # Independently calculate norm-before-gate. Reversing those operations gives
    # a substantially different value and this fixture catches that mistake.
    expected = read / (read.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    expected = expected * weights * (z / (1 + (-z).exp()))
    torch.testing.assert_close(output, expected.flatten(-2)[..., :7])


@pytest.mark.parametrize('projection', ['in_proj_a', 'in_proj_b'])
def test_nonfinite_pre_gate_projection_cannot_be_hidden_by_activation(projection):
    m = lane()
    with torch.no_grad():
        getattr(m, projection).weight.fill_(float('-inf'))
    with pytest.raises(FloatingPointError):
        m(torch.ones(1, 2, 7), torch.tensor([[0, 1]], dtype=torch.uint8))


def test_cpu_lane_construction_never_discovers_or_seeds_cuda_devices(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU construction must not touch CUDA device RNGs')
    monkeypatch.setattr(torch.cuda, 'device_count', forbidden)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', forbidden)
    m = lane()
    assert m.out_proj.weight.device.type == 'cpu'


def test_output_projection_receives_projection_dtype_under_autocast():
    m = lane()
    seen = []
    handle = m.out_proj.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].dtype))
    x, roles = sample()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        output, _ = m(x, roles)
    handle.remove()
    assert output.dtype == torch.bfloat16
    assert seen == [torch.bfloat16]
