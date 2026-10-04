"""Pure parallel associative lane; production execution uses approved FLA kernels.

There is no episode state on this module. The explicit CPU test mode exists solely
for injection of tests' reference kernels, never as a runtime backend fallback.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from qwen_vl.research.lane_contract import StepLaneSpec


def _finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f'Nonfinite step lane {name}')


def _approved(kernel, name):
    return callable(kernel) and getattr(kernel, '__name__', '') == name and getattr(kernel, '__module__', '').startswith('fla.ops.gated_delta_rule.')


class ParallelStepLane(nn.Module):
    def __init__(self, hidden_size: int, spec: StepLaneSpec, *, device, dtype,
                 chunk_kernel=None, recurrent_kernel=None, lane_seed: int = 0,
                 test_only: bool = False):
        super().__init__()
        if (chunk_kernel is None) != (recurrent_kernel is None):
            raise RuntimeError('Both approved FLA kernels must be supplied together')
        if chunk_kernel is None:
            if test_only:
                raise RuntimeError('Test-only CPU mode requires explicit test reference kernels')
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
            except ImportError as exc:
                raise RuntimeError('Step lane requires approved installed FLA kernels') from exc
            chunk_kernel, recurrent_kernel = chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        if not test_only and not (_approved(chunk_kernel, 'chunk_gated_delta_rule') and _approved(recurrent_kernel, 'fused_recurrent_gated_delta_rule')):
            raise RuntimeError('Step lane requires approved FLA kernels; torch fallback is unsupported')
        if test_only and torch.device(device).type != 'cpu':
            raise RuntimeError('Test reference kernels are CPU test-only')
        if type(lane_seed) is not int or lane_seed < 0:
            raise ValueError('lane_seed must be a nonnegative integer')
        if hidden_size <= 0 or spec.num_key_heads <= 0 or spec.num_value_heads % spec.num_key_heads:
            raise ValueError('Invalid grouped query/key head layout')
        if spec.clock_mode not in ('step_end', 'token_rate_calibrated') or spec.prefix_mode != 'read_only_zero':
            raise ValueError('Unsupported clock/prefix rule')
        if len(spec.half_lives_steps) != spec.num_value_heads or any(not math.isfinite(h) or h <= 0 for h in spec.half_lives_steps):
            raise ValueError('Each value head requires a finite positive half-life')
        if not 0 < spec.beta_init < 1:
            raise ValueError('Invalid beta initialization')
        self.spec, self.hidden_size, self.lane_seed = spec, hidden_size, lane_seed
        self._chunk_kernel, self._recurrent_kernel = chunk_kernel, recurrent_kernel
        self._test_only = test_only
        kwargs = dict(device=device, dtype=dtype)
        key_width, value_width = spec.num_key_heads * spec.key_dim, spec.num_value_heads * spec.value_dim
        # Linear constructors also consume RNG. Fork only the target CUDA device:
        # discovering/seeding all GPUs initializes unnecessary contexts per rank.
        target = torch.device(device)
        cuda_devices = []
        if target.type == 'cuda':
            cuda_devices = [torch.cuda.current_device() if target.index is None else target.index]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.random.default_generator.manual_seed(lane_seed)
            for index in cuda_devices:
                torch.cuda.default_generators[index].manual_seed(lane_seed)
            self.in_proj_qkv = nn.Linear(hidden_size, 2 * key_width + value_width, bias=False, **kwargs)
            self.in_proj_z = nn.Linear(hidden_size, value_width, bias=False, **kwargs)
            self.in_proj_a = nn.Linear(hidden_size, spec.num_value_heads, bias=False, **kwargs)
            self.in_proj_b = nn.Linear(hidden_size, spec.num_value_heads, bias=True, **kwargs)
            self.A_log = nn.Parameter(torch.zeros(spec.num_value_heads, **kwargs))
            self.dt_bias = nn.Parameter(torch.empty(spec.num_value_heads, **kwargs))
            self.norm = nn.Module()
            self.norm.weight = nn.Parameter(torch.ones(spec.value_dim, **kwargs))
            self.out_proj = nn.Linear(value_width, hidden_size, bias=False, **kwargs)
            nn.init.normal_(self.in_proj_qkv.weight, std=spec.projection_init_std)
            nn.init.normal_(self.in_proj_z.weight, std=spec.projection_init_std)
            nn.init.zeros_(self.in_proj_a.weight)
            nn.init.zeros_(self.in_proj_b.weight)
            nn.init.zeros_(self.out_proj.weight)
            half_lives = torch.tensor(spec.half_lives_steps, device=device, dtype=torch.float32)
            beta = spec.beta_init
            if spec.clock_mode == 'token_rate_calibrated':
                n_ref = spec.token_rate_reference
                if n_ref is None or not math.isfinite(n_ref) or n_ref <= 0:
                    raise ValueError('Calibrated clock requires positive token_rate_reference')
                half_lives = half_lives * n_ref
                beta = -math.expm1(math.log1p(-beta) / n_ref)
            with torch.no_grad():
                self.dt_bias.copy_(torch.log(torch.expm1(math.log(2) / half_lives)))
                self.in_proj_b.bias.fill_(math.log(beta / (1 - beta)))

    def forward(self, x: torch.Tensor, roles: torch.Tensor, initial_state: torch.Tensor | None = None,
                *, return_final_state: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        spec = self.spec
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 1 or x.shape[-1] != self.hidden_size:
            raise ValueError('Step lane requires nonempty B1 unpadded [1,L,hidden] input')
        if self._test_only and x.device.type != 'cpu':
            raise RuntimeError('Test reference kernels are CPU test-only')
        if roles.shape != x.shape[:2] or roles.dtype != torch.uint8 or roles.device != x.device:
            raise ValueError('Step lane roles must be uint8 [1,L] on input device')
        if roles.gt(3).any():
            raise ValueError('Invalid step lane role')
        _finite(x, 'input')
        state_shape = (x.shape[0], spec.num_value_heads, spec.key_dim, spec.value_dim)
        if initial_state is not None:
            if initial_state.shape != state_shape or initial_state.dtype != torch.float32 or initial_state.device != x.device:
                raise ValueError('Initial lane state must have FP32 key-by-value shape on input device')
            _finite(initial_state, 'initial state')
        batch, length, _ = x.shape
        key_width = spec.num_key_heads * spec.key_dim
        value_width = spec.num_value_heads * spec.value_dim
        qkv = F.silu(self.in_proj_qkv(x))
        _finite(qkv, 'qkv')
        q, k, v = qkv.split((key_width, key_width, value_width), dim=-1)
        repeats = spec.num_value_heads // spec.num_key_heads
        q = q.reshape(batch, length, spec.num_key_heads, spec.key_dim).repeat_interleave(repeats, dim=2)
        k = k.reshape(batch, length, spec.num_key_heads, spec.key_dim).repeat_interleave(repeats, dim=2)
        v = v.reshape(batch, length, spec.num_value_heads, spec.value_dim)
        z = self.in_proj_z(x).reshape(batch, length, spec.num_value_heads, spec.value_dim)
        a, b = self.in_proj_a(x), self.in_proj_b(x)
        _finite(a, 'a projection')
        _finite(b, 'b projection')
        g_raw = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        beta_raw = b.sigmoid()
        for tensor, name in ((z, 'z'), (g_raw, 'decay gate'), (beta_raw, 'write gate')):
            _finite(tensor, name)
        write = roles.eq(3) if spec.clock_mode == 'step_end' else roles.ne(0)
        # No in-place mutations of intermediates saved for full-episode backward.
        g = torch.where(write[..., None], g_raw, 0.)
        beta = torch.where(write[..., None], beta_raw, 0.)
        # The fused recurrent backend is a serving kernel; functional/autograd
        # calls always use the differentiable chunk path, even at length one.
        serving_token = length == 1 and return_final_state and not torch.is_grad_enabled()
        kernel = self._recurrent_kernel if serving_token else self._chunk_kernel
        # Protect the caller's functional state even against kernels that alias or
        # mutate initial_state. clone preserves the full BPTT graph.
        read, final_state = kernel(q=q, k=k, v=v, g=g, beta=beta,
                                   scale=spec.key_dim ** -0.5,
                                   initial_state=None if initial_state is None else initial_state.clone(),
                                   output_final_state=return_final_state,
                                   use_qk_l2norm_in_kernel=True)
        if read.shape != v.shape:
            raise ValueError('FLA lane output has incorrect head layout')
        _finite(read, 'kernel output')
        if return_final_state:
            if final_state is None or final_state.shape != state_shape or final_state.device != x.device or final_state.dtype != torch.float32:
                raise ValueError('FLA lane final state must have FP32 key-by-value shape')
            _finite(final_state, 'final state')
        elif final_state is not None:
            raise ValueError('FLA lane unexpectedly returned state')
        read_fp32 = read.float()
        normalized = read_fp32 * torch.rsqrt(read_fp32.square().mean(-1, keepdim=True) + spec.eps)
        gated = normalized * self.norm.weight.float() * F.silu(z.float())
        _finite(gated, 'normalized gated output')
        output = self.out_proj(gated.to(v.dtype).reshape(batch, length, value_width))
        _finite(output, 'projected output')
        return output, final_state
