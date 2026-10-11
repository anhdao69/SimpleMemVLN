"""Committed-write magnitude of the native gated delta rule (score-only side pass).

For value head h and token i, with key-by-value state S (FP32):

    S_decay  = exp(g_i) * S_prev
    delta_i  = beta_i * (v_i - S_decay^T k_i)          # committed delta
    S_next   = S_decay + k_i delta_i^T
    score_i  = mean_h ||delta_i,h||_2                  # FP32

``prepare_gdn_inputs`` reproduces the native Qwen3.5 GDN preparation from the
decoder layer's *input* hidden state using the native modules (input RMSNorm,
in_proj_qkv/b/a, depthwise causal conv + SiLU, sigmoid beta, softplus gate,
FLA-style L2 key normalization).  It never touches the differentiable native
mixer call: the decoder layer still runs unchanged (and checkpointed); this
pass runs under ``torch.no_grad`` outside the checkpoint, so recomputation
can never replace the scores that built a plan.

``_scan_blocks`` evaluates the exact block (WY) form per internal block of at
most 64 tokens with carried state S0:

    L_ij  = beta_i exp(c_i - c_j) k_i.k_j   (j < i; exponent masked before exp)
    Delta = solve_triangular(I + L, beta*v - beta*exp(c)*K S0)
    S_end = exp(c_last) S0 + sum_i exp(c_last - c_i) k_i Delta_i^T

``Delta`` already contains beta; it is never multiplied by beta again.  The
helper state is not the policy state and is discarded after scoring (training)
or kept only inside the bounded session metadata (serving).  Dense per-token
matrices are never materialized; transient buffers are bounded by
``max_super_block_tokens``.
"""
import torch
import torch.nn.functional as F
from qwen_vl.research.episodic_contract import SCORE_BLOCK_TOKENS

L2_EPS = 1e-6


def _gdn(decoder_layer):
    mixer = getattr(decoder_layer, 'linear_attn', None)
    if mixer is None or decoder_layer.layer_type != 'linear_attention':
        raise ValueError('Committed scores come only from native GDN layers')
    return mixer


def empty_conv_tail(decoder_layer, device, dtype=torch.bfloat16):
    mixer = _gdn(decoder_layer)
    return torch.zeros(mixer.conv_kernel_size - 1, mixer.conv_dim, device=device, dtype=dtype)


@torch.no_grad()
def prepare_gdn_inputs(decoder_layer, hidden, conv_tail):
    """Native GDN k/v/g/beta for ``hidden`` [T, width] given the previous conv inputs.

    Returns FP32 L2-normalized keys [T, Hk, Dk], FP32 values [T, Hv, Dv],
    FP32 log decay g [T, Hv], FP32 beta [T, Hv] and the new conv tail.
    """
    mixer = _gdn(decoder_layer)
    if hidden.ndim != 2:
        raise ValueError('Expected [tokens, width] hidden state')
    normed = decoder_layer.input_layernorm(hidden[None])[0]
    mixed = mixer.in_proj_qkv(normed)                              # [T, conv_dim]
    b = mixer.in_proj_b(normed)
    a = mixer.in_proj_a(normed)
    window = torch.cat((conv_tail.to(mixed.dtype), mixed), 0)        # causal left context
    conv = F.conv1d(window.t()[None], mixer.conv1d.weight, mixer.conv1d.bias,
                    groups=mixer.conv_dim)[0].t()                    # [T, conv_dim]
    conv = F.silu(conv)
    new_tail = window[-(mixer.conv_kernel_size - 1):].clone()
    _, key, value = torch.split(conv, [mixer.key_dim, mixer.key_dim, mixer.value_dim], dim=-1)
    key = key.reshape(-1, mixer.num_k_heads, mixer.head_k_dim).float()
    key = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + L2_EPS)
    value = value.reshape(-1, mixer.num_v_heads, mixer.head_v_dim).float()
    beta = b.sigmoid().float()
    g = -mixer.A_log.float().exp() * F.softplus(a.float() + mixer.dt_bias)
    return key, value, g.float(), beta, new_tail


def internal_blocks(segments, block=SCORE_BLOCK_TOKENS):
    """Split each (start, end) segment into blocks of <= ``block`` from its own start."""
    result = []
    for start, end in segments:
        if end <= start:
            raise ValueError('Empty score segment')
        result.extend((s, min(s + block, end)) for s in range(start, end, block))
    return result


@torch.no_grad()
def _scan_blocks(key, value, g, beta, state, blocks, value_per_key):
    """Exact FP32 committed deltas for consecutive internal blocks.

    ``blocks`` are (start, end) relative to the given tensors and must tile them
    contiguously.  Returns (per-token scores [N], final state [Hv, Dk, Dv]).
    """
    device, block_len = key.device, SCORE_BLOCK_TOKENS
    count = len(blocks)
    index = torch.full((count, block_len), -1, dtype=torch.long)
    cursor = 0
    for row, (start, end) in enumerate(blocks):
        if start != cursor or not 0 < end - start <= block_len:
            raise ValueError('Internal blocks must tile the super-block')
        index[row, :end - start] = torch.arange(start, end)
        cursor = end
    if cursor != key.shape[0]:
        raise ValueError('Internal blocks must cover every token')
    index = index.to(device, non_blocking=True)
    valid = index.ge(0)
    flat = index.clamp(min=0)
    mask = valid[..., None].to(key.dtype)
    kb = (key[flat].repeat_interleave(value_per_key, dim=2) * mask[..., None])   # [n, L, Hv, Dk]
    vb = value[flat] * mask[..., None]                                          # [n, L, Hv, Dv]
    gb = g[flat] * mask                                                          # [n, L, Hv]
    bb = beta[flat] * mask
    kb, vb = kb.transpose(1, 2), vb.transpose(1, 2)                              # [n, Hv, L, D]
    c = gb.cumsum(1).transpose(1, 2)                                             # [n, Hv, L]
    bb = bb.transpose(1, 2)
    lower = torch.ones(block_len, block_len, dtype=torch.bool, device=device).tril(-1)
    exponent = torch.where(lower, c[..., :, None] - c[..., None, :], torch.zeros((), device=device))
    decay_pairs = torch.where(lower, exponent.exp(), torch.zeros((), device=device))
    system = bb[..., :, None] * decay_pairs * (kb @ kb.transpose(-1, -2))
    system = system + torch.eye(block_len, device=device, dtype=key.dtype)
    rhs = torch.cat((bb[..., None] * vb, (bb * c.exp())[..., None] * kb), -1)
    solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    width = vb.shape[-1]
    u, w = solved[..., :width], solved[..., width:]
    last = c[..., -1]                                                            # pads carry g=0
    decay = last.exp()                                                           # [n, Hv]
    keys_to_end = (last[..., None] - c).exp()[..., None] * kb                    # [n, Hv, L, Dk]
    norms = torch.empty(count, kb.shape[1], block_len, device=device, dtype=key.dtype)
    for row in range(count):
        delta = torch.baddbmm(u[row], w[row], state, alpha=-1.0)                 # [Hv, L, Dv]
        norms[row] = delta.norm(dim=-1)
        state = torch.baddbmm(decay[row][:, None, None] * state,
                              keys_to_end[row].transpose(-1, -2), delta)
    scores = norms.mean(1)[valid]
    return scores, state


class CommittedWriteScorer:
    """Carried score-helper state for one scoring GDN layer of one episode."""

    def __init__(self, decoder_layer, device, max_super_block_tokens=4096):
        mixer = _gdn(decoder_layer)
        self.layer = decoder_layer
        self.value_per_key = mixer.num_v_heads // mixer.num_k_heads
        self.conv_tail = empty_conv_tail(decoder_layer, device)
        self.state = torch.zeros(mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim,
                                 device=device, dtype=torch.float32)
        self.max_super_block_tokens = max_super_block_tokens
        self.pending = []   # prepared response tokens awaiting their closing segment

    def _prepare(self, hidden):
        key, value, g, beta, self.conv_tail = prepare_gdn_inputs(self.layer, hidden, self.conv_tail)
        return key, value, g, beta

    def _scan(self, prepared, segments):
        key, value, g, beta = prepared
        scores, self.state = _scan_blocks(key, value, g, beta, self.state,
                                          internal_blocks(segments), self.value_per_key)
        return scores

    @torch.no_grad()
    def score_sequence(self, hidden, segments):
        """Offline: scores [T] for a complete token stream tiled by ``segments``."""
        hidden = hidden.reshape(-1, hidden.shape[-1])
        if segments[0][0] != 0 or segments[-1][1] != hidden.shape[0]:
            raise ValueError('Score segments must cover the whole sequence')
        out = torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32)
        batch, start = [], segments[0][0]
        for segment in segments:
            if batch and segment[1] - start > self.max_super_block_tokens:
                self._score_range(hidden, start, batch, out)
                batch, start = [], segment[0]
            batch.append(segment)
        self._score_range(hidden, start, batch, out)
        return out

    def _score_range(self, hidden, start, segments, out):
        end = segments[-1][1]
        prepared = self._prepare(hidden[start:end])
        relative = [(a - start, b - start) for a, b in segments]
        out[start:end] = self._scan(prepared, relative)

    # ---- serving: identical segmentation, one append at a time ----------
    @torch.no_grad()
    def append(self, hidden, kind):
        """``kind``: prefix | observation | response_token | response_close."""
        hidden = hidden.reshape(-1, hidden.shape[-1])
        prepared = self._prepare(hidden)
        if kind in ('prefix', 'observation'):
            if self.pending:
                raise RuntimeError('Unclosed response before a new segment')
            return self._scan(prepared, [(0, hidden.shape[0])])
        if kind not in ('response_token', 'response_close'):
            raise ValueError(f'Unknown score segment kind {kind!r}')
        self.pending.append(prepared)
        if kind == 'response_token':
            return None
        joined = [torch.cat(parts, 0) for parts in zip(*self.pending)]
        self.pending = []
        return self._scan(joined, [(0, joined[0].shape[0])])
