"""Native-GDN scoring features for the M5 GDN-only S sweep (score-only side passes).

Every function runs under ``torch.no_grad`` on FP32 tensors produced by the
unchanged ``prepare_gdn_inputs`` (native input RMSNorm, projections, causal
conv + SiLU, sigmoid beta, softplus decay, additive-eps L2 key normalization)
and uses the legacy segmentation and block tiling (``internal_blocks``: blocks
of at most 64 tokens from each segment's own start).  Key-by-value state per
value head; keys reach value heads through the native contiguous repeat:

    Sbar_i = exp(g_i) S_{i-1};   vhat_i = Sbar_i^T k_i;   e_i = v_i - vhat_i
    delta_i = beta_i e_i;        S_i = Sbar_i + k_i delta_i^T

``residual_block_scan`` solves the *unscaled* block system (beta belongs to
column j and is never divided out, so beta = 0 is exact):

    E_t + sum_{j<t} exp(c_t - c_j) (k_t.k_j) beta_j E_j = v_t - exp(c_t) S0^T k_t

and evaluates the decayed-prior prediction directly,
``vhat_t = exp(c_t) S0^T k_t + sum_{j<t} exp(c_t - c_j)(k_t.k_j) beta_j E_j``.
No query scale enters ``e``.

``headwise_committed_scan`` repeats the qualified WY recurrence of
``committed_write._scan_blocks`` expression for expression, keeping value-head
norms (S6) and, for tracked observation segments, the exact
decay-transported write ``W = sum_i (prod_{j>i} alpha_j) k_i delta_i^T``
carried as its own accumulator (S8).  The legacy S0 scorer itself is not
modified and remains the only S0/S7 path.
"""
import math
import numpy as np
import torch
from qwen_vl.models.committed_write import (
    CommittedWriteScorer, _gdn, empty_conv_tail, internal_blocks, prepare_gdn_inputs)
from qwen_vl.research.episodic_contract import SCORE_BLOCK_TOKENS, VISUAL_TOKENS_PER_GROUP
from qwen_vl.research.gdn_sweep_contract import (
    BOUNDARY_BACKEND, HEADWISE_BACKEND, LEGACY_BACKEND, PREPARE_BACKEND, RESIDUAL_BACKEND)

# Boundary statistics per tracked observation segment (full-layer Frobenius).
BOUNDARY_FIELDS = ('net_drift', 'transported_write', 'before_norm', 'decay_part_norm',
                   'decay_write_cross', 'after_norm')
RAW_FEATURE_SCHEMA = 'gdn_raw_v2'
RAW_FIELDS = ('beta', 'log_decay_g', 'value_norm', 'error_norm', 'committed_norm',
              'prediction_norm', 'prediction_dot_value', 'cosine_error', 'key_norm_sq',
              'prediction_near_zero', 'value_near_zero')
F0_SCHEMA, F1_SCHEMA, F1_LEGACY_SCHEMA = 'F0_v1', 'F1_v2', 'F1_v1_legacy'
F0_CHANNELS = ('log_beta', 'log_neg_g', 'log_value_norm')
F1_CHANNELS = F0_CHANNELS + ('log_error_norm', 'log_prediction_norm', 'cosine_error')
F1_LEGACY_CHANNELS = F0_CHANNELS + ('log_error_norm', 'log_committed_norm', 'cosine_error')


def _tile(key, value, g, beta, blocks, value_per_key):
    """Block tensors, built with exactly the expressions of the legacy scan."""
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
    return count, valid, kb, vb, bb, c, decay_pairs


@torch.no_grad()
def headwise_committed_scan(key, value, g, beta, state, blocks, value_per_key, heads=None,
                            tracked=()):
    """Legacy WY committed scan with per-head control and boundary statistics.

    ``heads`` (LongTensor) selects the value heads averaged into each token's
    score; ``None`` averages all heads exactly like the legacy scorer.
    ``tracked``: inclusive (first_row, last_row) block-row ranges, one per
    observation segment.  Returns (scores [N], final state, stats [len, 6]).
    """
    count, valid, kb, vb, bb, c, decay_pairs = _tile(key, value, g, beta, blocks, value_per_key)
    device, block_len = key.device, SCORE_BLOCK_TOKENS
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
    starts = {first: last_row for first, last_row in tracked}
    if len(starts) != len(tracked) or any(not 0 <= a <= b < count for a, b in tracked):
        raise ValueError('Tracked segments must be disjoint block-row ranges')
    stats, open_until, before, write, log_decay = [], None, None, None, None
    for row in range(count):
        if row in starts:
            if open_until is not None:
                raise ValueError('Tracked segments overlap')
            open_until, before = starts[row], state.clone()
            write, log_decay = torch.zeros_like(state), torch.zeros_like(last[row])
        delta = torch.baddbmm(u[row], w[row], state, alpha=-1.0)                 # [Hv, L, Dv]
        norms[row] = delta.norm(dim=-1)
        if open_until is not None:
            write = torch.baddbmm(decay[row][:, None, None] * write,
                                  keys_to_end[row].transpose(-1, -2), delta)
            log_decay = log_decay + last[row]
        state = torch.baddbmm(decay[row][:, None, None] * state,
                              keys_to_end[row].transpose(-1, -2), delta)
        if open_until == row:
            stats.append(boundary_statistics(before, state, write, log_decay))
            open_until = None
    if open_until is not None:
        raise ValueError('Tracked segment extends past the scanned blocks')
    if heads is None:
        scores = norms.mean(1)[valid]
    else:
        scores = norms.index_select(1, heads.to(device)).mean(1)[valid]
    stats = torch.stack(stats) if stats else torch.empty(0, len(BOUNDARY_FIELDS), device=device,
                                                         dtype=key.dtype)
    return scores, state, stats


def boundary_statistics(before, after, write, log_decay):
    """Full-layer Frobenius drift terms; ``write`` is the carried transported write."""
    decay_part = (log_decay.exp() - 1)[:, None, None] * before                   # (A - 1) S_before
    return torch.stack(((after - before).norm(), write.norm(), before.norm(), decay_part.norm(),
                        (decay_part * write).sum(), after.norm()))


@torch.no_grad()
def residual_block_scan(key, value, g, beta, state, blocks, value_per_key):
    """Unscaled residual block solve; no division by beta.

    Returns per-token, per-value-head FP tensors [N, Hv]: residual norm
    ``||e||``, decayed-prior prediction norm ``||vhat||``, ``vhat . v``; and
    the final carried state.  ``vhat`` is the pre-write read, never S_i^T k_i.
    """
    count, valid, kb, vb, bb, c, decay_pairs = _tile(key, value, g, beta, blocks, value_per_key)
    device, block_len = key.device, SCORE_BLOCK_TOKENS
    coupling = bb[..., None, :] * decay_pairs * (kb @ kb.transpose(-1, -2))     # beta in column j
    system = coupling + torch.eye(block_len, device=device, dtype=key.dtype)
    decayed_keys = c.exp()[..., None] * kb                                       # exp(c_t) k_t
    solved = torch.linalg.solve_triangular(system, torch.cat((vb, decayed_keys), -1),
                                           upper=False, unitriangular=True)
    width = vb.shape[-1]
    u, w = solved[..., :width], solved[..., width:]
    last = c[..., -1]
    decay = last.exp()
    keys_to_end = (last[..., None] - c).exp()[..., None] * kb
    shape = (count, kb.shape[1], block_len)
    error_norm, prediction_norm, dot = (torch.empty(shape, device=device, dtype=key.dtype)
                                        for _ in range(3))
    for row in range(count):
        error = torch.baddbmm(u[row], w[row], state, alpha=-1.0)                 # [Hv, L, Dv]
        prediction = torch.baddbmm(coupling[row] @ error, decayed_keys[row], state)
        error_norm[row] = error.norm(dim=-1)
        prediction_norm[row] = prediction.norm(dim=-1)
        dot[row] = (prediction * vb[row]).sum(-1)
        delta = bb[row][..., None] * error
        state = torch.baddbmm(decay[row][:, None, None] * state,
                              keys_to_end[row].transpose(-1, -2), delta)
    flat = [x.transpose(1, 2)[valid] for x in (error_norm, prediction_norm, dot)]  # [N, Hv]
    return (*flat, state)


def cosine_error(dot, prediction_norm, value_norm, epsilon=1e-6, zero_value_threshold=1e-6):
    """HAM-cos-local@M5 error ``1 - clip(vhat.v / (|vhat||v| + eps), -1, 1)``.

    A value with ``|v| <= threshold`` scores zero (flagged); a zero prediction
    with a nonzero value scores one.  Nonfinite inputs raise, never become priorities.
    """
    for tensor in (dot, prediction_norm, value_norm):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError('Nonfinite cosine-error input')
    error = 1 - (dot / (prediction_norm * value_norm + epsilon)).clamp(-1, 1)
    zero_value = value_norm <= zero_value_threshold
    return torch.where(zero_value, torch.zeros((), dtype=error.dtype, device=error.device), error), zero_value


def drift_priority(stats, statistic, floor):
    """S8-N/S8-W frame priority: numerator / max(||S_before||_F, eps_l), FP32."""
    if stats.ndim != 2 or stats.shape[1] != len(BOUNDARY_FIELDS):
        raise ValueError('Expected [frames, boundary fields] statistics')
    if not bool(torch.isfinite(stats).all()):
        raise ValueError('Nonfinite drift statistics')
    column = {'net_state_drift': 0, 'transported_write': 1}[statistic]
    if not floor > 0:
        raise ValueError('Drift denominator floor must be positive')
    return stats[:, column] / torch.clamp_min(stats[:, 2], floor)


def frame_mean_priority(visual_scores):
    """S7: arithmetic mean of a frame's 300 FP32 S0 scores, exactly rounded in float64.

    ``math.fsum`` makes the value independent of reduction order, so offline
    plans and serving promotions use bitwise-identical frame priorities.
    """
    values = np.asarray(visual_scores)
    if values.dtype != np.float32 or values.shape != (VISUAL_TOKENS_PER_GROUP,):
        raise TypeError('Frame mean needs the 300 FP32 visual scores of one frame')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite frame score')
    return math.fsum(values.astype(np.float64).tolist()) / VISUAL_TOKENS_PER_GROUP


def token_score(statistic, head_reduction, beta, value_norm, error_norm=None, cosine=None):
    """Per-token FP32 priority from per-head statistics [T, Hv]."""
    if statistic == 'error_norm':
        per_head = error_norm
    elif statistic == 'gated_value_norm':
        per_head = beta * value_norm
    elif statistic == 'value_norm':
        per_head = value_norm
    elif statistic == 'cosine_error':
        per_head = cosine
    elif statistic == 'gate_complement_residual':
        per_head = (1 - beta) * error_norm
    else:
        raise ValueError(f'No token statistic {statistic!r}')
    if head_reduction == 'mean':
        return per_head.mean(-1)
    if head_reduction == 'min':
        return per_head.min(-1).values
    raise ValueError(f'Unsupported head reduction {head_reduction!r}')


class GdnSweepScorer:
    """Carried score-helper state for one scoring GDN layer, one episode and one arm.

    Segmentation is identical to ``CommittedWriteScorer``: the prefix, each
    observation block and each complete response are separate segments; the
    helper state is never reset within an episode.  ``score_sequence`` (offline)
    and ``append`` (serving) return ``(token_scores, frame_stats)``.
    """

    def __init__(self, spec, gdn_layer, decoder_layer, device, max_super_block_tokens=4096):
        mixer = _gdn(decoder_layer)
        self.spec, self.gdn_layer, self.backend = spec, gdn_layer, spec.score_backend
        self.legacy = None
        if self.backend == LEGACY_BACKEND:
            self.legacy = CommittedWriteScorer(decoder_layer, device, max_super_block_tokens)
            return
        if self.backend not in (HEADWISE_BACKEND, BOUNDARY_BACKEND, RESIDUAL_BACKEND, PREPARE_BACKEND):
            raise ValueError(f'No GDN scorer for backend {self.backend!r}')
        self.layer = decoder_layer
        self.value_per_key = mixer.num_v_heads // mixer.num_k_heads
        self.conv_tail = empty_conv_tail(decoder_layer, device)
        self.state = torch.zeros(mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim,
                                 device=device, dtype=torch.float32)
        self.max_super_block_tokens = max_super_block_tokens
        self.pending = []
        self.heads = None
        if self.backend == HEADWISE_BACKEND:
            self.heads = torch.tensor(spec.slow_heads(gdn_layer), dtype=torch.long, device=device)

    def _prepare(self, hidden):
        key, value, g, beta, self.conv_tail = prepare_gdn_inputs(self.layer, hidden, self.conv_tail)
        return key, value, g, beta

    def _scan(self, prepared, segments, tracked_segments):
        """Scores for one super-block; ``tracked_segments`` index into ``segments``."""
        key, value, g, beta = prepared
        blocks, rows = [], []
        for start, end in segments:
            first = len(blocks)
            blocks.extend(internal_blocks([(start, end)]))
            rows.append((first, len(blocks) - 1))
        tracked = [rows[i] for i in tracked_segments]
        spec = self.spec
        if self.backend == PREPARE_BACKEND:
            return token_score(spec.statistic, spec.head_reduction, beta, value.norm(dim=-1)), None
        if self.backend in (HEADWISE_BACKEND, BOUNDARY_BACKEND):
            scores, self.state, stats = headwise_committed_scan(
                key, value, g, beta, self.state, blocks, self.value_per_key, heads=self.heads,
                tracked=tracked if self.backend == BOUNDARY_BACKEND else ())
            if self.backend == HEADWISE_BACKEND:
                return scores, None
            return None, stats
        error_norm, prediction_norm, dot, self.state = residual_block_scan(
            key, value, g, beta, self.state, blocks, self.value_per_key)
        cosine = None
        if spec.statistic == 'cosine_error':
            cosine, _ = cosine_error(dot, prediction_norm, value.norm(dim=-1),
                                     spec.cosine_epsilon, spec.zero_value_threshold)
        return token_score(spec.statistic, spec.head_reduction, beta, value.norm(dim=-1),
                           error_norm, cosine), None

    @torch.no_grad()
    def score_sequence(self, hidden, segments, observation_segments=()):
        """Offline: (token scores [T] or None, frame stats [len(observation_segments), 6] or None)."""
        hidden = hidden.reshape(-1, hidden.shape[-1])
        if segments[0][0] != 0 or segments[-1][1] != hidden.shape[0]:
            raise ValueError('Score segments must cover the whole sequence')
        if self.legacy is not None:
            return self.legacy.score_sequence(hidden, segments), None
        tokens = (None if self.backend == BOUNDARY_BACKEND else
                  torch.empty(hidden.shape[0], device=hidden.device, dtype=torch.float32))
        observed = {index: slot for slot, index in enumerate(observation_segments)}
        if len(observed) != len(observation_segments):
            raise ValueError('Duplicate observation segment')
        frames = [None] * len(observation_segments)
        batch, start = [], segments[0][0]
        for index, segment in enumerate(segments):
            if batch and segment[1] - start > self.max_super_block_tokens:
                self._score_range(hidden, start, batch, tokens, frames, observed)
                batch, start = [], segment[0]
            batch.append((index, segment))
        self._score_range(hidden, start, batch, tokens, frames, observed)
        if self.backend != BOUNDARY_BACKEND:
            return tokens, None
        if any(f is None for f in frames):
            raise RuntimeError('Missing observation boundary statistics')
        return None, (torch.stack(frames) if frames else
                      torch.empty(0, len(BOUNDARY_FIELDS), device=hidden.device))

    def _score_range(self, hidden, start, batch, tokens, frames, observed):
        end = batch[-1][1][1]
        prepared = self._prepare(hidden[start:end])
        relative = [(a - start, b - start) for _, (a, b) in batch]
        tracked = [slot for slot, (index, _) in enumerate(batch) if index in observed]
        scores, stats = self._scan(prepared, relative, tracked)
        if scores is not None:
            tokens[start:end] = scores
        if stats is not None:
            for row, slot in enumerate(tracked):
                frames[observed[batch[slot][0]]] = stats[row]

    # ---- serving: identical segmentation, one append at a time ----------
    @torch.no_grad()
    def append(self, hidden, kind):
        """``kind``: prefix | observation | response_token | response_close.

        Returns ``None`` for an unclosed response token, otherwise
        (token scores or None, frame stats [1, 6] for an observation or None).
        """
        hidden = hidden.reshape(-1, hidden.shape[-1])
        if self.legacy is not None:
            scores = self.legacy.append(hidden, kind)
            return None if scores is None else (scores, None)
        prepared = self._prepare(hidden)
        if kind in ('prefix', 'observation'):
            if self.pending:
                raise RuntimeError('Unclosed response before a new segment')
            return self._scan(prepared, [(0, hidden.shape[0])], [0] if kind == 'observation' else [])
        if kind not in ('response_token', 'response_close'):
            raise ValueError(f'Unknown score segment kind {kind!r}')
        self.pending.append(prepared)
        if kind == 'response_token':
            return None
        joined = [torch.cat(parts, 0) for parts in zip(*self.pending)]
        self.pending = []
        return self._scan(joined, [(0, joined[0].shape[0])], [])


# ---- gdn_raw_v2 raw records and learned-ranker encoders (Phase B) --------
@torch.no_grad()
def raw_features(key, value, g, beta, state, blocks, value_per_key, epsilon=1e-6,
                 zero_value_threshold=1e-6):
    """``gdn_raw_v2`` FP32 per-token, per-value-head scalars (no head averaging)."""
    error_norm, prediction_norm, dot, state = residual_block_scan(
        key, value, g, beta, state, blocks, value_per_key)
    value_norm = value.norm(dim=-1)
    cosine, zero_value = cosine_error(dot, prediction_norm, value_norm, epsilon, zero_value_threshold)
    key_norm_sq = (key * key).sum(-1).repeat_interleave(value_per_key, dim=1)
    record = dict(beta=beta, log_decay_g=g, value_norm=value_norm, error_norm=error_norm,
                  committed_norm=beta * error_norm, prediction_norm=prediction_norm,
                  prediction_dot_value=dot, cosine_error=cosine, key_norm_sq=key_norm_sq,
                  prediction_near_zero=prediction_norm <= zero_value_threshold,
                  value_near_zero=zero_value)
    return dict(schema=RAW_FEATURE_SCHEMA, fields=record), state


def encode_features(raw, schema):
    """Per-head channels in the declared order, [T, Hv, C]; never reinterprets old arrays."""
    if raw.get('schema') != RAW_FEATURE_SCHEMA:
        raise ValueError(f'{schema} needs {RAW_FEATURE_SCHEMA} raw records')
    f = raw['fields']
    channels = [torch.log(torch.clamp_min(f['beta'], 1e-6)),
                torch.log(torch.clamp_min(-f['log_decay_g'], 1e-8)),
                torch.log(torch.clamp_min(f['value_norm'], 1e-6))]
    if schema == F0_SCHEMA:
        pass
    elif schema == F1_SCHEMA:
        channels += [torch.log(torch.clamp_min(f['error_norm'], 1e-6)),
                     torch.log(torch.clamp_min(f['prediction_norm'], 1e-6)), f['cosine_error']]
    elif schema == F1_LEGACY_SCHEMA:          # analysis only; never a valid F1_v2 input
        channels += [torch.log(torch.clamp_min(f['error_norm'], 1e-6)),
                     torch.log(torch.clamp_min(f['committed_norm'], 1e-6)), f['cosine_error']]
    else:
        raise ValueError(f'Unknown feature schema {schema!r}')
    return dict(schema=schema, values=torch.stack(channels, -1))


def require_schema(encoded, schema):
    """A ranker consumes exactly its declared schema."""
    if encoded.get('schema') != schema:
        raise ValueError(f'Feature schema {encoded.get("schema")!r} is not {schema!r}')
    return encoded['values']


class GdnFeatureExtractor:
    """``gdn_raw_v2`` fields of one GDN layer for one episode (learned rankers, labels).

    Same native preparation, segmentation, super-block batching and carried
    conv tail/state as ``GdnSweepScorer``, so the label collector, the
    offline training plans and serving promotions read the same features.
    ``with_state=False`` (F0: beta, g, ||v||) never reads recurrent state.
    ``sequence`` (offline) and ``append`` (serving) return field -> FP32 [N, Hv].
    """

    STATELESS_FIELDS = ('beta', 'log_decay_g', 'value_norm')

    def __init__(self, decoder_layer, device, with_state=True, max_super_block_tokens=4096):
        mixer = _gdn(decoder_layer)
        self.layer, self.with_state = decoder_layer, with_state
        self.value_per_key = mixer.num_v_heads // mixer.num_k_heads
        self.conv_tail = empty_conv_tail(decoder_layer, device)
        self.state = torch.zeros(mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim,
                                 device=device, dtype=torch.float32)
        self.max_super_block_tokens = max_super_block_tokens
        self.pending = []

    def _prepare(self, hidden):
        key, value, g, beta, self.conv_tail = prepare_gdn_inputs(self.layer, hidden, self.conv_tail)
        return key, value, g, beta

    def _fields(self, prepared, segments):
        key, value, g, beta = prepared
        if not self.with_state:
            return dict(beta=beta, log_decay_g=g, value_norm=value.norm(dim=-1))
        blocks = []
        for start, end in segments:
            blocks.extend(internal_blocks([(start, end)]))
        raw, self.state = raw_features(key, value, g, beta, self.state, blocks, self.value_per_key)
        return raw['fields']

    @torch.no_grad()
    def sequence(self, hidden, segments):
        hidden = hidden.reshape(-1, hidden.shape[-1])
        if segments[0][0] != 0 or segments[-1][1] != hidden.shape[0]:
            raise ValueError('Feature segments must cover the whole sequence')
        parts, batch, start = [], [], segments[0][0]
        for segment in segments:
            if batch and segment[1] - start > self.max_super_block_tokens:
                parts.append(self._range(hidden, start, batch))
                batch, start = [], segment[0]
            batch.append(segment)
        parts.append(self._range(hidden, start, batch))
        return {name: torch.cat([p[name] for p in parts], 0) for name in parts[0]}

    def _range(self, hidden, start, batch):
        end = batch[-1][1]
        return self._fields(self._prepare(hidden[start:end]), [(a - start, b - start) for a, b in batch])

    @torch.no_grad()
    def append(self, hidden, kind):
        """``kind``: prefix | observation | response_token | response_close; None while a response is open."""
        hidden = hidden.reshape(-1, hidden.shape[-1])
        prepared = self._prepare(hidden)
        if kind in ('prefix', 'observation'):
            if self.pending:
                raise RuntimeError('Unclosed response before a new segment')
            return self._fields(prepared, [(0, hidden.shape[0])])
        if kind not in ('response_token', 'response_close'):
            raise ValueError(f'Unknown score segment kind {kind!r}')
        self.pending.append(prepared)
        if kind == 'response_token':
            return None
        joined = [torch.cat(parts, 0) for parts in zip(*self.pending)]
        self.pending = []
        return self._fields(joined, [(0, joined[0].shape[0])])
