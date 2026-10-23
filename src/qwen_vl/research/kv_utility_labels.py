"""Exact fixed-query local LOO utility of one native visual KV row (``SHADOW_FIXED_QUERY_LOO``).

For one query at one full-attention layer (post-RoPE query, native sigmoid
output gate ``z``, output projection ``W_O``) and a visible key set ``C``,
removing logical row ``i`` from every query head that reads it changes head
``h``'s output by the stable form

    delta_o_h = a_hi (V_hi - o_h^(-i)),    a_hi = softmax_C(logits_h)[i],

where ``o_h^(-i)`` is the masked-softmax output without ``i``, and the module
output by ``delta_y = W_O concat_h(z_h * delta_o_h)`` (an o_proj bias cancels).
The utility is ``||delta_y||_2 / sqrt(d_model)``: cross-head contributions are
summed before the norm, never replaced by per-head magnitudes.  Query head
``h`` reads KV head ``h // (Hq / Hkv)`` (the native ``repeat_kv`` order), so a
logical row is removed from its whole GQA group at once, as M5 evicts it.

Shadow context (plan Section 10.3): a candidate resident in the reference
bank is removed from the reference context itself; an eligible non-resident
candidate replaces one displaced bank row (chosen independently of every
utility), so occupancy never exceeds the reference bank.  All arithmetic is
FP64 on the archived native BF16 Q/K/V/gate.  ``a/(1-a)`` closed forms are
used only while ``a <= 0.5`` (at most one bit lost to cancellation); larger
attention probabilities are recomputed with an explicit masked softmax.
"""
import math
import torch

STABLE_THRESHOLD = 0.5


def _grouped(query, kv_heads):
    heads, width = query.shape
    if heads % kv_heads:
        raise ValueError('Query heads must be a multiple of KV heads')
    return query.reshape(kv_heads, heads // kv_heads, width)


def head_logits(query, keys, scale):
    """[Hq, n] scaled logits; ``query`` [Hq, D], ``keys`` [n, Hkv, D]."""
    groups = keys.shape[1]
    return torch.einsum('grd,ngd->grn', _grouped(query, groups), keys).reshape(query.shape[0], -1) * scale


def head_mix(weights, values):
    """[Hq, D] = sum_n weights[h, n] V[n, kv(h)]."""
    groups = values.shape[1]
    mixed = torch.einsum('grn,ngd->grd', weights.reshape(groups, -1, weights.shape[-1]), values)
    return mixed.reshape(weights.shape[0], values.shape[-1])


def head_rows(values, query_heads):
    """KV rows [m, Hkv, D] expanded to their query heads [m, Hq, D]."""
    return values.repeat_interleave(query_heads // values.shape[1], dim=1)


def attention_output(query, keys, values, scale, keep=None):
    """Softmax attention output [Hq, D] of one query; ``keep`` [n] selects visible rows."""
    logits = head_logits(query, keys, scale)
    if keep is not None:
        if not bool(keep.any()):
            raise ValueError('A masked softmax needs at least one visible row')
        logits = logits.masked_fill(~keep[None], float('-inf'))
    return head_mix(torch.softmax(logits, -1), values)


def project(delta, gate, w_o):
    """``W_O concat_h(z_h * delta_h)``: [..., Hq, D] -> [..., d_model]."""
    return (gate * delta).flatten(-2) @ w_o.T


class FixedQueryContext:
    """One frozen query/layer reference context; every candidate shares its statistics.

    ``query``/``gate`` [Hq, D] (gate already passed through the native
    sigmoid), ``keys``/``values`` [n, Hkv, D] the reference visible rows,
    ``w_o`` [d_model, Hq*D].  All FP64.
    """

    def __init__(self, query, gate, keys, values, scale, w_o):
        for name, tensor in (('query', query), ('gate', gate), ('keys', keys), ('values', values), ('w_o', w_o)):
            if tensor.dtype != torch.float64:
                raise TypeError(f'{name} must be FP64')
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'Nonfinite {name}')
        if query.shape != gate.shape or keys.shape != values.shape or keys.shape[-1] != query.shape[-1]:
            raise ValueError('Inconsistent query/gate/key/value shapes')
        if w_o.shape[1] != query.numel():
            raise ValueError('Output projection does not match concatenated heads')
        self.query, self.gate, self.keys, self.values = query, gate, keys, values
        self.scale, self.w_o = float(scale), w_o
        self.heads, self.d_model = query.shape[0], w_o.shape[0]
        self.logits = head_logits(query, keys, scale)
        self.lse = torch.logsumexp(self.logits, -1)
        self.prob = (self.logits - self.lse[:, None]).exp()
        self.out = head_mix(self.prob, values)
        self.fallbacks = 0

    def output(self):
        """Reconstructed unmodified module output [d_model] (no o_proj bias)."""
        return project(self.out, self.gate, self.w_o)

    def _utility(self, delta):
        return project(delta, self.gate, self.w_o).norm(dim=-1) / math.sqrt(self.d_model)

    def _without(self, row):
        keep = torch.ones(self.keys.shape[0], dtype=torch.bool, device=self.keys.device)
        keep[row] = False
        return keep

    def resident(self, rows):
        """Raw utilities [r] of candidates already in the reference context (``rows`` index it)."""
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.keys.device).reshape(-1)
        if rows.numel() == 0:
            return torch.empty(0, dtype=torch.float64, device=self.keys.device)
        a = self.prob[:, rows].T                                              # [r, Hq]
        v = head_rows(self.values[rows], self.heads)                          # [r, Hq, D]
        o_minus = (self.out[None] - a[..., None] * v) / (1 - a)[..., None]
        for slot in (a > STABLE_THRESHOLD).any(-1).nonzero().flatten().tolist():
            o_minus[slot] = attention_output(self.query, self.keys, self.values, self.scale,
                                             self._without(int(rows[slot])))
            self.fallbacks += 1
        return self._utility(a[..., None] * (v - o_minus))

    def shadow(self, keys, values, displaced):
        """Raw utilities [e] of eligible candidates absent from the context.

        Each candidate replaces context row ``displaced`` (same occupancy);
        its probability against the remaining rows W is
        ``sigmoid(z_i - logsumexp_W)`` and ``o^(-i)`` is the output over W.
        """
        if keys.shape[0] == 0:
            return torch.empty(0, dtype=torch.float64, device=self.keys.device)
        if keys.dtype != torch.float64 or values.dtype != torch.float64:
            raise TypeError('Shadow candidates must be FP64')
        if not (bool(torch.isfinite(keys).all()) and bool(torch.isfinite(values).all())):
            raise ValueError('Nonfinite shadow candidate')
        displaced = int(displaced)
        if not 0 <= displaced < self.keys.shape[0]:
            raise IndexError('Displaced row outside the reference context')
        p_d = self.prob[:, displaced]
        if bool((p_d > STABLE_THRESHOLD).any()):
            keep = self._without(displaced)
            lse_w = torch.logsumexp(self.logits.masked_fill(~keep[None], float('-inf')), -1)
            o_w = attention_output(self.query, self.keys, self.values, self.scale, keep)
            self.fallbacks += 1
        else:
            v_d = head_rows(self.values[displaced:displaced + 1], self.heads)[0]
            lse_w = self.lse + torch.log1p(-p_d)
            o_w = (self.out - p_d[:, None] * v_d) / (1 - p_d)[:, None]
        z = head_logits(self.query, keys, self.scale).T                       # [e, Hq]
        a = torch.sigmoid(z - lse_w[None])
        v = head_rows(values, self.heads)
        return self._utility(a[..., None] * (v - o_w[None]))


def visible_rows(prefix, bank_ids, recent_start, query_position):
    """Original IDs visible to a pre-action query: prefix, bank (sorted), causal recent window."""
    import numpy as np
    bank = np.asarray(bank_ids, dtype=np.int64)
    if bank.size and (bank[0] < prefix or bank[-1] >= recent_start or np.any(np.diff(bank) <= 0)):
        raise ValueError('Bank rows must be sorted and lie between prefix and recent window')
    if not prefix <= recent_start <= query_position:
        raise ValueError('Query must lie in its recent window')
    return np.concatenate((np.arange(prefix, dtype=np.int64), bank,
                           np.arange(recent_start, query_position + 1, dtype=np.int64)))


def native_qkv(module, hidden, position_embeddings):
    """Recompute a native ``Qwen3_5Attention``'s post-RoPE Q, gate, K, V on the full sequence.

    Same projections, per-head norms, chunk order and rotary call as the
    pinned native forward (identical shapes, so identical kernels).  Returns
    query [Hq, L, D], gate [L, Hq, D] (pre-sigmoid), key/value [L, Hkv, D].
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as native
    length = hidden.shape[1]
    shape = (1, length, -1, module.head_dim)
    query, gate = torch.chunk(module.q_proj(hidden).view(1, length, -1, module.head_dim * 2), 2, dim=-1)
    query = module.q_norm(query.view(shape)).transpose(1, 2)
    key = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
    value = module.v_proj(hidden).view(shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = native.apply_rotary_pos_emb(query, key, cos, sin)
    return query[0], gate[0], key[0].transpose(0, 1), value[0].transpose(0, 1)
