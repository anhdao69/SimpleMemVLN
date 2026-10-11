"""Text-only prefix + eight-step FlashAttention, no dense production mask."""
import torch
from transformers import AttentionInterface


def _window_groups(prefix, spans):
    return [(0, prefix, None)] + [
        (start, end, spans[max(0, t - 7)][0])
        for t, (start, end) in enumerate(spans)
    ]


def _pack_windows(query, key, value, groups, prefix):
    """Pack independent causal windows, bounded by the selected group batch."""
    qs, ks, vs, qlens, klens = [], [], [], [], []
    for start, end, first in groups:
        qs.append(query[0, :, start:end].transpose(0, 1))
        if first is None:
            k, v = key[0, :, :prefix], value[0, :, :prefix]
        else:
            k = torch.cat((key[0, :, :prefix], key[0, :, first:end]), 1)
            v = torch.cat((value[0, :, :prefix], value[0, :, first:end]), 1)
        ks.append(k.transpose(0, 1))
        vs.append(v.transpose(0, 1))
        qlens.append(end - start)
        klens.append(k.shape[1])
    def cumulative(lengths):
        return torch.tensor([0] + list(__import__('itertools').accumulate(lengths)),
                            dtype=torch.int32, device=query.device)
    return (torch.cat(qs), torch.cat(ks), torch.cat(vs),
            cumulative(qlens), cumulative(klens), max(qlens), max(klens), klens)


def _varlen(q, k, v, cuq, cuk, maxq, maxk, scaling):
    from flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func(q, k, v, cuq, cuk, maxq, maxk,
                                 dropout_p=0.0, softmax_scale=scaling, causal=True)


class _BatchedWindowAttention(torch.autograd.Function):
    """Recompute bounded batches of windows; retain only original Q/K/V.

    FlashAttention's bottom-right causal alignment applies separately to each
    packed window. Prefix and preceding seven groups are repeated only inside
    the current batch. Explicit gradient scatter retains complete-episode BPTT.
    """

    @staticmethod
    def forward(ctx, query, key, value, scaling, prefix, spans, batch_steps):
        ctx.save_for_backward(query, key, value)
        ctx.scaling, ctx.prefix = scaling, prefix
        ctx.groups, ctx.batch_steps = _window_groups(prefix, spans), batch_steps
        output = query.new_empty((1, query.shape[2], query.shape[1], value.shape[-1]))
        for offset in range(0, len(ctx.groups), batch_steps):
            groups = ctx.groups[offset:offset + batch_steps]
            q, k, v, cuq, cuk, maxq, maxk, _ = _pack_windows(query, key, value, groups, prefix)
            output[0, groups[0][0]:groups[-1][1]] = _varlen(q, k, v, cuq, cuk, maxq, maxk, scaling)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        query, key, value = ctx.saved_tensors
        dq, dk, dv = (torch.zeros_like(x) for x in (query, key, value))
        for offset in reversed(range(0, len(ctx.groups), ctx.batch_steps)):
            groups = ctx.groups[offset:offset + ctx.batch_steps]
            with torch.enable_grad():
                packed = _pack_windows(query, key, value, groups, ctx.prefix)
                q, k, v = (x.detach().requires_grad_() for x in packed[:3])
                cuq, cuk, maxq, maxk, klens = packed[3:]
                out = _varlen(q, k, v, cuq, cuk, maxq, maxk, ctx.scaling)
                gq, gk, gv = torch.autograd.grad(out, (q, k, v),
                    grad_output[0, groups[0][0]:groups[-1][1]].contiguous())
            dq[0, :, groups[0][0]:groups[-1][1]].copy_(gq.transpose(0, 1))
            kcursor = sum(klens)
            for (start, end, first), klen in reversed(list(zip(groups, klens))):
                kcursor -= klen
                kg, vg = (g[kcursor:kcursor+klen].transpose(0, 1) for g in (gk, gv))
                dk[0, :, :ctx.prefix].add_(kg[:, :ctx.prefix])
                dv[0, :, :ctx.prefix].add_(vg[:, :ctx.prefix])
                if first is not None:
                    dk[0, :, first:end].add_(kg[:, ctx.prefix:])
                    dv[0, :, first:end].add_(vg[:, ctx.prefix:])
        return dq, dk, dv, None, None, None, None


def _episodic_groups(prefix, spans, recent, plan):
    """(query start, end, first recent key, sorted episodic IDs) per packed sequence."""
    if plan is not None:
        plan.verify()
        if plan.groups != len(spans) or plan.recent_groups != recent:
            raise ValueError('Episodic plan does not match this episode/window')
    groups = [(0, prefix, None, None)]
    for t, (start, end) in enumerate(spans):
        ids = None if plan is None else plan.group_ids(t)
        groups.append((start, end, spans[max(0, t - recent + 1)][0], ids))
    return groups


def _key_index(groups, prefix):
    """Concatenated original-token K/V source indices for one bounded batch."""
    import numpy as np
    pieces, klens = [], []
    for start, end, first, ids in groups:
        if first is None:
            piece = np.arange(prefix, dtype=np.int64)
        else:
            old = ids if ids is not None else np.empty(0, np.int64)
            if old.size and (int(old[0]) < prefix or int(old[-1]) >= first):
                raise ValueError('Episodic rows must lie between prefix and recent window')
            piece = np.concatenate((np.arange(prefix, dtype=np.int64), old,
                                    np.arange(first, end, dtype=np.int64)))
        pieces.append(piece)
        klens.append(piece.size)
    return np.concatenate(pieces), klens


def _pack_gathered(query, key, value, groups, prefix):
    import numpy as np
    index, klens = _key_index(groups, prefix)
    device = query.device
    index = torch.from_numpy(index).to(device, non_blocking=True)
    q = query[0, :, groups[0][0]:groups[-1][1]].transpose(0, 1)
    k = key[0].index_select(1, index).transpose(0, 1)
    v = value[0].index_select(1, index).transpose(0, 1)
    qlens = [end - start for start, end, _, _ in groups]
    def cumulative(lengths):
        return torch.tensor(np.concatenate(([0], np.cumsum(lengths))), dtype=torch.int32, device=device)
    return (q.contiguous(), k.contiguous(), v.contiguous(), cumulative(qlens), cumulative(klens),
            max(qlens), max(klens), klens, index)


class _EpisodicWindowAttention(torch.autograd.Function):
    """Prefix + immutable episodic rows + R recent groups, original positions kept.

    Every packed sequence's current query block is its key suffix, so FA2's
    bottom-right causal alignment exposes exactly the current causal prefix.
    Only original Q/K/V and the immutable plan are saved.  Backward recomputes
    one bounded batch at a time and scatter-adds K/V gradients by original
    token ID into FP32 buffers, so an old row read by many later windows (and
    the prefix) receives the summed contribution of every window.
    """

    @staticmethod
    def forward(ctx, query, key, value, scaling, prefix, spans, recent, plan, batch_steps):
        ctx.save_for_backward(query, key, value)
        ctx.scaling, ctx.prefix, ctx.batch_steps, ctx.plan = scaling, prefix, batch_steps, plan
        ctx.groups = _episodic_groups(prefix, spans, recent, plan)
        output = query.new_empty((1, query.shape[2], query.shape[1], value.shape[-1]))
        for offset in range(0, len(ctx.groups), batch_steps):
            groups = ctx.groups[offset:offset + batch_steps]
            q, k, v, cuq, cuk, maxq, maxk, _, _ = _pack_gathered(query, key, value, groups, prefix)
            output[0, groups[0][0]:groups[-1][1]] = _varlen(q, k, v, cuq, cuk, maxq, maxk, scaling)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        query, key, value = ctx.saved_tensors
        if ctx.plan is not None:
            ctx.plan.verify()
        prefix = ctx.prefix
        dq = torch.zeros_like(query)
        # BF16 inputs accumulate in FP32; FP64 reference inputs keep FP64.
        accumulate = torch.promote_types(key.dtype, torch.float32)
        dk = torch.zeros(key.shape[1:], dtype=accumulate, device=key.device)
        dv = torch.zeros(value.shape[1:], dtype=accumulate, device=value.device)
        for offset in reversed(range(0, len(ctx.groups), ctx.batch_steps)):
            groups = ctx.groups[offset:offset + ctx.batch_steps]
            with torch.enable_grad():
                packed = _pack_gathered(query, key, value, groups, prefix)
                q, k, v = (x.detach().requires_grad_() for x in packed[:3])
                cuq, cuk, maxq, maxk, klens, index = packed[3:]
                out = _varlen(q, k, v, cuq, cuk, maxq, maxk, ctx.scaling)
                gq, gk, gv = torch.autograd.grad(
                    out, (q, k, v), grad_output[0, groups[0][0]:groups[-1][1]].contiguous())
            dq[0, :, groups[0][0]:groups[-1][1]].copy_(gq.transpose(0, 1))
            gk, gv = gk.to(accumulate).transpose(0, 1), gv.to(accumulate).transpose(0, 1)
            cursor = sum(klens)
            for (start, end, first, ids), klen in reversed(list(zip(groups, klens))):
                cursor -= klen
                for grad, target in ((gk, dk), (gv, dv)):
                    piece = grad[:, cursor:cursor + klen]
                    target[:, :prefix].add_(piece[:, :prefix])
                    if first is None:
                        continue
                    old = 0 if ids is None else ids.size
                    if old:
                        # IDs are unique within one window: deterministic scatter.
                        target.index_add_(1, index[cursor + prefix:cursor + prefix + old],
                                          piece[:, prefix:prefix + old])
                    target[:, first:end].add_(piece[:, prefix + old:])
        return dq, dk.to(key.dtype)[None], dv.to(value.dtype)[None], None, None, None, None, None, None


def _flash(q, k, v, scaling):
    from flash_attn import flash_attn_func
    return flash_attn_func(
        q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(),
        v.transpose(1, 2).contiguous(), dropout_p=0.0,
        softmax_scale=scaling, causal=True,
    )


class _WindowAttention(torch.autograd.Function):
    """Recompute one window at a time; accumulate into shared gradient buffers.

    Ordinary per-step slicing creates full-sequence gradient temporaries and
    keeps overlapping KV copies alive during decoder checkpoint replay. Saving
    Q/K/V once and accumulating slice gradients explicitly bounds this scratch
    space without truncating any temporal gradient.
    """

    @staticmethod
    def forward(ctx, query, key, value, scaling, prefix, spans):
        ctx.save_for_backward(query, key, value)
        ctx.scaling, ctx.prefix, ctx.spans = scaling, prefix, spans
        parts = [_flash(query[..., :prefix, :], key[..., :prefix, :],
                        value[..., :prefix, :], scaling)]
        for t, (start, end) in enumerate(spans):
            first = spans[max(0, t - 7)][0]
            k = torch.cat((key[..., :prefix, :], key[..., first:end, :]), -2)
            v = torch.cat((value[..., :prefix, :], value[..., first:end, :]), -2)
            parts.append(_flash(query[..., start:end, :], k, v, scaling))
        return torch.cat(parts, dim=1)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        query, key, value = ctx.saved_tensors
        prefix = ctx.prefix
        dq, dk, dv = (torch.zeros_like(x) for x in (query, key, value))
        groups = [(0, prefix, None)] + [
            (start, end, ctx.spans[max(0, t - 7)][0])
            for t, (start, end) in enumerate(ctx.spans)
        ]
        for start, end, first in reversed(groups):
            with torch.enable_grad():
                q = query[..., start:end, :].detach().requires_grad_()
                if first is None:
                    k = key[..., :prefix, :].detach().requires_grad_()
                    v = value[..., :prefix, :].detach().requires_grad_()
                else:
                    k = torch.cat((key[..., :prefix, :], key[..., first:end, :]), -2).detach().requires_grad_()
                    v = torch.cat((value[..., :prefix, :], value[..., first:end, :]), -2).detach().requires_grad_()
                out = _flash(q, k, v, ctx.scaling)
                gq, gk, gv = torch.autograd.grad(
                    out, (q, k, v), grad_output[:, start:end].contiguous()
                )
            dq[..., start:end, :].copy_(gq)
            dk[..., :prefix, :].add_(gk[..., :prefix, :])
            dv[..., :prefix, :].add_(gv[..., :prefix, :])
            if first is not None:
                dk[..., first:end, :].add_(gk[..., prefix:, :])
                dv[..., first:end, :].add_(gv[..., prefix:, :])
        return dq, dk, dv, None, None, None


def step_flash_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling=None,
    dropout=0.0,
    step_plan=None,
    stream_append=False,
    window_attention_batch_steps=1,
    episodic_recent_groups=None,
    episodic_plan=None,
    **kwargs
):
    if attention_mask is not None or query.shape[0] != 1 or dropout:
        raise ValueError("Window8 requires unpadded B1 and dropout=0")

    def attend(q, k, v):
        return _flash(q, k, v, scaling)

    if stream_append:
        # The session already removed expired groups before native cache update.
        return attend(query, key, value), None
    if step_plan is None:
        raise ValueError("Offline Window8 requires immutable prefix and step spans")
    prefix_length, spans = step_plan
    if query.shape[-2] != key.shape[-2]:
        raise ValueError("Offline attention cannot consume a serving cache")
    cursor = prefix_length
    for start, end in spans:
        if start != cursor or end <= start:
            raise ValueError("Step plan must cover the full sequence exactly")
        cursor = end
    if cursor != query.shape[-2]:
        raise ValueError("Uncovered query tokens")
    if type(window_attention_batch_steps) is not int or not 1 <= window_attention_batch_steps <= 64:
        raise ValueError("Window attention batch steps must be an integer in [1,64]")
    if episodic_recent_groups is not None:
        if type(episodic_recent_groups) is not int or episodic_recent_groups < 1:
            raise ValueError('Recent groups must be a positive integer')
        if episodic_plan is not None and episodic_plan.layer_idx != getattr(module, 'layer_idx', None):
            raise ValueError('Episodic plan routed to the wrong attention layer')
        return _EpisodicWindowAttention.apply(
            query, key, value, scaling, prefix_length, tuple(spans), episodic_recent_groups,
            episodic_plan, window_attention_batch_steps), None
    if episodic_plan is not None:
        raise ValueError('Episodic plan supplied without the episodic window')
    if window_attention_batch_steps > 1:
        return _BatchedWindowAttention.apply(query, key, value, scaling, prefix_length,
                                             tuple(spans), window_attention_batch_steps), None
    return _WindowAttention.apply(query, key, value, scaling, prefix_length,
                                  tuple(spans)), None


def register_window_attention():
    # HF 5.11 interprets any name containing "flash" as an external FA loader.
    AttentionInterface.register("simplememvln_step_attention", step_flash_attention)
