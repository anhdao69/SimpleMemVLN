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
    if window_attention_batch_steps > 1:
        return _BatchedWindowAttention.apply(query, key, value, scaling, prefix_length,
                                             tuple(spans), window_attention_batch_steps), None
    return _WindowAttention.apply(query, key, value, scaling, prefix_length,
                                  tuple(spans)), None


def register_window_attention():
    # HF 5.11 interprets any name containing "flash" as an external FA loader.
    AttentionInterface.register("simplememvln_step_attention", step_flash_attention)
