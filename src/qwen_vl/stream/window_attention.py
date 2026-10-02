"""Text-only prefix + eight-step FlashAttention, no dense production mask."""
import torch
from transformers import AttentionInterface


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
    return _WindowAttention.apply(query, key, value, scaling, prefix_length,
                                  tuple(spans)), None


def register_window_attention():
    # HF 5.11 interprets any name containing "flash" as an external FA loader.
    AttentionInterface.register("simplememvln_step_attention", step_flash_attention)
