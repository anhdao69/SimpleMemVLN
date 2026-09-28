"""Text-only prefix + eight-step FlashAttention, no dense production mask."""
import torch
from transformers import AttentionInterface


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
    from flash_attn import flash_attn_func

    if attention_mask is not None or query.shape[0] != 1 or dropout:
        raise ValueError("Window8 requires unpadded B1 and dropout=0")

    def attend(q, k, v):
        return flash_attn_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            dropout_p=0.0,
            softmax_scale=scaling,
            causal=True,
        )

    if stream_append:
        # The session already removed expired groups before native cache update.
        return attend(query, key, value), None
    if step_plan is None:
        raise ValueError("Offline Window8 requires immutable prefix and step spans")
    prefix_length, spans = step_plan
    if query.shape[-2] != key.shape[-2]:
        raise ValueError("Offline attention cannot consume a serving cache")
    outputs = [
        attend(
            query[..., :prefix_length, :],
            key[..., :prefix_length, :],
            value[..., :prefix_length, :],
        )
    ]
    cursor = prefix_length
    for t, (start, end) in enumerate(spans):
        if start != cursor or end <= start:
            raise ValueError("Step plan must cover the full sequence exactly")
        first = spans[max(0, t - 7)][0]
        k = torch.cat((key[..., :prefix_length, :], key[..., first:end, :]), dim=-2)
        v = torch.cat((value[..., :prefix_length, :], value[..., first:end, :]), dim=-2)
        outputs.append(attend(query[..., start:end, :], k, v))
        cursor = end
    if cursor != query.shape[-2]:
        raise ValueError("Uncovered query tokens")
    return torch.cat(outputs, dim=1), None


def register_window_attention():
    # HF 5.11 interprets any name containing "flash" as an external FA loader.
    AttentionInterface.register("simplememvln_step_attention", step_flash_attention)
