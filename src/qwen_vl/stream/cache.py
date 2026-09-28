"""Keep GDN in FP32 and evict only full-attention token entries."""
import torch
from transformers.cache_utils import DynamicCache, LinearAttentionLayer


class FP32LinearAttentionLayer(LinearAttentionLayer):
    def update_recurrent_state(self, recurrent_states, **kwargs):
        if not self.is_recurrent_states_initialized:
            self.recurrent_states = torch.empty_like(
                recurrent_states, dtype=torch.float32
            )
            self.is_recurrent_states_initialized = True
        self.recurrent_states.copy_(recurrent_states)
        return self.recurrent_states


def make_stream_cache_fp32(text_config):
    cache = DynamicCache(config=text_config)
    for i, kind in enumerate(text_config.layer_types):
        if kind == "linear_attention":
            cache.layers[i] = FP32LinearAttentionLayer()
    return cache


def kv_length(cache):
    lengths = [
        layer.keys.shape[-2]
        for layer in cache.layers
        if hasattr(layer, "keys") and layer.keys is not None
    ]
    if lengths and len(set(lengths)) != 1:
        raise RuntimeError(f"Inconsistent full-attention lengths: {lengths}")
    return lengths[0] if lengths else 0


def evict_kv(cache, prefix_length, remove_tokens):
    if remove_tokens < 0:
        raise ValueError("Negative eviction")
    for layer in cache.layers:
        if hasattr(layer, "keys") and layer.keys is not None:
            if prefix_length + remove_tokens > layer.keys.shape[-2]:
                raise ValueError("Eviction exceeds resident history")
            for field in ("keys", "values"):
                value = getattr(layer, field)
                setattr(
                    layer,
                    field,
                    torch.cat(
                        (
                            value[..., :prefix_length, :],
                            value[..., prefix_length + remove_tokens :, :],
                        ),
                        dim=-2,
                    ),
                )


def assert_state_dtypes(cache):
    for layer in cache.layers:
        if (
            isinstance(layer, LinearAttentionLayer)
            and layer.is_recurrent_states_initialized
        ):
            if layer.recurrent_states.dtype != torch.float32:
                raise RuntimeError("Recurrent state lost FP32 precision")
