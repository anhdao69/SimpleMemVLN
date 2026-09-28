"""FP32 CPU oracle: independent of FA2/FLA GPU numerical tolerances."""
import torch
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as native
from qwen_vl.stream.cache import make_stream_cache_fp32


def test_fp32_native_mixed_chunk_continuation_and_temporal_gradient(monkeypatch):
    for name in [
        "FusedRMSNormGated",
        "chunk_gated_delta_rule",
        "fused_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ]:
        monkeypatch.setattr(native, name, None)
    torch.manual_seed(42)
    cfg = Qwen3_5TextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=128,
        layer_types=["linear_attention", "full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 3, 3],
        },
        attn_implementation="sdpa",
    )
    model = native.Qwen3_5TextModel(cfg).float().eval()
    assert (
        model.layers[0].linear_attn.chunk_gated_delta_rule
        is native.torch_chunk_gated_delta_rule
    )
    emb = torch.randn(1, 37, 64, requires_grad=True)
    pos = torch.arange(37).view(1, 1, -1).expand(4, 1, -1)
    full = model(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state
    full[0, -1, 0].backward()
    assert emb.grad[0, :8].abs().sum() > 0
    with torch.no_grad():
        cache = make_stream_cache_fp32(cfg)
        parts = []
        for s, e in [(0, 8), (8, 19), (19, 20), (20, 35), (35, 37)]:
            parts.append(
                model(
                    inputs_embeds=emb[:, s:e],
                    position_ids=pos[..., s:e],
                    past_key_values=cache,
                    use_cache=True,
                ).last_hidden_state
            )
        torch.testing.assert_close(torch.cat(parts, 1), full, atol=2e-5, rtol=2e-5)
        changed = emb.detach().clone()
        changed[:, 20:] += 10
        alternate = model(
            inputs_embeds=changed, position_ids=pos, use_cache=False
        ).last_hidden_state
        torch.testing.assert_close(
            alternate[:, :20], full[:, :20], atol=2e-5, rtol=2e-5
        )
