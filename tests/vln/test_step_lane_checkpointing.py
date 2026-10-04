"""Actual pinned decoder path, with a CPU reference lane for algebraic checkpoint tests."""
import torch
import pytest
from torch.utils.checkpoint import checkpoint
from step_lane_test_utils import reference_kernel, tiny_spec


def test_actual_native_decoder_kwargs_survive_nonreentrant_checkpoint(monkeypatch):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    from transformers import Qwen3_5TextConfig
    import qwen_vl.models.install_step_lane as install
    from qwen_vl.models.parallel_step_lane import ParallelStepLane
    from types import SimpleNamespace
    from torch import nn
    # Actual decoder layer/normalization/MLP and native Torch GDN CPU equations.
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    for backend in ('FusedRMSNormGated','chunk_gated_delta_rule','fused_recurrent_gated_delta_rule','causal_conv1d_fn'):
        monkeypatch.setattr(native,backend,None)
    cfg=Qwen3_5TextConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                         layer_types=['linear_attention'],linear_num_key_heads=1,
                         linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8)
    decoder=Qwen3_5DecoderLayer(cfg,0)
    def factory(width,spec,**kwargs):
        kwargs.update(chunk_kernel=reference_kernel,recurrent_kernel=reference_kernel,test_only=True)
        return ParallelStepLane(width,spec,**kwargs)
    monkeypatch.setattr(install,'ParallelStepLane',factory)
    model=nn.Module()
    model.config=SimpleNamespace(text_config=cfg)
    model.backbone=nn.Module()
    model.backbone.model=nn.Module()
    model.backbone.model.language_model=nn.Module()
    model.backbone.model.language_model.layers=nn.ModuleList([decoder])
    install.install_step_lanes(model,tiny_spec(layers=(0,)),init_seed=19)
    with torch.no_grad():
        decoder.linear_attn.step_lane.out_proj.weight.normal_(std=.02)
    x=torch.randn(1,13,32,requires_grad=True)
    roles=torch.tensor([[0,0,1,1,2,3,1,2,3,1,1,2,3]],dtype=torch.uint8)
    kw=dict(position_embeddings=None,step_lane_roles=roles)
    direct=decoder(x,**kw)
    grads=torch.autograd.grad(direct.square().sum(),(x,*decoder.parameters()))
    actual=checkpoint(decoder,x,use_reentrant=False,**kw)
    kw['step_lane_roles']=torch.zeros_like(roles)
    recomputed=torch.autograd.grad(actual.square().sum(),(x,*decoder.parameters()))
    torch.testing.assert_close(actual,direct,rtol=0,atol=0)
    for a,b in zip(grads,recomputed):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
