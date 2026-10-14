"""Pinned H100 kernels: zero adapter native equality, active cached parity/gradients."""
import copy
import torch
import pytest
from types import SimpleNamespace
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 native GDN qualification')


def test_h100_geoaddress_zero_and_active_native_cache_modes():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from qwen_vl.models.geoaddress_runtime import install_geoaddress,geoaddress_gdn
    from qwen_vl.stream.cache import make_stream_cache_fp32
    from qwen_vl.models.geoaddress_loss import address_relation_loss,teacher_relations
    cfg=Qwen3_5TextConfig(hidden_size=2560,linear_num_value_heads=32,linear_num_key_heads=16,
                         linear_key_head_dim=128,linear_value_head_dim=128,dtype=torch.bfloat16)
    gdn=Qwen3_5GatedDeltaNet(cfg,0).cuda().bfloat16()
    with torch.no_grad():gdn.A_log.zero_();gdn.dt_bias.zero_()
    original=copy.deepcopy(gdn)
    text=SimpleNamespace(layers=[SimpleNamespace(linear_attn=gdn)])
    install_geoaddress(text,layer_index=0)
    a,b=[make_stream_cache_fp32(cfg) for _ in range(2)]
    with torch.no_grad():
        for length in (137,61,1):
            x=torch.randn(1,length,2560,device='cuda',dtype=torch.bfloat16)
            mask=torch.arange(length,device='cuda')[None]%3!=0
            expected=original(x,cache_params=a)
            actual=gdn(x,cache_params=b,geoaddress_image_mask=mask)
            assert torch.equal(actual,expected)
            torch.testing.assert_close(a.layers[0].recurrent_states,b.layers[0].recurrent_states,atol=0,rtol=0)
            torch.testing.assert_close(a.layers[0].conv_states,b.layers[0].conv_states,atol=0,rtol=0)
        gdn.geoaddress.q_up.weight.normal_(0,.01);gdn.geoaddress.k_up.weight.normal_(0,.01)
        x=torch.randn(1,340,2560,device='cuda',dtype=torch.bfloat16);mask=torch.ones(1,340,dtype=torch.bool,device='cuda')
        full=gdn(x,geoaddress_image_mask=mask)
        cache=make_stream_cache_fp32(cfg);parts=[]
        for start,end in [(0,137),(137,339),(339,340)]:
            parts.append(gdn(x[:,start:end],cache_params=cache,geoaddress_image_mask=mask[:,start:end]))
        streamed=torch.cat(parts,1)
        relative=float((full.float()-streamed.float()).norm()/full.float().norm())
        assert relative<.03,relative
    qpos=torch.arange(300,332,device='cuda')[None];kpos=torch.arange(300,device='cuda')[None]
    probs=teacher_relations(torch.randn(1,32,2048,device='cuda'),torch.randn(1,300,2048,device='cuda'))
    leaf=x.clone().requires_grad_()
    _,loss=geoaddress_gdn(gdn,leaf,mask,relation=lambda q,k:address_relation_loss(q[0,qpos,:2],k[0,kpos,:2],probs))
    loss.backward()
    assert leaf.grad[:,:300].abs().sum()>0 and leaf.grad[:,332:].abs().max()==0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in gdn.geoaddress.parameters())
    print('GEOADDRESS_CUDA_PASS',{'active_stream_relative':relative,'aux_loss':float(loss)})
