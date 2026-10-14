import copy
from types import SimpleNamespace
import pytest
import torch
from test_episodic_forward import tiny
from qwen_vl.stream.cache import make_stream_cache_fp32


def install(text):
    from qwen_vl.models.geoaddress_runtime import install_geoaddress
    return install_geoaddress(text, layer_index=0, hidden_size=48, head_dim=8)


def test_zero_adapter_native_cached_modes_and_idempotent(tiny):
    gdn=tiny.layers[0].linear_attn
    original=copy.deepcopy(gdn)
    install(tiny);weights={n:p.clone() for n,p in gdn.geoaddress.named_parameters()}
    install(tiny)
    assert all(torch.equal(p,weights[n]) for n,p in gdn.geoaddress.named_parameters())
    a,b=[make_stream_cache_fp32(tiny.config) for _ in range(2)]
    with torch.no_grad():
        for length in (7,5,1,3):
            x=torch.randn(1,length,48);mask=torch.arange(length)[None]%2==0
            expected=original(x,cache_params=a)
            actual=gdn(x,cache_params=b,geoaddress_image_mask=mask)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            torch.testing.assert_close(a.layers[0].recurrent_states,b.layers[0].recurrent_states,atol=0,rtol=0)
            torch.testing.assert_close(a.layers[0].conv_states,b.layers[0].conv_states,atol=0,rtol=0)
    with pytest.raises(ValueError):gdn(x)


def test_active_adapter_continuation_decode_and_source_causality(tiny):
    install(tiny);gdn=tiny.layers[0].linear_attn
    with torch.no_grad():
        gdn.geoaddress.q_up.weight.normal_(0,.05);gdn.geoaddress.k_up.weight.normal_(0,.05)
    x=torch.randn(1,17,48);mask=torch.zeros(1,17,dtype=torch.bool);mask[:,2:9]=True
    with torch.no_grad():
        full=gdn(x,geoaddress_image_mask=mask)
        cache=make_stream_cache_fp32(tiny.config);out=[]
        for start,end in [(0,7),(7,12),(12,13),(13,17)]:
            out.append(gdn(x[:,start:end],cache_params=cache,geoaddress_image_mask=mask[:,start:end]))
        torch.testing.assert_close(torch.cat(out,1),full,atol=2e-6,rtol=2e-5)
        changed=x.clone();changed[:,10:]+=5
        actual=gdn(changed,geoaddress_image_mask=mask)
        torch.testing.assert_close(actual[:,:10],full[:,:10],atol=0,rtol=0)
    assert 'geoaddress.q_down.weight' in gdn.state_dict()


def test_functional_relation_checkpoint_gradients_and_no_future_dependency(tiny):
    from qwen_vl.models.geoaddress_loss import address_relation_loss,teacher_relations
    from qwen_vl.models.geoaddress_runtime import geoaddress_gdn
    from torch.utils.checkpoint import checkpoint
    install(tiny);gdn=tiny.layers[0].linear_attn
    x=torch.randn(1,340,48);mask=torch.ones(1,340,dtype=torch.bool)
    qpos=torch.arange(300,332)[None];kpos=torch.arange(300)[None]
    probs=teacher_relations(torch.randn(1,32,16),torch.randn(1,300,16))
    relation=lambda q,k:address_relation_loss(q[0,qpos,:2],k[0,kpos,:2],probs)
    outputs=[]
    for enabled in (False,True):
        gdn.zero_grad();leaf=x.clone().requires_grad_()
        fn=lambda h:geoaddress_gdn(gdn,h,mask,relation=relation)
        out,loss=checkpoint(fn,leaf,use_reentrant=False) if enabled else fn(leaf)
        loss.backward()
        assert leaf.grad[:,:300].abs().sum()>0 and leaf.grad[:,332:].count_nonzero()==0
        assert gdn.geoaddress.k_up.weight.grad.abs().sum()>0
        outputs.append((out.detach(),loss.detach(),leaf.grad.clone()))
    for a,b in zip(*outputs):torch.testing.assert_close(a,b,atol=0,rtol=0)
