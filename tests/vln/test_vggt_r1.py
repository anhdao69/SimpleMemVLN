import pytest
import torch
from torch.utils.checkpoint import checkpoint
from test_step_lane_module import lane,sample
from qwen_vl.models.vggt_g3_heads import NativeStateHead


def run(m,x,roles,head,context,anchors,targets):
    from qwen_vl.models.vggt_r1_forward import lane_forward_with_queries
    return lane_forward_with_queries(m,x,roles,head,context,anchors,torch.full_like(anchors,8),targets)


def test_lane_probes_preserve_original_outputs_and_navigation_gradients():
    m=lane();torch.nn.init.normal_(m.out_proj.weight)
    x,roles=sample();x.requires_grad_()
    head=NativeStateHead(heads=2,key_dim=3,value_dim=5,context_heads=3,query_hidden=8,decoder_hidden=12,target_dim=4)
    anchors=torch.tensor([2,5,5,7]);context=torch.randn(4,3,3);targets=torch.randn(4,5,4)
    expected,_=m(x,roles)
    actual,loss=run(m,x,roles,head,context,anchors,targets)
    torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-7)
    a=torch.autograd.grad(expected.square().sum(),(x,*m.parameters()),retain_graph=True)
    b=torch.autograd.grad(actual.square().sum(),(x,*m.parameters()),retain_graph=True)
    for i,j in zip(a,b):torch.testing.assert_close(i,j,rtol=1e-5,atol=1e-6)
    assert torch.isfinite(loss)


def test_auxiliary_reaches_previous_writers_with_zero_output_and_no_future_or_query_gradient():
    m=lane();x,roles=sample();x.requires_grad_()
    head=NativeStateHead(heads=2,key_dim=3,value_dim=5,context_heads=3,query_hidden=8,decoder_hidden=12,target_dim=4)
    context=torch.randn(1,3,3,requires_grad=True);targets=torch.randn(1,5,4,requires_grad=True)
    out,loss=run(m,x,roles,head,context,torch.tensor([5]),targets)
    assert torch.count_nonzero(out)==0
    loss.backward()
    assert m.in_proj_qkv.weight.grad.abs().sum()>0
    assert m.in_proj_b.bias.grad.abs().sum()>0
    assert x.grad[:,3].abs().sum()>0
    assert torch.count_nonzero(x.grad[:,4:])==0
    assert context.grad is None and targets.grad is None


def test_empty_pairs_and_checkpoint_parity():
    m=lane();x,roles=sample()
    head=NativeStateHead(heads=2,key_dim=3,value_dim=5,context_heads=3,query_hidden=8,decoder_hidden=12,target_dim=4)
    empty=torch.empty(0,dtype=torch.long)
    out,zero=run(m,x,roles,head,torch.empty(0,3,3),empty,torch.empty(0,5,4))
    assert torch.equal(out,m(x,roles)[0]);zero.backward()
    assert all(p.grad is not None for p in head.parameters())
    anchors=torch.tensor([5]);context=torch.randn(1,3,3);targets=torch.randn(1,5,4)
    results=[]
    for enabled in (False,True):
        m.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True);leaf=x.clone().requires_grad_()
        fn=lambda h:run(m,h,roles,head,context,anchors,targets)
        output,loss=checkpoint(fn,leaf,use_reentrant=False) if enabled else fn(leaf)
        loss.backward();results.append((output.detach(),loss.detach(),leaf.grad.clone()))
    for a,b in zip(*results):torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_step_end_anchor_is_rejected():
    m=lane();x,roles=sample()
    head=NativeStateHead(heads=2,key_dim=3,value_dim=5,context_heads=3,query_hidden=8,decoder_hidden=12,target_dim=4)
    with pytest.raises(ValueError,match='pre-action'):
        run(m,x,roles,head,torch.randn(1,3,3),torch.tensor([3]),torch.randn(1,5,4))
