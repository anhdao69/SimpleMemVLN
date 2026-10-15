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


def test_empty_and_nonempty_support_have_identical_gradient_readiness_order():
    orders = []
    for count in (1, 0):
        m = lane();torch.nn.init.normal_(m.out_proj.weight)
        x, roles = sample();x.requires_grad_()
        head = NativeStateHead(heads=2,key_dim=3,value_dim=5,context_heads=3,
                               query_hidden=8,decoder_hidden=12,target_dim=4)
        order = []
        for prefix, module in (('lane',m),('head',head)):
            for name, p in module.named_parameters():
                p.register_hook(lambda grad, name=prefix+'.'+name: order.append(name))
        anchors = torch.tensor([5],dtype=torch.long)[:count]
        context = torch.randn(count,3,3);targets = torch.randn(count,5,4)
        def forward(h):return run(m,h,roles,head,context,anchors,targets)
        output,loss = checkpoint(forward,x,use_reentrant=False)
        (output.square().sum()+loss).backward()
        assert all(p.grad is not None for p in head.parameters())
        if not count:
            assert loss.item()==0
            assert all(torch.count_nonzero(p.grad)==0 for p in head.parameters())
        orders.append(order)
    assert orders[0]==orders[1]


def test_empty_support_preserves_lane_outputs_and_gradients_without_extra_rows():
    m=lane();torch.nn.init.normal_(m.out_proj.weight)
    x,roles=sample();x.requires_grad_()
    head=NativeStateHead(heads=2,key_dim=3,value_dim=5,context_heads=3,query_hidden=8,decoder_hidden=12,target_dim=4)
    sizes=[];original=m._chunk_kernel
    def kernel(**kwargs):
        sizes.append(kwargs['q'].shape[1])
        return original(**kwargs)
    m._chunk_kernel=kernel
    expected=m(x,roles)[0]
    out,loss=run(m,x,roles,head,torch.empty(0,3,3),torch.empty(0,dtype=torch.long),torch.empty(0,5,4))
    assert torch.equal(out,expected) and loss.item()==0
    assert sizes==[x.shape[1],x.shape[1]]
    params=(x,*m.parameters())
    expected_grads=torch.autograd.grad(expected.square().sum(),params,retain_graph=True)
    actual_grads=torch.autograd.grad(out.square().sum()+loss,params,retain_graph=True)
    for actual,reference in zip(actual_grads,expected_grads):
        torch.testing.assert_close(actual,reference,atol=0,rtol=0)
    loss.backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad)==0 for p in head.parameters())
