import pytest
import torch


def make_adapter(**kw):
    from qwen_vl.models.geoaddress import GeoAddress
    return GeoAddress(**kw)


def test_zero_identity_parameter_count_rng_and_seed():
    torch.manual_seed(31);rng=torch.random.get_rng_state().clone()
    a=make_adapter();b=make_adapter()
    assert torch.equal(rng,torch.random.get_rng_state())
    assert sum(p.numel() for p in a.parameters())==45056
    assert all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))
    x=torch.randn(1,4,2560);q=torch.randn(1,4,16,128);k=torch.randn_like(q)
    out=a(x,q,k,torch.tensor([[True,False,True,False]]))
    assert torch.equal(out[0],q) and torch.equal(out[1],k)


def test_active_adapter_only_image_rows_and_selected_raw_heads():
    a=make_adapter(hidden_size=6,head_dim=3,rank=2)
    with torch.no_grad():
        a.q_up.weight.fill_(.1);a.k_up.weight.fill_(.2)
    x=torch.randn(1,4,6);q=torch.randn(1,4,4,3);k=torch.randn_like(q)
    mask=torch.tensor([[True,False,True,False]])
    qo,ko=a(x,q,k,mask)
    for out,old,down,up in [(qo,q,a.q_down,a.q_up),(ko,k,a.k_down,a.k_up)]:
        assert torch.equal(out[:,~mask[0]],old[:,~mask[0]])
        assert torch.equal(out[:,:,2:],old[:,:,2:])
        expected=old[:,:,:2]+up(down(x)).view(1,4,2,3)*mask[:,:,None,None]
        torch.testing.assert_close(out[:,:,:2],expected)
        assert not torch.equal(out,old)
    assert q.data_ptr()!=qo.data_ptr()


def test_gradient_opening_and_single_token():
    a=make_adapter(hidden_size=6,head_dim=3,rank=2)
    opt=torch.optim.SGD(a.parameters(),lr=.1)
    x=torch.randn(1,1,6,requires_grad=True);q=torch.randn(1,1,4,3,requires_grad=True);k=torch.randn_like(q,requires_grad=True)
    for step in range(2):
        opt.zero_grad();qo,ko=a(x,q,k,torch.ones(1,1,dtype=torch.bool));(qo.square().sum()+ko.square().sum()).backward()
        assert a.q_up.weight.grad.abs().sum()>0 and a.k_up.weight.grad.abs().sum()>0
        if step==0:assert a.q_down.weight.grad.count_nonzero()==a.k_down.weight.grad.count_nonzero()==0
        else:assert a.q_down.weight.grad.abs().sum()>0 and a.k_down.weight.grad.abs().sum()>0
        opt.step()
    assert x.grad.abs().sum()>0 and q.grad.abs().sum()>0 and k.grad.abs().sum()>0


def test_bad_mask_shapes_and_nonfinite_rejected():
    a=make_adapter(hidden_size=6,head_dim=3,rank=2)
    x=torch.randn(1,4,6);q=torch.randn(1,4,4,3)
    for mask in [None,torch.ones(4,dtype=torch.bool),torch.ones(1,4)]:
        with pytest.raises(ValueError):a(x,q,q,mask)
    x[0,0,0]=float('nan')
    with pytest.raises(FloatingPointError):a(x,q,q,torch.ones(1,4,dtype=torch.bool))
