import torch
import pytest


def test_rectangle_sampling_and_empty_delayed_plan():
    from qwen_vl.research.geoaddress_sampling import query_cells
    from qwen_vl.research.vggt_g3 import sample_pairs
    ids=query_cells('fixture',2,19)
    assert ids.shape==(32,) and ids.unique().numel()==32
    assert torch.equal(ids,query_cells('fixture',2,19))
    for i,idx in enumerate(ids.tolist()):
        y,x=divmod(idx,20);r,c=divmod(i,8)
        assert r*15//4<=y<(r+1)*15//4 and c*20//8<=x<(c+1)*20//8
    assert sample_pairs('fixture',8)==()


def test_teacher_probabilities_detached_and_spatial_permutation():
    from qwen_vl.models.geoaddress_loss import teacher_relations
    q=torch.randn(1,32,64,requires_grad=True);k=torch.randn(1,300,64,requires_grad=True)
    p=teacher_relations(q,k)
    assert p.shape==(1,32,300) and p.dtype==torch.float32 and not p.requires_grad
    torch.testing.assert_close(p.sum(-1),torch.ones(1,32))
    perm=torch.randperm(300)
    torch.testing.assert_close(teacher_relations(q,k[:,perm]),p[:,:,perm])
    with pytest.raises(ValueError):teacher_relations(q,k[:,:299])
    with pytest.raises(FloatingPointError):teacher_relations(q*0,k)


def test_direct_relation_gradients_permutations_and_empty():
    from qwen_vl.models.geoaddress_loss import address_relation_loss,teacher_relations
    q=torch.randn(1,32,2,5,requires_grad=True);k=torch.randn(1,300,2,5,requires_grad=True)
    p=teacher_relations(torch.randn(1,32,16),torch.randn(1,300,16)).requires_grad_()
    loss=address_relation_loss(q,k,p);loss.backward()
    assert torch.isfinite(loss) and loss>0 and p.grad is None
    assert q.grad.abs().sum()>0 and k.grad.abs().sum()>0
    perm=torch.randperm(300)
    torch.testing.assert_close(address_relation_loss(q,k[:,perm],p[:,:,perm]),loss)
    assert not torch.isclose(address_relation_loss(q,k[:,perm],p),loss,rtol=1e-5,atol=1e-5)
    q.grad=None;k.grad=None;empty=address_relation_loss(q[:0],k[:0],p[:0]);empty.backward()
    assert empty==0 and q.grad is not None and k.grad is not None


def test_matching_distributions_zero_kl_and_additive_native_normalization():
    from qwen_vl.models.geoaddress_loss import relation_kl_sum,native_normalize
    q=torch.randn(2,32,1,5).expand(-1,-1,2,-1);k=torch.randn(2,300,1,5).expand(-1,-1,2,-1)
    qn,kn=native_normalize(q),native_normalize(k)
    logits=torch.einsum('pihd,pjhd->phij',qn,kn)/.1
    p=logits[:,0].softmax(-1).detach()
    assert abs(float(relation_kl_sum(qn,kn,p)))<1e-6
    x=torch.full((1,5),1e-5)
    torch.testing.assert_close(native_normalize(x),x*torch.rsqrt(x.square().sum(-1,keepdim=True)+1e-6))


def test_metadata_real_visual_rows_and_no_teacher_for_a0():
    from qwen_vl.research.geoaddress_sampling import address_metadata
    ep=dict(dataset='r2r',scene_id='s',steps=[dict(rgb_path=f'/trajectory_data/p/{i}.png') for i in range(42)])
    spans=[(i*310,(i+1)*310) for i in range(42)]
    ids=torch.zeros(1,42*310,dtype=torch.long)
    for start,end in spans:ids[:,start+3:start+303]=99
    result={'input_ids':ids,'step_plan':(0,spans)}
    meta,frames=address_metadata(ep,result,99,'A1',429)
    q,k=meta['aux_query_positions'],meta['aux_key_positions']
    assert q.shape==(8,32) and k.shape==(8,300) and len(frames)<=16
    assert (ids[0,q]==99).all() and (ids[0,k]==99).all()
    assert (k.max(-1).values<q.min(-1).values).all()
    for p,(si,ti) in enumerate(zip(meta['aux_teacher_sources'],meta['aux_teacher_reads'])):
        assert k[p,0]==frames[si]*310+3
        assert torch.equal(q[p],frames[ti]*310+3+meta['aux_query_cells'][p])
    control,frames=address_metadata(ep,result,99,'A0',429)
    assert not frames and control['aux_pair_count']==0 and 'aux_teacher_sources' not in control
