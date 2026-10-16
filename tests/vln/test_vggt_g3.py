import pytest
import torch
from qwen_vl.research.vggt_g3 import sample_pairs, observation_anchors, region_targets, insert_probes
from qwen_vl.models.vggt_g3_heads import NativeStateHead


def recurrence(q,k,v,g,beta):
    # Independent high precision key x value reference, including post-write read.
    state=torch.zeros(q.shape[0],q.shape[2],q.shape[3],v.shape[3],dtype=q.dtype,device=q.device)
    out=[]
    for t in range(q.shape[1]):
        qt,kt=q[:,t],k[:,t]
        qt=qt*torch.rsqrt(qt.square().sum(-1,keepdim=True)+1e-6)
        kt=kt*torch.rsqrt(kt.square().sum(-1,keepdim=True)+1e-6)
        state=state*g[:,t,:,None,None].exp()
        residual=v[:,t]-torch.einsum('bhkv,bhk->bhv',state,kt)
        state=state+kt[...,None]*(beta[:,t,:,None]*residual)[...,None,:]
        out.append(torch.einsum('bhkv,bhk->bhv',state,qt)/q.shape[-1]**.5)
    return torch.stack(out,1),state


def test_sampling_bounds_determinism_and_anchor_is_before_first_target():
    for T in (1,7,8,9,10,33,627):
        pairs=sample_pairs('fixture',T,429)
        assert pairs==sample_pairs('fixture',T,429)
        assert len(pairs)<=8 and len(set(pairs))==len(pairs)
        assert all(0<=s<t<T and 8<=t-s<=32 for s,t in pairs)
        if T<=8: assert not pairs
        if T==627: assert len(pairs)==8
    assert observation_anchors(torch.tensor([9,10,22,23]),torch.tensor([0,0,1,1]),2).tolist()==[8,21]
    with pytest.raises(ValueError): observation_anchors(torch.tensor([0]),torch.tensor([0]),1)


def test_regions_unpositioned_and_area_weighted():
    x=torch.arange(300,dtype=torch.float32).view(1,300,1).expand(2,300,4)
    r=region_targets(x)
    assert r.shape==(2,5,4)
    assert torch.equal(r[:,4],x.mean(1))
    assert not torch.equal(r[:,4],r[:,:4].mean(1))


def test_probes_are_identity_updates_and_causal_with_repeated_anchor():
    torch.manual_seed(7)
    q,k=[torch.randn(1,11,2,3,dtype=torch.float64,requires_grad=True) for _ in range(2)]
    v=torch.randn(1,11,2,5,dtype=torch.float64,requires_grad=True)
    g=(-torch.rand(1,11,2,dtype=torch.float64)).requires_grad_()
    beta=torch.rand(1,11,2,dtype=torch.float64,requires_grad=True)
    probe=torch.randn(3,2,3,dtype=torch.float64,requires_grad=True)
    base,state=recurrence(q,k,v,g,beta)
    expanded,real,reads=insert_probes((q,k,v,g,beta),torch.tensor([2,2,7]),probe)
    eq,ek,ev,eg,eb=expanded
    assert torch.count_nonzero(eg[:,reads])==torch.count_nonzero(eb[:,reads])==0
    actual,final=recurrence(*expanded)
    torch.testing.assert_close(actual[:,real],base,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(final,state,rtol=1e-13,atol=1e-13)
    for j,a in enumerate([2,2,7]):
        _,at=recurrence(q[:,:a+1],k[:,:a+1],v[:,:a+1],g[:,:a+1],beta[:,:a+1])
        pq=probe[j]*torch.rsqrt(probe[j].square().sum(-1,keepdim=True)+1e-6)
        expected=torch.einsum('bhkv,hk->bhv',at,pq)/3**.5
        torch.testing.assert_close(actual[:,reads[j]],expected,rtol=1e-13,atol=1e-13)
    grad1=torch.autograd.grad(base.square().sum(),(q,k,v,g,beta),retain_graph=True)
    grad2=torch.autograd.grad(actual[:,real].square().sum(),(q,k,v,g,beta),retain_graph=True)
    for a,b in zip(grad1,grad2):torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    auxgrads=torch.autograd.grad(actual[:,reads[0]].square().sum(),(q,k,v,g,beta),allow_unused=True)
    for grad in auxgrads:
        if grad is not None:assert torch.count_nonzero(grad[:,3:])==0
    assert auxgrads[2][:,:3].abs().sum()>0
    empty,real,reads=insert_probes((q,k,v,g,beta),torch.empty(0,dtype=torch.long),probe[:0])
    assert not len(reads) and empty[0] is q


def test_head_context_detached_target_detached_empty_and_rng():
    torch.manual_seed(31);rng=torch.random.get_rng_state().clone()
    head=NativeStateHead(heads=2,key_dim=3,value_dim=5,query_hidden=8,decoder_hidden=12,target_dim=4)
    assert torch.equal(rng,torch.random.get_rng_state())
    context=torch.randn(2,2,3,requires_grad=True)
    queries=head.queries(context,torch.tensor([8,32]))
    assert queries.shape==(10,2,3)
    queries.square().sum().backward()
    assert context.grad is None
    assert head.region_embedding.grad.abs().sum()>0
    head.zero_grad(set_to_none=True)
    reads=torch.randn(2,5,2,5,requires_grad=True);targets=torch.randn(2,5,4,requires_grad=True)
    head.loss(reads,targets).backward()
    assert reads.grad.abs().sum()>0 and targets.grad is None
    head.zero_grad(set_to_none=True)
    zero=head.empty_loss(context)
    zero.backward()
    assert all(p.grad is not None for p in head.parameters()) and zero==0


def test_pair_quotas_and_shared_source_sample():
    from qwen_vl.research.vggt_sampling import sample_frames
    pairs=sample_pairs('long',627)
    assert all(s in sample_frames('long',627) for s,t in pairs)
    assert [sum(lo<=t-s<=hi for s,t in pairs) for lo,hi in ((8,15),(16,23),(24,32))]==[3,3,2]


def test_region_shape_empty_and_no_added_position():
    target=region_targets(torch.ones(3,300,2048))
    assert torch.equal(target,torch.ones(3,5,2048))
    assert region_targets(torch.empty(0,300,2048)).shape==(0,5,2048)


def test_invalid_probe_anchor_rejected():
    q=torch.zeros(1,3,1,2);g=torch.zeros(1,3,1)
    for anchors in ([-1],[3],[2,1]):
        with pytest.raises(ValueError):insert_probes((q,q,q,g,g),torch.tensor(anchors),torch.zeros(len(anchors),1,2))


@pytest.mark.parametrize('run',['G3','G3_M5'])
def test_dataset_maps_online_source_frames_to_pairs_without_target_leakage(monkeypatch,run):
    from types import SimpleNamespace
    from qwen_vl.data.episode_dataset import EpisodeDataset
    from qwen_vl.research.vggt_sampling import collection_trajectory_key
    episode={'dataset':'r2r','scene_id':'scene','episode_uid':'id','instruction':'go',
             'steps':[{'rgb_path':f'/trajectory_data/a/{i}.png'} for i in range(40)]}
    positions=torch.arange(40)*4+3
    encoded={'input_ids':torch.zeros(1,160,dtype=torch.long),'response_target_positions':positions,
             'response_action_index':torch.arange(40),'num_actions':40}
    dataset=object.__new__(EpisodeDataset);dataset.episodes=[episode];dataset.encoded_lengths=[160]
    dataset.serializer=SimpleNamespace(config={'runtime':{},'research':{'run_id':run}},
        encode_episode=lambda ep:dict(encoded))
    captured=[]
    def images(paths):
        captured.extend(paths);return torch.zeros(len(paths),3,448,592)
    monkeypatch.setattr('qwen_vl.data.vggt_online.preprocess_teacher_images',images)
    result=dataset[0];pairs=sample_pairs(collection_trajectory_key(episode),40)
    assert result['aux_pair_count']==len(pairs)
    assert result['aux_anchors'].tolist()==[int(positions[t]-1) for s,t in pairs]
    assert result['aux_lags'].tolist()==[t-s for s,t in pairs]
    assert [captured[i] for i in result['aux_source_indices']]==[episode['steps'][s]['rgb_path'] for s,t in pairs]
    assert 'aux_visual_positions' not in result and len(captured)==len(set(s for s,t in pairs))



def test_declared_parameter_count_and_context_identity_counterexample():
    head=NativeStateHead()
    assert sum(p.numel() for p in head.parameters())==10526368
    small=NativeStateHead(heads=2,key_dim=3,value_dim=3,query_hidden=8,decoder_hidden=12,target_dim=4)
    contexts=torch.tensor([[[1.,0,0],[0,1.,0]],[[0,0,1.],[1.,0,0]]])
    queries=small.queries(contexts,torch.tensor([8,8])).reshape(2,5,2,3)
    # Same identity state produces different reads solely from the query context.
    reads=queries*torch.rsqrt(queries.square().sum(-1,keepdim=True)+1e-6)/3**.5
    assert (reads[0]-reads[1]).abs().max()>1e-3
