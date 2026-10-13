"""Pure G3 sampling/target/row-layout contracts, no teacher or GPU dependency."""
import torch
from qwen_vl.research.vggt_sampling import sample_frames, stable_hash


def sample_pairs(trajectory_key, length, seed=429):
    candidates=[(s,t) for s in sample_frames(trajectory_key,length,seed)
                for t in range(s+8,min(s+33,length))]
    ranked=sorted(candidates,key=lambda pair:stable_hash([trajectory_key,*pair,seed]))
    selected=[]
    for low,high,quota in ((8,15,3),(16,23,3),(24,32,2)):
        selected.extend([p for p in ranked if low<=p[1]-p[0]<=high][:quota])
    selected.extend([p for p in ranked if p not in selected][:8-len(selected)])
    return tuple(sorted(selected,key=lambda p:(p[1],p[0])))


def observation_anchors(target_positions, action_indices, num_actions):
    if target_positions.ndim!=1 or target_positions.shape!=action_indices.shape:
        raise ValueError('Response target/index vectors disagree')
    anchors=[]
    for t in range(num_actions):
        positions=target_positions[action_indices==t]
        if positions.numel()==0 or int(positions.min())<1:
            raise ValueError('Every action needs a preceding observation anchor')
        anchors.append(positions.min()-1)
    return torch.stack(anchors) if anchors else target_positions.new_empty(0)


def region_targets(tokens):
    if tokens.ndim!=3 or tokens.shape[1]!=300:raise ValueError('Expected [F,300,D]')
    grid=tokens.float().reshape(-1,15,20,tokens.shape[-1])
    return torch.stack([grid[:,:7,:10].mean((1,2)),grid[:,:7,10:].mean((1,2)),
        grid[:,7:,:10].mean((1,2)),grid[:,7:,10:].mean((1,2)),grid.mean((1,2))],1)


def insert_probes(prepared, anchors, probe_queries):
    """Insert post-write reads AFTER real anchors, never into convolution inputs.

    q/k/v are [1,L,H,D], g/beta [1,L,H]; queries [P,H,Dk].
    Returned row maps are new tensors; inputs are not mutated. Maps are stable
    for repeated anchors and all inserted gates are literal effective zeros.
    """
    q,k,v,g,beta=prepared
    if q.shape[0]!=1 or anchors.ndim!=1 or len(anchors)!=len(probe_queries):
        raise ValueError('Probe layout requires batch1 and matching anchors/queries')
    length=q.shape[1];n=len(anchors);device=q.device
    anchors=anchors.to(device=device,dtype=torch.long)
    if n and (bool((anchors<0).any()) or bool((anchors>=length).any()) or bool((anchors[1:]<anchors[:-1]).any())):
        raise ValueError('Anchors must be sorted valid real sequence positions')
    if not n:return prepared,torch.arange(length,device=device),anchors
    probe_rows=anchors+1+torch.arange(n,device=device)
    real_rows=torch.arange(length,device=device)
    real_rows=real_rows+torch.searchsorted(anchors,real_rows,right=False)
    expanded=[]
    for index,tensor in enumerate(prepared):
        out=tensor.new_zeros((1,length+n,*tensor.shape[2:]))
        out=out.index_copy(1,real_rows,tensor)
        if index==0: out=out.index_copy(1,probe_rows,probe_queries[None].to(tensor.dtype))
        elif index==1:
            unit=tensor.new_zeros((1,n,*tensor.shape[2:]));unit[...,0]=1
            out=out.index_copy(1,probe_rows,unit)
        expanded.append(out)
    return tuple(expanded),real_rows,probe_rows
