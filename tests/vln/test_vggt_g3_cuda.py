"""Actual pinned FLA/GDN qualification; never substitute a slow production kernel."""
import functools
import torch
import pytest
from torch.utils.checkpoint import checkpoint
from qwen_vl.research.vggt_g3 import insert_probes
from qwen_vl.models.vggt_g3_heads import NativeStateHead
from qwen_vl.models.vggt_g3_forward import native_state_gdn
from test_vggt_g3 import recurrence
pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='H100 FLA qualification')


def relative(actual,expected):
    return float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12))


def test_real_fla_32x128_identity_state_reads_and_gradients():
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as kernel
    torch.manual_seed(429)
    q,k,v=[torch.randn(1,137,32,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for _ in range(3)]
    g=(-torch.rand(1,137,32,device='cuda')*.2).requires_grad_()
    beta=torch.rand(1,137,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    probes=torch.randn(10,32,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    base,state=kernel(q,k,v,g=g,beta=beta,output_final_state=True,use_qk_l2norm_in_kernel=True)
    packed,real,read=insert_probes((q,k,v,g,beta),torch.tensor([31]*5+[99]*5,device='cuda'),probes)
    pq,pk,pv,pg,pb=packed
    actual,final=kernel(pq,pk,pv,g=pg,beta=pb,output_final_state=True,use_qk_l2norm_in_kernel=True)
    assert relative(actual[:,real],base)<.02
    assert relative(final,state)<.02
    ref,refstate=recurrence(*[t.detach().double() for t in packed])
    assert relative(actual[:,read],ref[:,read])<.02
    a=torch.autograd.grad(base.float().square().sum(),(q,k,v,g,beta),retain_graph=True)
    b=torch.autograd.grad(actual[:,real].float().square().sum(),(q,k,v,g,beta),retain_graph=True)
    errors=[relative(x,y) for x,y in zip(b,a)]
    assert max(errors)<.04,errors
    aux=torch.autograd.grad(actual[:,read[:5]].float().square().sum(),(q,k,v,g,beta),retain_graph=True)
    assert aux[2][:,:32].abs().sum()>0
    for grad in aux:
        assert float(grad[:,32:].abs().max())<1e-7
    print('FLA_G3',{'output':relative(actual[:,real],base),'state':relative(final,state),
                    'probe_reference':relative(actual[:,read],ref[:,read]),'gradients':errors})


@pytest.mark.parametrize('uniform_empty',[False,True])
def test_native_prep_checkpoint_and_probe_free_parity(uniform_empty):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg=Qwen3_5TextConfig(hidden_size=2560,linear_num_value_heads=32,linear_num_key_heads=16,
                         linear_key_head_dim=128,linear_value_head_dim=128,dtype=torch.bfloat16)
    gdn=Qwen3_5GatedDeltaNet(cfg,24).cuda().bfloat16()
    with torch.no_grad():gdn.A_log.zero_();gdn.dt_bias.zero_()
    head=NativeStateHead().cuda().bfloat16()
    torch.manual_seed(5)
    x=torch.randn(1,137,2560,device='cuda',dtype=torch.bfloat16)
    anchors=torch.tensor([31,99],device='cuda');lags=torch.tensor([8,32],device='cuda')
    targets=torch.randn(2,5,2048,device='cuda')
    native=gdn(x)
    empty,zero=native_state_gdn(gdn,x,head,anchors[:0],lags[:0],targets[:0],uniform_empty=uniform_empty)
    assert torch.equal(native,empty) and zero==0
    results=[]
    for enabled in (False,True):
        leaf=x.clone().requires_grad_();gdn.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        fn=lambda h:native_state_gdn(gdn,h,head,anchors,lags,targets,uniform_empty=uniform_empty)
        out,loss=checkpoint(fn,leaf,use_reentrant=False) if enabled else fn(leaf)
        assert relative(out,native)<.02
        loss.backward()
        assert leaf.grad[:,:100].abs().sum()>0 and leaf.grad[:,100:].abs().max()<1e-7
        assert gdn.in_proj_qkv.weight.grad.abs().sum()>0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        results.append((out.detach(),loss.detach(),leaf.grad.clone(),head.region_embedding.grad.clone()))
    for a,b in zip(*results):torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_actual_online_g3_targets_are_unpositioned_region_means():
    from pathlib import Path
    import json
    from qwen_vl.research.vggt_g3_teacher import OnlineG3Teacher,g3_teacher_identity
    from qwen_vl.research.vggt_g3 import region_targets
    from qwen_vl.research.vggt_omega import pool_teacher_tokens
    from qwen_vl.data.vggt_online import preprocess_teacher_images
    root=Path('/mnt/data/vmo-ai-task/anhdh35')
    weights=root/'models/vggt-omega/vggt_omega_1b_512.pt'
    if not weights.is_file():pytest.skip('Pinned Omega assets unavailable')
    source=root/'third_party/vggt-omega'
    spec=dict(teacher_source=str(source),teacher_weights=str(weights),teacher_residency='resident',
              teacher_batch_size=8,teacher_identity=g3_teacher_identity(source,weights))
    teacher=OnlineG3Teacher(spec,'cuda')
    episode=json.loads((root/'g1-omega-20261013/fixtures/r2rce_short.jsonl').read_text().splitlines()[0])
    images=preprocess_teacher_images([s['rgb_path'] for s in episode['steps'][:2]])
    target=teacher(images)
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        out,start=teacher.model.aggregator(images.cuda()[:,None])
        expected=region_targets(pool_teacher_tokens(out[23],start,(28,37)))
    torch.testing.assert_close(target,expected,atol=0,rtol=0)
    assert target.shape==(2,5,2048) and not target.is_inference()
    assert not target.requires_grad and all(not p.requires_grad for p in teacher.model.parameters())
    assert teacher(torch.empty(0,3,448,592)).shape==(0,5,2048)
