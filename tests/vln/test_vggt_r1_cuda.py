import torch
import pytest
from torch.utils.checkpoint import checkpoint
from qwen_vl.models.vggt_g3_heads import NativeStateHead
from qwen_vl.models.vggt_r1_forward import native_output_and_context,lane_forward_with_queries
from qwen_vl.models.parallel_step_lane import ParallelStepLane
from qwen_vl.research.lane_contract import StepLaneSpec
from test_vggt_g3_cuda import relative

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='Actual H100/FLA R1 qualification')


def test_r1_real_native_once_lane_probes_and_checkpoint_gradients():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg=Qwen3_5TextConfig(hidden_size=2560,linear_num_value_heads=32,linear_num_key_heads=16,
                        linear_key_head_dim=128,linear_value_head_dim=128,dtype=torch.bfloat16)
    gdn=Qwen3_5GatedDeltaNet(cfg,24).cuda().bfloat16()
    with torch.no_grad():gdn.A_log.zero_();gdn.dt_bias.zero_()
    lane=ParallelStepLane(2560,StepLaneSpec(),device='cuda',dtype=torch.bfloat16,lane_seed=453)
    head=NativeStateHead(heads=4,context_heads=32).cuda().bfloat16()
    x=torch.randn(1,137,2560,device='cuda',dtype=torch.bfloat16)
    roles=torch.ones((1,137),device='cuda',dtype=torch.uint8);roles[:,[31,63,95,127]]=3
    anchors=torch.tensor([60,100],device='cuda');lags=torch.tensor([8,32],device='cuda')
    targets=torch.randn(2,5,2048,device='cuda')
    expected=gdn(x);calls=[]
    hook=gdn.in_proj_qkv.register_forward_hook(lambda *_:calls.append(1))
    native,context=native_output_and_context(gdn,x,anchors)
    hook.remove()
    assert len(calls)==1 and torch.equal(native,expected) and not context.requires_grad
    results=[]
    for enabled in (False,True):
        lane.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        leaf=x.clone().requires_grad_()
        fn=lambda h:lane_forward_with_queries(lane,h,roles,head,context,anchors,lags,targets)
        out,loss=checkpoint(fn,leaf,use_reentrant=False) if enabled else fn(leaf)
        assert out.count_nonzero()==0
        loss.backward()
        assert lane.in_proj_qkv.weight.grad.abs().sum()>0
        assert lane.in_proj_b.bias.grad.abs().sum()>0
        assert leaf.grad[:,:96].abs().sum()>0 and leaf.grad[:,96:].abs().max()<1e-7
        results.append((loss.detach(),leaf.grad.clone(),head.region_embedding.grad.clone()))
    for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=0,atol=0)
    with torch.no_grad():lane.out_proj.weight.normal_(0,.02)
    base=lane(x,roles)[0]
    out,_=lane_forward_with_queries(lane,x,roles,head,context,anchors,lags,targets)
    assert relative(out,base)<.02
    print('R1_FLA_OUTPUT_RELATIVE',relative(out,base))


def test_r1_online_omega_targets_match_g3_region_definition_exactly():
    import json
    from pathlib import Path
    from qwen_vl.research.vggt_r1_teacher import OnlineR1Teacher,r1_teacher_identity
    from qwen_vl.research.vggt_g3 import region_targets
    from qwen_vl.research.vggt_omega import pool_teacher_tokens
    from qwen_vl.data.vggt_online import preprocess_teacher_images
    root=Path('/mnt/data/vmo-ai-task/anhdh35');weights=root/'models/vggt-omega/vggt_omega_1b_512.pt'
    if not weights.exists():pytest.skip('Pinned teacher assets missing')
    source=root/'third_party/vggt-omega'
    spec=dict(teacher_source=str(source),teacher_weights=str(weights),teacher_batch_size=8,
              teacher_residency='resident',teacher_identity=r1_teacher_identity(source,weights))
    teacher=OnlineR1Teacher(spec,'cuda')
    ep=json.loads((root/'g1-omega-20261013/fixtures/r2rce_short.jsonl').read_text().splitlines()[0])
    images=preprocess_teacher_images([s['rgb_path'] for s in ep['steps'][:2]])
    actual=teacher(images)
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        out,start=teacher.model.aggregator(images.cuda()[:,None])
        expected=region_targets(pool_teacher_tokens(out[23],start,(28,37)))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert actual.shape==(2,5,2048) and not actual.requires_grad and not actual.is_inference()
    assert all(not p.requires_grad for p in teacher.model.parameters())
