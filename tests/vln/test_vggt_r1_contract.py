import copy
import json
from pathlib import Path
import pytest
import torch
from qwen_vl.research.vggt_g1 import parse_g1
from qwen_vl.models.vggt_g3_heads import NativeStateHead


def config():
    record=json.loads(Path('reports/vggt_five/R1/evidence/baseline_registry.json').read_text())['R0']
    cfg=copy.deepcopy(record['metadata']['config'])
    cfg['research']=dict(run_id='R1',student_layer=24,head_lr=1e-4,lambda_max=.1,ramp_updates=771,
        teacher_source='source',teacher_weights='weights',teacher_identity={'test':1},baseline=record)
    return cfg


def test_exact_registered_r0_is_accepted_and_mismatches_rejected():
    cfg=config()
    assert parse_g1(cfg)['run_id']=='R1'
    edits=[('model.step_lane.layers',[24]),('model.step_lane.init_seed',430),
           ('model.step_lane.clock_mode','token_rate_calibrated'),('training.step_lane_lr',5e-5),
           ('observations.append_action_tokens',False),('model.output_mode','candidate_logits')]
    for path,value in edits:
        bad=copy.deepcopy(cfg);target=bad
        parts=path.split('.')
        for p in parts[:-1]:target=target[p]
        target[parts[-1]]=value
        with pytest.raises(ValueError):parse_g1(bad)
    bad=copy.deepcopy(cfg);bad['research'].pop('baseline')
    with pytest.raises(ValueError):parse_g1(bad)


def test_lane_head_native_context_shapes_rng_and_detachment():
    torch.manual_seed(56);rng=torch.random.get_rng_state().clone()
    head=NativeStateHead(heads=4,context_heads=32)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert sum(p.numel() for p in head.parameters())==5017760
    context=torch.randn(2,32,128,requires_grad=True)
    query=head.queries(context,torch.tensor([8,32]))
    assert query.shape==(10,4,128)
    query.square().sum().backward();assert context.grad is None
    reads=torch.randn(2,5,4,128,requires_grad=True);target=torch.randn(2,5,2048,requires_grad=True)
    head.loss(reads,target).backward()
    assert reads.grad.abs().sum()>0 and target.grad is None


def test_online_r1_identity_ignores_only_unused_position(monkeypatch):
    import qwen_vl.research.vggt_r1_teacher as r1
    import qwen_vl.research.geoaddress_teacher as address
    base=dict(weights_sha256='a',source_sha256='b',position='unused',position_sha256='one')
    monkeypatch.setattr(address,'teacher_identity',lambda *_:dict(base))
    first=r1.r1_teacher_identity('s','w')
    base['position_sha256']='two'
    assert first==r1.r1_teacher_identity('s','w')
    base['weights_sha256']='changed'
    assert first!=r1.r1_teacher_identity('s','w')
