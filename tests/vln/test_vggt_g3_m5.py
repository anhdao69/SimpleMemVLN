"""Combined visibility and state supervision; CPU reference, not a CUDA gate."""
import copy
from pathlib import Path
import pytest
import torch
import yaml
from qwen_vl.research.vggt_g1 import parse_g1, serving_config
from qwen_vl.models.episodic_forward import episodic_text_forward
from qwen_vl.models.vggt_g3_heads import NativeStateHead
from test_episodic_forward import tiny, _episode, _spec, _offline
from test_vggt_g3 import recurrence


def config():
    root=Path(__file__).resolve().parents[2]
    cfg=yaml.safe_load((root/'configs/episodic/science_common.yaml').read_text())
    cfg.update(yaml.safe_load((root/'configs/episodic/arm_m5.yaml').read_text()))
    cfg['research']=yaml.safe_load((root/'configs/vggt/g3_omega_h4.yaml').read_text())['research']
    cfg['research']['run_id']='G3_M5'
    return cfg


def test_combination_requires_exact_m5_and_strips_only_research():
    cfg=config()
    assert parse_g1(cfg)['run_id']=='G3_M5'
    assert serving_config(cfg)['memory']==cfg['memory']
    for key,value in [('recent_groups_including_current',8),('episodic_budget_tokens',600),
                      ('episodic_score','random_hash_v1'),('episodic_per_source_cap',300),
                      ('episodic_granularity','visual_group')]:
        bad=copy.deepcopy(cfg);bad['memory'][key]=value
        with pytest.raises(ValueError):parse_g1(bad)
    bad=copy.deepcopy(cfg);bad['research']['run_id']='G3'
    with pytest.raises(ValueError):parse_g1(bad)


def reference_kernel(q,k,v,*,g,beta,**kwargs):
    return recurrence(q,k,v,g,beta)


# Test-only independent recurrence replacing unavailable CUDA FLA.
reference_kernel.__module__='fla.ops.gated_delta_rule.test_reference'
reference_kernel.__name__='chunk_gated_delta_rule'


@pytest.mark.parametrize('support',[False,True])
def test_combined_checkpoint_gradients_visibility_and_causality(tiny,support):
    spec=_spec(tiny,4,1200,'committed_mean_l2_v1')
    layout,embeds,positions=_episode(6)
    # Place state supervision at layer2 in the tiny model; production is24.
    tiny.layers[2].linear_attn.chunk_gated_delta_rule=reference_kernel
    head=NativeStateHead(heads=4,key_dim=8,value_dim=6,query_hidden=16,decoder_hidden=24,target_dim=12)
    anchors=torch.tensor([layout.observation_ends[4]-1] if support else [],dtype=torch.long)
    lags=torch.tensor([8] if support else [],dtype=torch.long)
    targets=torch.randn(len(anchors),5,12)
    kwargs=dict(step_plan=(layout.prefix,layout.group_spans),layout=layout,episode_uid='ep',
                window_attention_batch_steps=3,state_head=head,anchors=anchors,lags=lags,
                targets=targets,state_layer=2)
    results=[]
    for checkpoint in (False,True):
        tiny.train();tiny.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        if checkpoint:tiny.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:tiny.gradient_checkpointing_disable()
        leaf=embeds.clone().requires_grad_();diag={}
        out,aux=episodic_text_forward(tiny,leaf,positions,spec,diagnostics=diag,**kwargs)
        (out[:,-4:].square().sum()+aux).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        assert leaf.grad[:,layout.visual_starts[0]:layout.visual_starts[0]+300].abs().sum()>0
        assert out.shape==embeds.shape and torch.isfinite(aux)
        results.append((out.detach(),aux.detach(),leaf.grad.clone(),diag['plans']))
    for a,b in zip(results[0][:3],results[1][:3]):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
    tiny.eval();tiny.gradient_checkpointing_disable()
    # No probe row ever becomes a visual candidate or future visibility edge.
    with torch.no_grad():
        base_diag={};base=_offline(tiny,spec,layout,embeds,positions,diagnostics=base_diag)
        torch.testing.assert_close(results[0][0],base,rtol=1e-4,atol=1e-5)
        for index,plan in base_diag['plans'].items():
            for t in range(layout.groups):
                assert (plan.group_ids(t)==results[0][3][index].group_ids(t)).all()
        changed=embeds.clone();changed[:,layout.group_spans[5][0]:]+=7
        future,_=episodic_text_forward(tiny,changed,positions,spec,**kwargs)
        torch.testing.assert_close(future[:,:layout.group_spans[5][0]],results[0][0][:,:layout.group_spans[5][0]],rtol=0,atol=0)
    if not support:assert results[0][1]==0


def test_empty_support_preserves_parameter_readiness_order(tiny):
    from qwen_vl.models.vggt_g3_forward import native_state_gdn
    gdn=tiny.layers[2].linear_attn;gdn.chunk_gated_delta_rule=reference_kernel
    head=NativeStateHead(heads=4,key_dim=8,value_dim=6,query_hidden=16,decoder_hidden=24,target_dim=12)
    parameters=list(gdn.named_parameters())+[(f'head.{n}',p) for n,p in head.named_parameters()]
    orders=[]
    for supported in (True,False):
        gdn.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True);order=[]
        handles=[p.register_hook(lambda g,n=n:order.append(n)) for n,p in parameters]
        anchors=torch.tensor([4] if supported else [],dtype=torch.long)
        out,loss=native_state_gdn(gdn,torch.randn(1,11,48,requires_grad=True),head,
            anchors,torch.full_like(anchors,8),torch.randn(len(anchors),5,12),uniform_empty=True)
        (out.square().sum()+loss).backward()
        for handle in handles:handle.remove()
        assert len(order)==len(parameters)
        orders.append(order)
    assert orders[0]==orders[1]
