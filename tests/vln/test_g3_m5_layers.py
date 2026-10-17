"""Layer ablation contracts and independent causal gradient reference."""
import copy
import pytest
import torch
from qwen_vl.research.vggt_g1 import parse_g1
from qwen_vl.models.vggt_g3_heads import build_state_heads
from qwen_vl.models.episodic_forward import episodic_text_forward
from test_vggt_g3_m5 import config, reference_kernel
from test_episodic_forward import tiny, _episode, _spec, _offline


def layered(layers):
    cfg=config();cfg['research'].pop('student_layer')
    cfg['research']['student_layers']=layers
    return cfg


@pytest.mark.parametrize('layers',[[8],[16],[24],[8,16,24]])
def test_layer_contract(layers):
    assert parse_g1(layered(layers))['student_layers']==layers


@pytest.mark.parametrize('layers',[[],[8,8],[16,8],[7],[32],[-1],[True],[8.0],'8'])
def test_reject_invalid_layers(layers):
    with pytest.raises(ValueError):parse_g1(layered(layers))


def test_reject_ambiguous_or_other_run():
    cfg=layered([8]);cfg['research']['student_layer']=24
    with pytest.raises(ValueError):parse_g1(cfg)
    cfg=layered([8]);cfg['research']['run_id']='G1'
    with pytest.raises(ValueError):parse_g1(cfg)


def heads(layers):
    return build_state_heads(layers,heads=4,key_dim=8,value_dim=6,
                            query_hidden=16,decoder_hidden=24,target_dim=12)


def test_independent_heads_preserve_rng_and_initialization():
    torch.manual_seed(123);before=torch.get_rng_state().clone()
    group=heads([0,2]);single=heads([2])
    assert torch.equal(before,torch.get_rng_state())
    for (name,p),q in zip(group['0'].named_parameters(),group['2'].parameters()):
        assert p.data_ptr()!=q.data_ptr()
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        torch.testing.assert_close(q,single['2'].state_dict()[name],rtol=0,atol=0)


@pytest.mark.parametrize('support',[False,True])
def test_multi_layer_mean_causal_checkpoint_gradients(tiny,support):
    spec=_spec(tiny,4,1200,'committed_mean_l2_v1')
    # Two groups suffice for independent state gradients; baseline tests cover
    # eviction and old-memory gradients with the identical selector.
    layout,embeds,positions=_episode(2)
    for index in (0,2):tiny.layers[index].linear_attn.chunk_gated_delta_rule=reference_kernel
    group=heads([0,2])
    anchors=torch.tensor([layout.observation_ends[0]-1] if support else [],dtype=torch.long)
    kwargs=dict(step_plan=(layout.prefix,layout.group_spans),layout=layout,episode_uid='ep',
                anchors=anchors,lags=torch.full_like(anchors,8),targets=torch.randn(len(anchors),5,12))
    params=list(tiny.parameters())+list(group.parameters())
    def run(selected,checkpoint=False):
        tiny.train();tiny.zero_grad(set_to_none=True);group.zero_grad(set_to_none=True)
        if checkpoint:tiny.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else:tiny.gradient_checkpointing_disable()
        leaf=embeds.clone().requires_grad_();diag={}
        out,aux=episodic_text_forward(tiny,leaf,positions,spec,state_heads=selected,diagnostics=diag,**kwargs)
        (out[:,-4:].square().sum()+aux).backward()
        grads=[None if p.grad is None else p.grad.clone() for p in params]
        return out.detach(),aux.detach(),leaf.grad.clone(),grads,diag
    both=run(group)
    singles=[run({str(i):group[str(i)]}) for i in (0,2)]
    torch.testing.assert_close(both[1],sum(x[1] for x in singles)/2)
    for index,g in enumerate(both[3]):
        if g is None:continue
        reference=sum(x[3][index] if x[3][index] is not None else torch.zeros_like(g) for x in singles)/2
        torch.testing.assert_close(g,reference,rtol=2e-4,atol=2e-5)
    checkpoint=run(group,True)
    for a,b in zip(both[:3],checkpoint[:3]):torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
    for a,b in zip(both[3],checkpoint[3]):
        if a is not None:torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in group.parameters())
    tiny.eval();tiny.gradient_checkpointing_disable()
    with torch.no_grad():
        ordinary=_offline(tiny,spec,layout,embeds,positions)
        torch.testing.assert_close(both[0],ordinary,rtol=1e-4,atol=1e-5)
        changed=embeds.clone();changed[:,layout.group_spans[1][0]:]+=7
        future,future_aux=episodic_text_forward(tiny,changed,positions,spec,state_heads=group,**kwargs)
        torch.testing.assert_close(future_aux,both[1],rtol=0,atol=0)
        # Explicit singleton API must retain the legacy head semantics.
        _,legacy=episodic_text_forward(tiny,embeds,positions,spec,state_head=group['2'],state_layer=2,**kwargs)
        torch.testing.assert_close(legacy,singles[1][1],rtol=0,atol=0)
    if not support:assert both[1]==0


@pytest.mark.parametrize('missing',[False,True])
def test_export_requires_every_configured_head(tmp_path,missing):
    import json
    from scripts.vln.export_vggt_g1 import export_checkpoint
    source=tmp_path/'checkpoint';source.mkdir()
    (source/'navigation.json').write_text(json.dumps({'config':layered([8,16,24])}))
    state={'backbone.weight':torch.ones(3)}
    for index in ([8,24] if missing else [8,16,24]):
        state[f'research_aux.state_heads.{index}.query.1.weight']=torch.ones(2)
    torch.save(state,source/'pytorch_model.bin')
    if missing:
        with pytest.raises(ValueError,match='head'):export_checkpoint(source,tmp_path/'export')
    else:
        dest=export_checkpoint(source,tmp_path/'export')
        assert list(torch.load(dest/'pytorch_model.bin',weights_only=True))==['backbone.weight']
        assert 'research' not in json.loads((dest/'navigation.json').read_text())['config']


def test_arm_configs_only_change_supervised_layers():
    import yaml
    from pathlib import Path
    root=Path(__file__).resolve().parents[2]
    baseline=yaml.safe_load((root/'configs/vggt/g3_m5_omega_h4.yaml').read_text())
    for name,layers in [('l8',[8]),('l16',[16]),('l8_l16_l24',[8,16,24])]:
        cfg=yaml.safe_load((root/f'configs/vggt/g3_m5_{name}_omega_h4.yaml').read_text())
        assert cfg['research'].pop('student_layers')==layers
        cfg['research']['student_layer']=24
        assert cfg==baseline


def test_reject_integer_head_keys_instead_of_silent_no_supervision(tiny):
    layout,embeds,positions=_episode(1)
    with pytest.raises(ValueError,match='canonical'):
        episodic_text_forward(tiny,embeds,positions,_spec(tiny,4,1200,'committed_mean_l2_v1'),
            step_plan=(layout.prefix,layout.group_spans),layout=layout,episode_uid='ep',
            state_heads={0:heads([0])['0']},anchors=torch.tensor([],dtype=torch.long),
            lags=torch.tensor([],dtype=torch.long),targets=torch.empty(0,5,12))
