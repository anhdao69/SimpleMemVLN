import copy
from types import SimpleNamespace
import pytest
import torch
from qwen_vl.research.vggt_g1 import auxiliary_weight, parse_g1
from qwen_vl.train.vln_runtime import resolve_config
from qwen_vl.train.trainer import QwenSFTTrainer


def config():
    return resolve_config('configs/episodic/science_common.yaml','configs/vln_empty.yaml',
        'configs/vln_memory_window8.yaml','configs/vggt/g3_joint_one_epoch.yaml',
        'configs/vggt/g3_omega_h4.yaml')

@pytest.mark.parametrize('step',[0,1,770,771,2623,3852])
def test_constant_weight_from_first_update(step):
    assert auxiliary_weight(step,0,.1)==.1


def test_original_ramp_and_invalid_config():
    cfg=config();assert parse_g1(cfg)['ramp_updates']==771
    assert auxiliary_weight(0,771,.1)==0
    assert auxiliary_weight(385,771,.1)==pytest.approx(.1*385/771)
    cfg['research']['ramp_updates']=0;cfg['research']['head_seed']=430
    assert parse_g1(cfg)['ramp_updates']==0
    for bad in [-1,0.5,True]:
        invalid=copy.deepcopy(cfg);invalid['research']['ramp_updates']=bad
        with pytest.raises(ValueError):parse_g1(invalid)
    invalid=copy.deepcopy(cfg);invalid['research']['head_seed']=True
    with pytest.raises(ValueError):parse_g1(invalid)


def test_constant_loss_and_gradient_use_existing_global_normalization():
    class Model:
        training=True
        navigation_config={'research':{'ramp_updates':0,'lambda_max':.1}}
        def __call__(self,**kw):return dict(loss_sum=nav,aux_loss_sum=geo)
    nav=torch.tensor(12.,requires_grad=True);geo=torch.tensor(8.,requires_grad=True)
    model=Model()
    trainer=SimpleNamespace(model=model,state=SimpleNamespace(global_step=0),accelerator=SimpleNamespace(num_processes=4))
    actual,out=QwenSFTTrainer.compute_loss(trainer,model,dict(num_actions=3,aux_pair_count=2,update_aux_count=16),return_outputs=True,num_items_in_batch=24)
    expected=nav*4/24+.1*geo*4/16
    torch.testing.assert_close(actual,expected)
    assert out['aux_weight']==.1
    actual.backward();assert geo.grad.item()==pytest.approx(.025)


def test_seed_changes_head_without_changing_rng():
    from qwen_vl.models.vggt_g3_heads import NativeStateHead
    rng=torch.random.get_rng_state().clone()
    kwargs=dict(heads=2,key_dim=3,value_dim=4,query_hidden=8,decoder_hidden=10,target_dim=5)
    a=NativeStateHead(**kwargs,seed=429);b=NativeStateHead(**kwargs,seed=430);c=NativeStateHead(**kwargs,seed=430)
    assert torch.equal(rng,torch.random.get_rng_state())
    assert not torch.equal(a.region_embedding,b.region_embedding)
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in b.state_dict().items())


def test_production_recipe_changes_only_seeds_and_constant_weight():
    original=config()
    fixed=resolve_config('configs/episodic/science_common.yaml','configs/vln_empty.yaml',
        'configs/vln_memory_window8.yaml','configs/vggt/g3_joint_one_epoch.yaml',
        'configs/vggt/g3_fixed01_seed430_h4.yaml')
    expected=copy.deepcopy(original)
    expected['training']['seed']=430
    expected['research'].update(sample_seed=430,head_seed=430,ramp_updates=0)
    assert fixed==expected
    assert fixed['research']['lambda_max']==.1
    assert fixed['training']['epochs']==1
    assert fixed['training']['expected_total_steps']==3852
    assert fixed['training']['expected_warmup_steps']==116
    assert not fixed['model'].get('episodic',{}).get('enabled',False)
