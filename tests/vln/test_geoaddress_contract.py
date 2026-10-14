import copy
import pytest
from qwen_vl.research.vggt_g1 import parse_g1,serving_config,serving_state_dict
from qwen_vl.contracts import serializer_version


def config(run):
    return dict(model=dict(output_mode='qwen_text',geoaddress=dict(version='geoaddress_v1',layer=24,rank=8,heads=[0,1],init_seed=429)),
                memory=dict(mode='window8'),research=dict(run_id=run,student_layer=24,lambda_max=0 if run=='A0' else .1,
                relation_projection='direct-native-qk-cosine-v1',ramp_updates=771,head_lr=1e-4,adapter_lr=5e-5,teacher_source='src',teacher_weights='weights',teacher_identity={'test':1}))


def test_a0_teacher_free_config_and_runtime_export():
    cfg=config('A0')
    for key in ('teacher_source','teacher_weights','teacher_identity'):cfg['research'].pop(key)
    assert parse_g1(cfg)['run_id']=='A0'
    served=serving_config(cfg)
    assert served['model']['geoaddress']==cfg['model']['geoaddress'] and 'research' not in served
    assert serializer_version(cfg)=='vln_append_only_chat_geoaddress_v1'
    weights={'backbone.model.language_model.layers.24.linear_attn.geoaddress.q_down.weight':1,'research_aux.address_projector.fc1.weight':2}
    assert len(serving_state_dict(weights))==1


def test_geoaddress_wrong_architecture_or_missing_runtime_config_rejected():
    from qwen_vl.research.geoaddress_contract import parse_geoaddress
    cfg=config('A0')
    assert parse_geoaddress(cfg)['layer']==24
    for key,value in [('layer',23),('rank',16),('heads',[2,3]),('version','unknown')]:
        bad=copy.deepcopy(cfg);bad['model']['geoaddress'][key]=value
        with pytest.raises(ValueError):parse_geoaddress(bad)
    bad=copy.deepcopy(cfg);bad['model'].pop('geoaddress')
    with pytest.raises(ValueError):parse_g1(bad)
    bad=copy.deepcopy(cfg);bad['research']['run_id']='G3'
    with pytest.raises(ValueError):parse_g1(bad)


def test_adapter_optimizer_groups_are_separate_and_exact():
    import torch
    from transformers import TrainingArguments
    from qwen_vl.train.trainer import QwenSFTTrainer
    from qwen_vl.models.geoaddress import GeoAddress
    model=torch.nn.Module();model.backbone=torch.nn.Linear(6,6)
    model.backbone.geoaddress=GeoAddress(hidden_size=6,head_dim=3,rank=2)
    model.research_aux=torch.nn.ModuleDict({'address_projector':torch.nn.Linear(3,4)})
    model.navigation_config=config('A1')
    args=TrainingArguments(output_dir='/tmp/a1-test',use_cpu=True,report_to='none',learning_rate=5e-6,weight_decay=.01)
    args.mm_projector_lr=5e-6
    trainer=QwenSFTTrainer(model=model,args=args);opt=trainer.create_optimizer()
    grouped={id(p):g for g in opt.param_groups for p in g['params']}
    assert len(grouped)==sum(len(g['params']) for g in opt.param_groups)==len(list(model.parameters()))
    for name,p in model.named_parameters():
        expected=5e-5 if '.geoaddress.' in name else 1e-4 if name.startswith('research_aux.') else 5e-6
        assert grouped[id(p)]['lr']==expected
