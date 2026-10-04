from copy import deepcopy
import json
import pytest
import torch


def configs():
    parent = dict(model={'output_mode': 'candidate_logits', 'action_head_mode': 'copied_linear'},
                  observations={'feedback_format': 'none'}, memory={'mode': 'full_context'},
                  runtime={'text_attention': 'flash_attention_2'}, training={'backbone_lr': 5e-6})
    child = deepcopy(parent)
    child['model']['step_lane'] = {'enabled': True}
    child['training']['step_lane_lr'] = 1e-4
    return parent, child


@pytest.mark.parametrize('section,key,value', [
    ('model', 'action_head_mode', 'lm_rows_trainable'),
    ('observations', 'feedback_format', 'candidate_token'),
    ('memory', 'mode', 'window8'),
    ('runtime', 'text_attention', 'sdpa'),
])
def test_warmstart_forbids_contract_conversion(section, key, value):
    from qwen_vl.train.lane_initialization import validate_parent_contract
    parent, child = configs()
    validate_parent_contract(parent, child)
    child[section][key] = value
    with pytest.raises(ValueError, match='parent'):
        validate_parent_contract(parent, child)


def test_warmstart_rejects_already_adapted_parent_and_disabled_target():
    from qwen_vl.train.lane_initialization import validate_parent_contract
    parent, child = configs()
    child['model']['step_lane']['enabled'] = False
    with pytest.raises(ValueError):
        validate_parent_contract(parent, child)
    child['model']['step_lane']['enabled'] = True
    parent['model']['step_lane'] = {'enabled': True}
    with pytest.raises(ValueError):
        validate_parent_contract(parent, child)


def test_hashes_all_actual_weight_files(tmp_path):
    from qwen_vl.train.lane_initialization import checkpoint_identity
    (tmp_path/'navigation.json').write_text(json.dumps({'config': configs()[0]}))
    (tmp_path/'model-00001-of-00002.safetensors').write_bytes(b'first')
    (tmp_path/'model-00002-of-00002.safetensors').write_bytes(b'second')
    identity = checkpoint_identity(tmp_path)
    assert len(identity['weight_sha256']) == 2
    previous = deepcopy(identity)
    (tmp_path/'model-00002-of-00002.safetensors').write_bytes(b'changed')
    assert checkpoint_identity(tmp_path)['weight_sha256'] != previous['weight_sha256']


@pytest.mark.parametrize('existing', ['final','checkpoint-1'])
def test_rejected_existing_output_does_not_load_or_replace_provenance(tmp_path, monkeypatch, existing):
    import sys
    import qwen_vl.train.train_episode as entry
    import qwen_vl.train.lane_initialization as initialization
    import qwen_vl.models.action_loss as balance
    (tmp_path/existing).mkdir()
    report=tmp_path/'lane_initialization.json'
    report.write_text('{"original":"identity"}\n')
    monkeypatch.setenv('WORLD_SIZE','4')
    monkeypatch.setattr(balance,'resolve_class_balance',lambda *a,**k:None)
    def forbidden(*a,**k):
        raise AssertionError('Rejected launch attempted to load/replace parent')
    monkeypatch.setattr(initialization,'initialize_step_lane_from_policy',forbidden)
    monkeypatch.setattr(sys,'argv',['train','--vln_config','configs/vln_dual_full_no_history_r2r.yaml',
                                   '--output_config','configs/vln_empty.yaml','--manifest','unused',
                                   '--output_dir',str(tmp_path),'--init-policy-checkpoint','unused'])
    with pytest.raises(ValueError,match='Explicit resume'):
        entry.train_episode()
    assert report.read_text()=='{"original":"identity"}\n'


def test_warmstart_reuses_trained_classifier_objects_and_values(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from torch import nn
    import qwen_vl.train.vln_runtime as runtime
    import qwen_vl.models.install_step_lane as installer
    import qwen_vl.research.lane_contract as contract
    import qwen_vl.data.episode_serializer as serialization
    from qwen_vl.train.lane_initialization import initialize_step_lane_from_policy
    parent,target=configs()
    (tmp_path/'navigation.json').write_text(json.dumps({'config':parent}))
    model=nn.Module()
    model.backbone=nn.Linear(4,4)
    model.classifier=nn.Linear(4,4)
    with torch.no_grad():
        model.classifier.weight.fill_(7)
    model.config=SimpleNamespace(text_config=object())
    original=dict(model.named_parameters())
    values={n:p.detach().clone() for n,p in original.items()}
    processor=object()
    monkeypatch.setattr(runtime,'load_checkpoint',lambda *a:(model,SimpleNamespace(processor=processor)))
    monkeypatch.setattr(contract,'parse_step_lane_spec',lambda *a:SimpleNamespace(to_dict=lambda:{'version':'test'}))
    def install(m,spec,**kwargs):
        m.backbone.step_lane=nn.Linear(4,4)
    monkeypatch.setattr(installer,'install_step_lanes',install)
    monkeypatch.setattr(serialization,'EpisodeSerializer',lambda p,c:SimpleNamespace(processor=p,config=c))
    adapted,serializer,report=initialize_step_lane_from_policy(tmp_path,'base',target,lane_seed=17)
    assert adapted is model and adapted.classifier.weight is original['classifier.weight']
    for n,p in original.items():
        assert dict(adapted.named_parameters())[n] is p
        assert torch.equal(p,values[n])
    assert serializer.processor is processor
    assert report['optimizer']=='fresh' and report['lane_seed']==17
