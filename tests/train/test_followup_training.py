import hashlib
import json
import sys
import types

import pytest
import torch

from qwen_vl.data.episode_dataset import EpisodeLengthSampler

LENGTHS = [(i * 37) % 501 + 10 for i in range(1000)]


def order(**kwargs):
    epoch = kwargs.pop('epoch', 0)
    sampler = EpisodeLengthSampler(LENGTHS, group_size=8, pool_size=64, world_size=4, **kwargs)
    sampler.set_epoch(epoch)
    return list(sampler)


def test_epoch_offset_draws_the_later_epoch_order():
    assert order(seed=429) == order(seed=429, epoch_offset=0)
    assert order(seed=429, epoch_offset=1) == order(seed=429, epoch=1) == order(seed=430)
    assert order(seed=429, epoch_offset=1, epoch=1) == order(seed=429, epoch=2)
    assert order(seed=429, epoch_offset=1) != order(seed=429)
    assert sorted(order(seed=429, epoch_offset=1)) == list(range(len(LENGTHS)))
    with pytest.raises(ValueError):
        EpisodeLengthSampler(LENGTHS, epoch_offset=-1)


class Model(torch.nn.Module):
    """The trainable parameter kinds of the navigation wrapper (names as in Qwen3.5-4B)."""

    def __init__(self, navigation_config):
        super().__init__()
        self.navigation_config = navigation_config
        lm = torch.nn.Module()
        lm.embed_tokens = torch.nn.Embedding(11, 6)
        layer = torch.nn.Module()
        layer.input_layernorm = torch.nn.LayerNorm(6, bias=False)
        layer.mlp = torch.nn.Module()
        layer.mlp.up_proj = torch.nn.Linear(6, 9, bias=False)
        layer.mlp.down_proj = torch.nn.Linear(9, 6, bias=False)
        layer.linear_attn = torch.nn.Module()
        layer.linear_attn.conv1d = torch.nn.Conv1d(4, 4, 3, groups=4, bias=False)
        layer.linear_attn.A_log = torch.nn.Parameter(torch.zeros(4))
        lm.layers = torch.nn.ModuleList([layer])
        lm.norm = torch.nn.LayerNorm(6, bias=False)
        visual = torch.nn.Module()
        visual.merger = torch.nn.Linear(6, 6)
        visual.merger.requires_grad_(False)
        self.backbone = torch.nn.Module()
        self.backbone.model = torch.nn.Module()
        self.backbone.model.language_model = lm
        self.backbone.model.visual = visual
        self.backbone.lm_head = torch.nn.Linear(6, 11, bias=False)


def build(monkeypatch, training):
    from qwen_vl.train import trainer as trainer_module
    from qwen_vl.train import zero_muon
    captured = {}

    class FakeCPUAdam:
        def __init__(self, groups, **kwargs):
            captured.update(kind='adam', groups=groups, kwargs=kwargs)

    fake = types.ModuleType('deepspeed.ops.adam')
    fake.DeepSpeedCPUAdam = FakeCPUAdam
    monkeypatch.setitem(sys.modules, 'deepspeed', types.ModuleType('deepspeed'))
    monkeypatch.setitem(sys.modules, 'deepspeed.ops', types.ModuleType('deepspeed.ops'))
    monkeypatch.setitem(sys.modules, 'deepspeed.ops.adam', fake)
    monkeypatch.setattr(zero_muon, 'zero_muon_cpu_adam',
                        lambda groups, layout, **kwargs: captured.update(kind='muon', groups=groups, layout=layout,
                                                                         kwargs=kwargs) or 'optimizer')
    trainer = object.__new__(trainer_module.QwenSFTTrainer)
    trainer.model = Model(dict(training=dict(training, cpu_optimizer_impl='deepspeed_cpu_adam')))
    trainer.optimizer = None
    trainer.args = types.SimpleNamespace(
        learning_rate=5e-6, mm_projector_lr=5e-6, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.95,
        adam_epsilon=1e-8, deepspeed=dict(zero_optimization=dict(stage=2, offload_optimizer=dict(device='cpu'))))
    trainer.create_optimizer()
    names = {id(p): n for n, p in trainer.model.named_parameters()}
    groups = [dict({k: v for k, v in g.items() if k != 'params'}, names=[names[id(p)] for p in g['params']])
              for g in captured['groups']]
    return captured, groups


def test_adamw_groups_are_unchanged_without_a_matrix_optimizer(monkeypatch):
    captured, groups = build(monkeypatch, {})
    assert captured['kind'] == 'adam'
    assert [g['names'] for g in groups] == [
        ['backbone.model.language_model.embed_tokens.weight', 'backbone.model.language_model.layers.0.mlp.up_proj.weight',
         'backbone.model.language_model.layers.0.mlp.down_proj.weight',
         'backbone.model.language_model.layers.0.linear_attn.conv1d.weight', 'backbone.lm_head.weight'],
        ['backbone.model.language_model.layers.0.input_layernorm.weight',
         'backbone.model.language_model.layers.0.linear_attn.A_log', 'backbone.model.language_model.norm.weight']]
    assert [g['weight_decay'] for g in groups] == [0.01, 0.0]


def test_muon_takes_only_hidden_matrices(monkeypatch):
    spec = dict(name='muon_zero_offload_v1', scope='language_model_layers_2d', momentum=0.95, nesterov=True,
                ns_steps=5, ns_coefficients=[3.4445, -4.775, 2.0315], eps=1.0e-07, adjust_lr_fn='match_rms_adamw',
                group_numel=60)
    captured, groups = build(monkeypatch, dict(matrix_optimizer=dict(spec, group_numel=50)))
    assert captured['kind'] == 'muon'
    adam = [g for g in groups if not g.get('use_muon')]
    muon = [g for g in groups if g.get('use_muon')]
    assert [g['names'] for g in adam] == [
        ['backbone.model.language_model.embed_tokens.weight',
         'backbone.model.language_model.layers.0.linear_attn.conv1d.weight', 'backbone.lm_head.weight'],
        ['backbone.model.language_model.layers.0.input_layernorm.weight',
         'backbone.model.language_model.layers.0.linear_attn.A_log', 'backbone.model.language_model.norm.weight']]
    assert [g['names'] for g in muon] == [['backbone.model.language_model.layers.0.mlp.up_proj.weight'],
                                          ['backbone.model.language_model.layers.0.mlp.down_proj.weight']]
    assert all(g['lr'] == 5e-6 and g['weight_decay'] == 0.01 for g in muon)
    assert captured['kwargs'] == dict(lr=5e-6, betas=(0.9, 0.95), eps=1e-8, adamw_mode=True, fp32_optimizer_states=True)
    assert sorted(captured['layout']) == [0, 1]


def test_muon_refuses_reordering_zero_configs(monkeypatch):
    spec = dict(name='muon_zero_offload_v1', scope='language_model_layers_2d', momentum=0.95, nesterov=True,
                ns_steps=5, ns_coefficients=[3.4445, -4.775, 2.0315], eps=1.0e-07, adjust_lr_fn='match_rms_adamw',
                group_numel=60)
    from qwen_vl.train import trainer as trainer_module
    trainer = object.__new__(trainer_module.QwenSFTTrainer)
    trainer.model = Model(dict(training=dict(matrix_optimizer=spec, cpu_optimizer_impl='deepspeed_cpu_adam')))
    trainer.optimizer = None
    trainer.args = types.SimpleNamespace(
        learning_rate=5e-6, mm_projector_lr=5e-6, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.95,
        adam_epsilon=1e-8, deepspeed=dict(zero_optimization=dict(stage=2, round_robin_gradients=True,
                                                                 offload_optimizer=dict(device='cpu'))))
    with pytest.raises(ValueError):
        trainer.create_optimizer()


def test_weights_initialization_checks_hash_and_contract(tmp_path, monkeypatch):
    from qwen_vl.train import lane_initialization, vln_runtime
    config = dict(model=dict(output_mode='qwen_text'), training=dict(backbone_lr=5e-6))
    (tmp_path / 'navigation.json').write_text(json.dumps(dict(config=config, serializer='v1')))
    (tmp_path / 'pytorch_model.bin').write_bytes(b'weights')
    digest = hashlib.sha256(b'weights').hexdigest()
    with pytest.raises(ValueError, match='SHA-256'):
        lane_initialization.initialize_weights_from_checkpoint(tmp_path, '0' * 64, None, config)
    with pytest.raises(ValueError, match='navigation config'):
        lane_initialization.initialize_weights_from_checkpoint(
            tmp_path, digest, None, dict(config, training=dict(backbone_lr=1e-5)))
    serializer = types.SimpleNamespace(metadata=lambda: dict(config=config, serializer='v1'))
    monkeypatch.setattr(vln_runtime, 'load_checkpoint',
                        lambda path, base: (torch.nn.Linear(2, 3), serializer))
    _, _, report = lane_initialization.initialize_weights_from_checkpoint(tmp_path, digest, None, config)
    assert report['weights_sha256'] == digest and report['optimizer'] == report['scheduler'] == 'fresh'
    assert report['parameters'] == 9
