import json
from pathlib import Path

import pytest

from scripts.memory_sweep import s8n_stage2 as stage2


def test_epoch_schedule_reproduces_s8n_and_sizes_the_dagger_epoch():
    assert stage2.epoch_schedule(30815, 2) == dict(expected_total_steps=3852, warmup_steps=116,
                                                   expected_warmup_steps=116)
    assert stage2.epoch_schedule(14195, 2) == dict(expected_total_steps=1775, warmup_steps=54,
                                                   expected_warmup_steps=54)
    assert stage2.epoch_schedule(800, 1)['warmup_steps'] == 6       # 0.03 * 200 is exactly 6, not 7


def test_policy_text_changes_only_the_schedule():
    import yaml
    text = stage2.policy_text(Path('.'), stage2.epoch_schedule(14195, 2))
    policy = yaml.safe_load(text)
    assert policy == dict(training=dict(epochs=1, expected_total_steps=1775, warmup_steps=54,
                                        expected_warmup_steps=54))


def test_foreign_steps_allows_own_shell_and_lane_steps():
    lines = ['5489.0 training', '5489.91 s8n-stage2-lane-A', '5489.92 s8n-stage2-lane-B', '5489.extern extern',
             '5489.93 rwq_train11', '5489.94 s8n-stage2-alignment']
    assert stage2.foreign_steps(lines, '5489.91', '5489') == ['5489.93 rwq_train11', '5489.94 s8n-stage2-alignment']
    assert stage2.foreign_steps(lines[:4], '5489.91', '5489') == []


def test_cpu_list():
    assert stage2.cpu_list('0-2,64-65,7') == {0, 1, 2, 7, 64, 65}


def test_compare_init():
    plan = {0: {0: ['a', 'b']}}
    good = dict(S8N_DAGGER=dict(losses=[0.1, 0.1, 0.09], grad_norms=[3.0, 3.0, 2.9], plan=plan),
                S8N_DAGGER_ORACLE=dict(losses=[0.3, 0.3, 0.28], grad_norms=[9.0, 9.0, 8.0], plan=plan))
    assert stage2.compare_init(good, 0.5)['passed']
    other = json.loads(json.dumps(good))
    other['S8N_DAGGER_ORACLE']['plan'] = {0: {0: ['b', 'a']}}
    other['S8N_DAGGER']['plan'] = plan
    assert 'the arms see different data orders' in stage2.compare_init(other, 0.5)['problems']
    assert not stage2.compare_init(dict(good, S8N_DAGGER=dict(good['S8N_DAGGER'], losses=[0.4, 0.4, 0.4])),
                                   0.5)['passed']
    assert not stage2.compare_init(dict(good, S8N_DAGGER_ORACLE=dict(good['S8N_DAGGER_ORACLE'],
                                                                   losses=[0.05, 0.05, 0.05])), 0.5)['passed']
    assert not stage2.compare_init(dict(good, S8N_DAGGER=dict(good['S8N_DAGGER'],
                                                            losses=[float('nan'), 0.1, 0.1])), 0.5)['passed']


def test_train_command_binds_manifest_policy_and_weights(tmp_path):
    campaign = object.__new__(stage2.Campaign)
    campaign.data = dict(seed=429, model='/model', min_free_disk_gib=220,
                         manifests=dict(S8N_DAGGER_ORACLE=dict(path='/m/oracle.jsonl')))
    campaign.configs, campaign.base = tmp_path, tmp_path
    campaign.parent = dict(weights_checkpoint='/s8n/final', weights_sha256='c' * 64)
    command = stage2.train_command(campaign, 'S8N_DAGGER_ORACLE', tmp_path / 'run', production=True, host_guard=864)
    torchrun = command[command.index('--') + 1:]
    assert torchrun[torchrun.index('--manifest') + 1] == '/m/oracle.jsonl'
    assert torchrun[torchrun.index('--policy_config') + 1] == str(tmp_path / 'policy_stage2.yaml')
    assert torchrun[-5:] == ['--init-weights-checkpoint', '/s8n/final', '--init-weights-sha256', 'c' * 64,
                             '--init-weights-new-schedule']
    assert '--sampler-epoch-offset' not in torchrun and '--resume_from_checkpoint' in torchrun
    assert command[command.index('--host-guard-gib') + 1] == '864'
