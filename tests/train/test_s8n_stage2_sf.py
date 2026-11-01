import json
from pathlib import Path

import pytest

from scripts.memory_sweep import s8n_stage2_sf as sf

SCHEDULES = {'S8N_DAGGER_SF': (14195, 1775, 54), 'S8N_DAGGER_SF_REMAINING': (30815, 3852, 116),
             'S8N_DAGGER_SF_FULL': (45010, 5627, 169)}
S8N_TRAINING = dict(expected_total_steps=3852, warmup_steps=116, expected_warmup_steps=116)


def campaign_data(tmp_path, lanes=None):
    return dict(version=sf.VERSION, root=str(tmp_path), source='.', base=str(tmp_path), accumulation=2, seed=429,
                model='/model', min_free_disk_gib=220,
                parent=dict(weights_checkpoint='/s8n/final', weights_sha256='c' * 64),
                manifests={arm: dict(path=f'/m/{arm}.jsonl', episodes=n, sha256='0' * 64)
                           for arm, (n, _, _) in SCHEDULES.items()},
                lanes=lanes or dict(A=dict(arms=['S8N_DAGGER_SF', 'S8N_DAGGER_SF_REMAINING']),
                                    B=dict(arms=['S8N_DAGGER_SF_FULL'])))


def write_campaign(tmp_path, **kwargs):
    path = tmp_path / 'campaign.json'
    path.write_text(json.dumps(campaign_data(tmp_path, **kwargs)))
    return path


def test_each_arm_gets_its_manifest_epoch_schedule(tmp_path):
    campaign = sf.Campaign(write_campaign(tmp_path))
    for arm, (_, updates, warmup) in SCHEDULES.items():
        assert campaign.schedules[arm] == dict(expected_total_steps=updates, warmup_steps=warmup,
                                               expected_warmup_steps=warmup)
    assert campaign.schedules['S8N_DAGGER_SF_REMAINING'] == S8N_TRAINING      # S8N's own epoch length


@pytest.mark.parametrize('lanes', [dict(A=dict(arms=['S8N_DAGGER_SF', 'S8N_DAGGER_SF_REMAINING'])),
                                   dict(A=dict(arms=['S8N_DAGGER_SF', 'S8N_DAGGER_SF_REMAINING']),
                                        B=dict(arms=['S8N_DAGGER_SF_FULL', 'S8N_DAGGER_SF']))])
def test_campaign_trains_each_arm_exactly_once(tmp_path, lanes):
    with pytest.raises(ValueError, match='exactly once'):
        sf.Campaign(write_campaign(tmp_path, lanes=lanes))


def test_policy_text_changes_only_the_schedule():
    import yaml
    for arm, (n, updates, warmup) in SCHEDULES.items():
        policy = yaml.safe_load(sf.policy_text(Path('.'), arm, sf.epoch_schedule(n, 2)))
        assert policy == dict(training=dict(epochs=1, expected_total_steps=updates, warmup_steps=warmup,
                                            expected_warmup_steps=warmup))
    original = yaml.safe_load(Path('configs/vggt/g3_joint_one_epoch.yaml').read_text())
    remaining = yaml.safe_load(sf.policy_text(Path('.'), 'S8N_DAGGER_SF_REMAINING', sf.epoch_schedule(30815, 2)))
    assert remaining == original


def test_schedule_changes_name_only_differing_keys():
    assert sf.schedule_changes(S8N_TRAINING, sf.epoch_schedule(30815, 2)) == {}
    assert sf.schedule_changes(S8N_TRAINING, sf.epoch_schedule(45010, 2)) == dict(
        expected_total_steps=[3852, 5627], warmup_steps=[116, 169], expected_warmup_steps=[116, 169])


def test_foreign_steps_allows_only_own_shell_and_stopfix_lanes():
    lines = ['5489.0 training', '5489.110 s8n-stage2-sf-lane-A', '5489.111 s8n-stage2-sf-lane-B',
             '5489.extern extern', '5489.112 rwq_train11', '5489.113 s8n-stage2-lane-A']
    assert sf.foreign_steps(lines, '5489.110', '5489') == ['5489.112 rwq_train11', '5489.113 s8n-stage2-lane-A']
    assert sf.foreign_steps(lines[:4], '5489.110', '5489') == []


def test_check_init():
    plan = {rank: {update: [f'u{rank}{update}a', f'u{rank}{update}b'] for update in range(3)} for rank in range(4)}
    uids = [uid for updates in plan.values() for episodes in updates.values() for uid in episodes]
    good = dict(losses=[0.5, 0.5, 0.4], grad_norms=[9.0, 8.0, 8.5], accumulation=2)
    assert sf.check_init(good, plan, uids) == dict(passed=True, problems=[], episodes=24)
    assert not sf.check_init(dict(good, losses=[float('nan'), 0.5, 0.4]), plan, uids)['passed']
    assert not sf.check_init(dict(good, grad_norms=[9.0, float('inf'), 8.0]), plan, uids)['passed']
    assert not sf.check_init(dict(good, losses=[0.5, 0.5]), plan, uids)['passed']
    assert not sf.check_init(good, {r: u for r, u in plan.items() if r != 3}, uids)['passed']
    assert not sf.check_init(good, {**plan, 0: {0: plan[0][0], 1: plan[0][1]}}, uids)['passed']
    assert not sf.check_init(good, {**plan, 1: {**plan[1], 2: ['u12a']}}, uids)['passed']
    assert 'episodes outside the manifest' in sf.check_init(good, plan, uids[1:])['problems'][0]


SCALEVLN_ARM = 'S8N_DAGGER_SF_REMAINING_SCALEVLN50K'


def write_scalevln_campaign(tmp_path, arms=(SCALEVLN_ARM,), version=None):
    data = campaign_data(tmp_path, lanes=dict(A=dict(arms=list(arms))))
    data.update(version=version or sf.SCALEVLN_VERSION,
                manifests={arm: dict(path=f'/m/{arm}.jsonl', episodes=80815, sha256='0' * 64) for arm in arms})
    path = tmp_path / 'campaign.json'
    path.write_text(json.dumps(data))
    return path


def test_scalevln_campaign_trains_its_arm_with_its_manifest_epoch_schedule(tmp_path):
    campaign = sf.Campaign(write_scalevln_campaign(tmp_path))
    assert campaign.version == sf.SCALEVLN_VERSION and campaign.arm_data == sf.SCALEVLN_ARMS
    schedule = dict(expected_total_steps=10102, warmup_steps=304, expected_warmup_steps=304)
    assert campaign.schedules == {SCALEVLN_ARM: schedule}
    assert sf.schedule_changes(S8N_TRAINING, schedule) == dict(
        expected_total_steps=[3852, 10102], warmup_steps=[116, 304], expected_warmup_steps=[116, 304])


@pytest.mark.parametrize('arms', [('S8N_DAGGER_SF_REMAINING',), (SCALEVLN_ARM, 'S8N_DAGGER_SF_REMAINING'),
                                  (SCALEVLN_ARM, SCALEVLN_ARM)])
def test_scalevln_campaign_trains_only_its_arm_exactly_once(tmp_path, arms):
    with pytest.raises(ValueError, match='exactly once'):
        sf.Campaign(write_scalevln_campaign(tmp_path, arms))


def test_campaign_versions_do_not_mix_arms(tmp_path):
    with pytest.raises(ValueError, match='exactly once'):           # the first campaign cannot train the new arm
        sf.Campaign(write_scalevln_campaign(tmp_path, version=sf.VERSION))
    with pytest.raises(ValueError, match='Unknown campaign version'):
        sf.Campaign(write_scalevln_campaign(tmp_path, version='s8n_stage2_stopfix_campaign_v2'))


def test_policy_text_names_its_campaign_version():
    import yaml
    text = sf.policy_text(Path('.'), SCALEVLN_ARM, sf.epoch_schedule(80815, 2), sf.SCALEVLN_VERSION)
    assert text.startswith(f'# {sf.SCALEVLN_VERSION} {SCALEVLN_ARM}: ')
    assert yaml.safe_load(text) == dict(training=dict(epochs=1, expected_total_steps=10102, warmup_steps=304,
                                                      expected_warmup_steps=304))
    # the first campaign's policy files keep their exact text
    assert sf.policy_text(Path('.'), 'S8N_DAGGER_SF', sf.epoch_schedule(14195, 2)).startswith(
        f'# {sf.VERSION} S8N_DAGGER_SF: ')
    with pytest.raises(ValueError, match='not an arm'):
        sf.policy_text(Path('.'), SCALEVLN_ARM, sf.epoch_schedule(80815, 2))
    with pytest.raises(ValueError, match='not an arm'):
        sf.policy_text(Path('.'), 'S8N_DAGGER_SF', sf.epoch_schedule(14195, 2), sf.SCALEVLN_VERSION)


def test_train_command_binds_the_scalevln_arm(tmp_path):
    campaign = sf.Campaign(write_scalevln_campaign(tmp_path))
    command = sf.train_command(campaign, SCALEVLN_ARM, tmp_path / 'run', production=True, host_guard=691.2)
    torchrun = command[command.index('--') + 1:]
    assert torchrun[torchrun.index('--manifest') + 1] == f'/m/{SCALEVLN_ARM}.jsonl'
    assert torchrun[torchrun.index('--policy_config') + 1] == str(tmp_path / 'configs' / f'policy_{SCALEVLN_ARM}.yaml')
    assert torchrun[torchrun.index('--memory_config') + 1] == str(tmp_path / 'configs/arm_S8N.yaml')
    assert torchrun[-5:] == ['--init-weights-checkpoint', '/s8n/final', '--init-weights-sha256', 'c' * 64,
                             '--init-weights-new-schedule']


def test_train_command_binds_each_arms_manifest_policy_and_weights(tmp_path):
    campaign = sf.Campaign(write_campaign(tmp_path))
    for arm in SCHEDULES:
        command = sf.train_command(campaign, arm, tmp_path / 'run', production=True, host_guard=864)
        torchrun = command[command.index('--') + 1:]
        assert torchrun[torchrun.index('--manifest') + 1] == f'/m/{arm}.jsonl'
        assert torchrun[torchrun.index('--policy_config') + 1] == str(tmp_path / 'configs' / f'policy_{arm}.yaml')
        assert torchrun[torchrun.index('--memory_config') + 1] == str(tmp_path / 'configs/arm_S8N.yaml')
        assert torchrun[-5:] == ['--init-weights-checkpoint', '/s8n/final', '--init-weights-sha256', 'c' * 64,
                                 '--init-weights-new-schedule']
        assert '--sampler-epoch-offset' not in torchrun and '--resume_from_checkpoint' in torchrun
        assert command[command.index('--host-guard-gib') + 1] == '864'
