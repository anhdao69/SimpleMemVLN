import copy
import json

import pytest
import yaml

from scripts.memory_sweep import s8n_followup as f

PARENT_LOG = {1: dict(loss=0.4837, grad_norm=50.74), 2: dict(loss=0.4863, grad_norm=67.20),
              3: dict(loss=0.5073, grad_norm=80.76)}
PLAN = {rank: {u: [f'e{rank}{u}a', f'e{rank}{u}b'] for u in range(3)} for rank in range(4)}


def run(losses, grad_norms, plan=PLAN):
    return dict(losses=list(losses), grad_norms=list(grad_norms), plan=copy.deepcopy(plan))


def runs():
    reference = run([r['loss'] for r in PARENT_LOG.values()], [r['grad_norm'] for r in PARENT_LOG.values()])
    other = {rank: {u: list(reversed(v)) for u, v in updates.items()} for rank, updates in PLAN.items()}
    return {f.REFERENCE: reference,
            'S8N_LR1E5': run([0.4837, 0.4863, 0.4990], [50.74, 67.20, 70.0]),
            'S8N_MUON': run([0.4837, 0.4863, 0.4000], [50.74, 67.20, 30.0]),
            'S8N_E2': run([0.12, 0.11, 0.13], [3.0, 2.5, 2.8], other)}


def test_reproduction_passes_for_matching_runs():
    result = f.compare_reproduction(PARENT_LOG, PLAN, runs())
    assert result['passed'], result['problems']


@pytest.mark.parametrize('arm,field,index,value', [
    (f.REFERENCE, 'losses', 2, 0.5083),          # update-3 loss off by 2e-3 relative
    (f.REFERENCE, 'grad_norms', 0, 52.0),
    ('S8N_MUON', 'losses', 1, 0.4870),           # update 2 still holds the original weights
    ('S8N_LR1E5', 'grad_norms', 0, 49.0),
    ('S8N_E2', 'losses', 0, 0.47),               # not started from the trained weights
])
def test_reproduction_detects_deviations(arm, field, index, value):
    data = runs()
    data[arm][field][index] = value
    assert not f.compare_reproduction(PARENT_LOG, PLAN, data)['passed']


def test_reproduction_checks_data_order():
    data = runs()
    data['S8N_LR1E5']['plan'][2][1] = ['x', 'y']
    assert not f.compare_reproduction(PARENT_LOG, PLAN, data)['passed']
    data = runs()
    data['S8N_E2']['plan'] = copy.deepcopy(PLAN)
    assert not f.compare_reproduction(PARENT_LOG, PLAN, data)['passed']


def test_arm_declarations_change_only_their_keys():
    assert f.ARMS['S8N_E2']['training'] == {} and f.ARMS['S8N_E2']['continue_parent']
    assert f.ARMS['S8N_LR1E5']['training'] == dict(backbone_lr=1e-5, classifier_lr=1e-5)
    assert f.ARMS['S8N_MUON']['training'] == dict(matrix_optimizer=f.MUON)
    from qwen_vl.research.matrix_optimizer_contract import parse_matrix_optimizer
    assert parse_matrix_optimizer(dict(cpu_optimizer_impl='deepspeed_cpu_adam', matrix_optimizer=f.MUON)) == f.MUON


def test_policy_files_extend_the_one_epoch_policy(tmp_path):
    (tmp_path / 'configs/vggt').mkdir(parents=True)
    base = dict(training=dict(epochs=1, expected_total_steps=3852, warmup_steps=116, expected_warmup_steps=116))
    (tmp_path / 'configs/vggt/g3_joint_one_epoch.yaml').write_text(yaml.safe_dump(base))
    assert yaml.safe_load(f.policy_text(tmp_path, f.REFERENCE)) == base
    assert yaml.safe_load(f.policy_text(tmp_path, 'S8N_E2')) == base
    lr = yaml.safe_load(f.policy_text(tmp_path, 'S8N_LR1E5'))
    assert lr == dict(training=dict(base['training'], backbone_lr=1e-5, classifier_lr=1e-5))
    muon = yaml.safe_load(f.policy_text(tmp_path, 'S8N_MUON'))
    assert muon['training']['matrix_optimizer'] == f.MUON
    assert isinstance(muon['training']['matrix_optimizer']['momentum'], float)


def test_config_diff_names_changed_paths():
    a = dict(training=dict(lr=1, seed=429), memory=dict(score='S8N'))
    b = dict(training=dict(lr=2, seed=429), memory=dict(score='S8N'), extra=1)
    assert f.config_diff(a, b) == ['extra', 'training.lr']


class FakeCampaign:
    def __init__(self, tmp_path):
        self.data = dict(manifest='/m.jsonl', model='/model', seed=429, min_free_disk_gib=220)
        self.parent = dict(weights_checkpoint='/s8n/final', weights_sha256='c' * 64)
        self.base = tmp_path
        self.configs = tmp_path / 'configs'

    def arm_memory_path(self, arm, role='training'):
        return self.configs / 'arm_S8N.yaml'

    def policy_path(self, arm):
        return self.configs / f'policy_{arm}.yaml'


def test_train_command_uses_the_arm_policy_and_parent_weights(tmp_path):
    c = FakeCampaign(tmp_path)
    for arm in (f.REFERENCE, *f.ARMS):
        command = f.train_command(c, arm, tmp_path / arm, production=True, host_guard=432.0)
        assert command[command.index('--policy_config') + 1] == str(c.policy_path(arm))
        assert command[command.index('--memory_config') + 1] == str(tmp_path / 'configs/arm_S8N.yaml')
        assert command[command.index('--deepspeed') + 1] == 'configs/deepspeed_zero2_cpu_optimizer.json'
        continuing = '--init-weights-checkpoint' in command
        assert continuing == (arm == 'S8N_E2')
        if continuing:
            assert command[-6:] == ['--init-weights-checkpoint', '/s8n/final', '--init-weights-sha256', 'c' * 64,
                                    '--sampler-epoch-offset', '1']
        assert command.count('--resume_from_checkpoint') == 1


def test_exposure_plan_reads_selected_updates(tmp_path):
    for rank in range(4):
        rows = [dict(update=u, episode_uid=f'r{rank}u{u}e{i}') for u in range(5) for i in range(2)]
        rows.append(dict(update=1, episode_uid=f'r{rank}u1e0'))        # replay after a resume
        (tmp_path / f'exposures_rank{rank}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    plan = f.exposure_plan(tmp_path, range(3))
    assert sorted(plan[0]) == [0, 1, 2] and plan[3][1] == ['r3u1e0', 'r3u1e1']
