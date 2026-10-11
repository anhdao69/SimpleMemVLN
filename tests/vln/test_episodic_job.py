"""In-job lifecycle with mocked heavy stages: one trainer per logical run."""
import json
from pathlib import Path
import pytest
import scripts.vln.episodic_job as job
from qwen_vl.train.campaign_lease import Lease


def _attempt(tmp_path, attempt, profile):
    d = tmp_path / 'runs' / 'P1-V0-s429' / attempt
    d.mkdir(parents=True)
    (d / 'launch_manifest.json').write_text(json.dumps(dict(
        campaign_root=str(tmp_path), run_id='P1-V0-s429', arm='v0', profile=profile, attempt=attempt,
        qualification='full', resources=dict(world=4, mem='512G'), submit_ablations=[])))
    return d


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    (tmp_path / 'campaign_manifest.json').write_text(json.dumps(dict(
        epochs_to_train=1, verify_updates=50, first_recovery_save_step=50)))
    calls = dict(qualify=[], train=[], upload=[])
    monkeypatch.setattr(job, 'preflight', lambda a: dict(problems=[]))
    monkeypatch.setattr(job, 'qualify', lambda a, mode: calls['qualify'].append((a.launch['attempt'], mode))
                        or dict(passed=True))
    monkeypatch.setattr(job, 'upload', lambda a, out: calls['upload'].append(a.launch['attempt']) or 0)
    monkeypatch.setattr(job, 'slurm_state', lambda job_id: None)       # previous jobs are gone
    return tmp_path, calls


def _fake_epoch(out, step):
    checkpoint = out / f'checkpoint-{step}'
    checkpoint.mkdir(parents=True)
    for name in ('pytorch_model.bin', 'trainer_state.json', 'resume_contract.json', 'navigation.json',
                 'optimizer.pt', 'scheduler.pt', *[f'rng_state_{r}.pth' for r in range(4)]):
        (checkpoint / name).write_text(json.dumps(dict(global_step=step)) if name == 'trainer_state.json' else 'x')
    from qwen_vl.train.recovery import mark_complete
    mark_complete(checkpoint, 4, True)
    (out / 'trainer_state.json').write_text(json.dumps(dict(max_steps=7704, global_step=step)))


def test_winner_trains_once_and_sibling_stands_by(campaign, monkeypatch):
    root, calls = campaign
    monkeypatch.setenv('SLURM_JOB_ID', '11')
    first = _attempt(root, 'P1-V0-s429-H4-a1', 'H4')

    def train(a):
        calls['train'].append(a.launch['attempt'])
        out = a.dir / 'train'
        _fake_epoch(out, 3852)
        return 0, out
    monkeypatch.setattr(job, 'train', train)
    monkeypatch.setattr('sys.argv', ['x', str(first)])
    assert job.main() == 0
    assert calls['train'] == ['P1-V0-s429-H4-a1'] and calls['upload'] == ['P1-V0-s429-H4-a1']
    assert Lease(root, 'P1-V0-s429').read()['state'] == 'COMPLETE'
    monkeypatch.setenv('SLURM_JOB_ID', '22')
    second = _attempt(root, 'P1-V0-s429-H2-a1', 'H2')
    monkeypatch.setattr('sys.argv', ['x', str(second)])
    assert job.main() == 0
    assert calls['train'] == ['P1-V0-s429-H4-a1']                 # no duplicate trajectory
    assert json.loads((second / 'status.json').read_text())['state'] == 'DUPLICATE_STANDBY'
    assert calls['qualify'] == [('P1-V0-s429-H4-a1', 'full')]       # standby never spent qualification


def test_interrupted_run_is_not_published_and_resume_requalifies(campaign, monkeypatch):
    root, calls = campaign
    monkeypatch.setenv('SLURM_JOB_ID', '31')
    attempt = _attempt(root, 'P1-V0-s429-H4-a1', 'H4')

    def stopped(a):
        out = a.dir / 'train'; out.mkdir(exist_ok=True)
        (out / 'trainer_state.json').write_text(json.dumps(dict(max_steps=7704, global_step=1200)))
        return 0, out
    monkeypatch.setattr(job, 'train', stopped)
    monkeypatch.setattr('sys.argv', ['x', str(attempt)])
    assert job.main() == 3
    assert calls['upload'] == []
    status = json.loads((attempt / 'status.json').read_text())
    assert status['state'] == 'FAILED' and status['stage'] == 'interrupted_before_epoch_boundary'
    monkeypatch.setenv('SLURM_JOB_ID', '32')                        # resubmitted same attempt
    monkeypatch.setattr(job, 'train', lambda a: (_fake_epoch(a.dir / 'train', 3852), (0, a.dir / 'train'))[1])
    assert job.main() == 0
    assert calls['qualify'][-1] == ('P1-V0-s429-H4-a1', 'startup')  # new allocation still gated
    assert calls['upload'] == ['P1-V0-s429-H4-a1']


def test_completed_attempt_rerun_publishes_without_training(campaign, monkeypatch):
    root, calls = campaign
    monkeypatch.setenv('SLURM_JOB_ID', '41')
    attempt = _attempt(root, 'P1-V0-s429-H4-a1', 'H4')
    monkeypatch.setattr(job, 'train', lambda a: (calls['train'].append(1), _fake_epoch(a.dir / 'train', 3852),
                                                 (0, a.dir / 'train'))[2])
    monkeypatch.setattr('sys.argv', ['x', str(attempt)])
    assert job.main() == 0 and len(calls['train']) == 1
    monkeypatch.setenv('SLURM_JOB_ID', '42')                         # e.g. upload retry
    assert job.main() == 0
    assert len(calls['train']) == 1 and calls['upload'] == ['P1-V0-s429-H4-a1'] * 2


def test_resume_refuses_while_previous_job_is_live(campaign, monkeypatch):
    root, calls = campaign
    monkeypatch.setenv('SLURM_JOB_ID', '51')
    attempt = _attempt(root, 'P1-V0-s429-H4-a1', 'H4')
    def interrupted(a):
        out = a.dir / 'train'; out.mkdir(exist_ok=True)
        (out / 'trainer_state.json').write_text(json.dumps(dict(max_steps=7704, global_step=10)))
        return 0, out
    monkeypatch.setattr(job, 'train', interrupted)
    monkeypatch.setattr('sys.argv', ['x', str(attempt)])
    assert job.main() == 3
    monkeypatch.setattr(job, 'slurm_state', lambda job_id: 'RUNNING')
    monkeypatch.setenv('SLURM_JOB_ID', '52')
    assert job.main() == 0
    assert json.loads((attempt / 'status.json').read_text())['state'] == 'DUPLICATE_STANDBY'


def test_stop_request_is_job_scoped(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from qwen_vl.train.train_episode import CheckpointRequest
    callback = CheckpointRequest(tmp_path)
    job.request_stop(tmp_path, '77')
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    args, state = SimpleNamespace(device='cpu'), SimpleNamespace(global_step=5)
    monkeypatch.setenv('SLURM_JOB_ID', '78')
    callback.on_step_end(args, state, control)
    assert not control.should_training_stop                          # another job's request
    monkeypatch.setenv('SLURM_JOB_ID', '77')
    callback.on_step_end(args, state, control)
    assert control.should_training_stop and control.should_save
