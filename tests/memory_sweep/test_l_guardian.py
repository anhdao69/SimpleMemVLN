import json
import os
import subprocess

import pytest

from scripts.memory_sweep import l_guardian as g

SPEC = dict(job='5248', arms=['L2', 'L0'])


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(g, 'ROOT', tmp_path)
    monkeypatch.setattr(g, 'OPS', tmp_path / 'guardian')
    (tmp_path / 'guardian').mkdir()
    (tmp_path / 'lanes/A').mkdir(parents=True)
    return tmp_path


def write_history(root, rows, corrupt=False):
    lines = [json.dumps(r) for r in rows] + (['{"status": "trunc'] if corrupt else [])
    (root / 'lanes/A/history.jsonl').write_text('\n'.join(lines) + '\n')
    (root / 'lanes/A/state.json').write_text(json.dumps(rows[-1] if rows else {}))


def fake_squeue(monkeypatch, stdout='', returncode=0, raises=None):
    def run(command, **kwargs):
        if raises:
            raise raises
        assert command[:5] == ['squeue', '--steps', '-h', '-j', '5248']
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr='')
    monkeypatch.setattr(g.subprocess, 'run', run)


def fake_launch(monkeypatch):
    launches = []
    monkeypatch.setattr(g, 'relaunch_lane', lambda lane, job, dry_run: launches.append((lane, job)) or {})
    return launches


def test_unfinished_and_undeclared_arms():
    rows = [dict(status='arm-complete', arm='L2'), dict(status='lane-complete', outcomes={'L2': 'complete', 'L0': 'failed'})]
    assert g.unfinished_arms(['L2', 'L0'], rows) == ['L0']
    rows.append(dict(status='lane-complete', outcomes={'L2': 'complete', 'L0': 'complete'}))
    assert g.unfinished_arms(['L2', 'L0'], rows) == []
    assert g.undeclared_arms(['L2', 'L0'], rows) == ['L0']        # complete, but the watcher needs a record
    assert g.unfinished_arms(['L2', 'L0'], [dict(status='lane-complete', outcomes={'L0': 'skipped'})]) == ['L2']


def test_corrupt_history_line_is_skipped(root):
    write_history(root, [dict(status='arm-complete', arm='L2')], corrupt=True)
    assert [r['arm'] for r in g.history('A')] == ['L2']


def test_alive_controller_and_squeue_failures_never_relaunch(root, monkeypatch):
    write_history(root, [dict(status='waiting-predecessor', step='5248.28')])
    launches = fake_launch(monkeypatch)
    state = dict(lanes={})
    fake_squeue(monkeypatch, '5248.0 train_vln\n5248.27 r4b0-laneA\n5248.28 l-lane-A\n')
    g.check_lane('A', SPEC, state, False)
    assert state['lanes']['A']['action'] == 'none (controller alive)' and state['lanes']['A']['steps'] == ['5248.28']
    for kwargs in (dict(returncode=1), dict(raises=subprocess.TimeoutExpired('squeue', 60)), dict(raises=OSError())):
        fake_squeue(monkeypatch, **kwargs)
        g.check_lane('A', SPEC, state, False)
        assert state['lanes']['A']['action'] == 'squeue failed; retry next cycle'
    assert launches == []


def test_absent_controller_relaunches_after_grace_with_cap_and_spacing(root, monkeypatch):
    write_history(root, [dict(status='blocked', step='5248.28', error='CalledProcessError')])
    launches = fake_launch(monkeypatch)
    fake_squeue(monkeypatch, '5248.0 train_vln\n')
    clock = [1000.0]
    monkeypatch.setattr(g.time, 'time', lambda: clock[0])
    state = dict(lanes={})
    g.check_lane('A', SPEC, state, False)
    assert 'grace' in state['lanes']['A']['action'] and launches == []
    clock[0] += g.GRACE
    g.check_lane('A', SPEC, state, False)
    assert launches == [('A', '5248')] and state['lanes']['A']['relaunches'] == 1
    clock[0] += 1                                         # absent again: a fresh grace period starts
    g.check_lane('A', SPEC, state, False)
    assert 'grace' in state['lanes']['A']['action']
    clock[0] += g.GRACE                                   # grace over, but too soon after the last relaunch
    g.check_lane('A', SPEC, state, False)
    assert len(launches) == 1 and 'spacing' in state['lanes']['A']['action']
    for _ in range(4):
        clock[0] += g.SPACING
        g.check_lane('A', SPEC, state, False)
        clock[0] += g.GRACE
        g.check_lane('A', SPEC, state, False)
    assert len(launches) == g.MAX_RELAUNCH and 'limit' in state['lanes']['A']['action']


def test_no_relaunch_when_complete_or_stopped(root, monkeypatch):
    launches = fake_launch(monkeypatch)
    fake_squeue(monkeypatch, '5248.0 train_vln\n')
    monkeypatch.setattr(g.time, 'time', lambda: 10**9)
    write_history(root, [dict(status='arm-complete', arm='L2'), dict(status='arm-complete', arm='L0'),
                         dict(status='lane-complete', outcomes={'L2': 'complete', 'L0': 'complete'})])
    state = dict(lanes={})
    g.check_lane('A', SPEC, state, False)
    assert state['lanes']['A']['action'] == 'none (all arms complete)'
    write_history(root, [dict(status='stopped-by-request', outcomes={})])
    state = dict(lanes={'A': dict(relaunches=0, missing_since=1.0, last_relaunch=0)})
    g.check_lane('A', SPEC, state, False)
    assert state['lanes']['A']['action'] == 'none (stopped by request)'
    write_history(root, [dict(status='blocked')])
    (root / 'lanes/A/STOP_AFTER_CURRENT').write_text('')
    g.check_lane('A', SPEC, state, False)
    assert state['lanes']['A']['action'] == 'none (stopped by request)' and launches == []


def test_interrupted_smoke_is_renamed_never_deleted(root):
    run = root / 'runs/L2_seed429_e1'
    (run / 'smoke-omp4/train').mkdir(parents=True)
    (run / 'smoke-omp4/train/partial.log').write_text('x')
    (run / 'smoke-omp8').mkdir()
    (run / 'smoke-omp8/result.json').write_text('{}')
    moved = g.set_aside_interrupted_smokes(['L2', 'L0'])
    assert len(moved) == 1 and not (run / 'smoke-omp4').exists() and (run / 'smoke-omp8/result.json').exists()
    assert (root / 'runs/L2_seed429_e1' / os.path.basename(moved[0]) / 'train/partial.log').read_text() == 'x'


def test_watcher_restart_rules(tmp_path, monkeypatch):
    ops = tmp_path / 'ops'
    ops.mkdir()
    monkeypatch.setattr(g, 'WATCHERS', {'M5_L': (ops, 'watch_publish_l.py')})
    monkeypatch.setattr(g, 'OPS', tmp_path)
    started = []
    monkeypatch.setattr(g.subprocess, 'Popen', lambda *a, **k: started.append(a) or type('P', (), {'pid': 1})())
    host = os.uname().nodename
    (ops / 'launch.json').write_text(json.dumps(dict(host='other-login')))
    (ops / 'status.json').write_text(json.dumps(dict(pid=999999999, watcher='running')))
    state = dict(watchers={})
    g.check_watcher('M5_L', state, False)
    assert state['watchers']['M5_L']['action'].startswith('not checked') and not started
    (ops / 'launch.json').write_text(json.dumps(dict(host=host)))
    (ops / 'status.json').write_text(json.dumps(dict(pid=os.getpid(), watcher='running')))
    monkeypatch.setattr(g, 'process_alive', lambda pid, marker: True)
    g.check_watcher('M5_L', state, False)
    assert state['watchers']['M5_L']['action'] == 'none (alive)' and not started
    monkeypatch.setattr(g, 'process_alive', lambda pid, marker: False)
    for _ in range(g.MAX_RELAUNCH + 2):
        g.check_watcher('M5_L', state, False)
    assert len(started) == g.MAX_RELAUNCH and state['watchers']['M5_L']['action'] == 'dead; restart limit reached'
    (ops / 'status.json').write_text(json.dumps(dict(pid=1, watcher='finished')))
    g.check_watcher('M5_L', state, False)
    assert state['watchers']['M5_L']['action'] == 'none (finished)'


@pytest.mark.skipif(not os.path.exists('/proc/self/cmdline'), reason='needs Linux /proc')
def test_process_alive_checks_command_line():
    assert g.process_alive(os.getpid(), 'python') or g.process_alive(os.getpid(), 'pytest')
    assert not g.process_alive(os.getpid(), 'watch_publish_l.py')
    assert not g.process_alive(None, 'x') and not g.process_alive(2**30, 'x')
