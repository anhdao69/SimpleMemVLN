"""Single-winner lease, stale handling with scheduler evidence and exact-ID cancellation."""
import json
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
import pytest
from qwen_vl.train.campaign_lease import Lease, append_jsonl, cancel_recorded_sibling


def _claim(args):
    root, attempt = args
    ok, _ = Lease(root, 'P1-V0-s429').claim(attempt, attempt[-1], attempt.split('-')[2])
    return attempt if ok else None


def test_simultaneous_candidates_have_exactly_one_winner(tmp_path):
    attempts = [f'P1-V0-H4-a{i}' for i in range(12)] + [f'P1-V0-H2-a{i}' for i in range(12)]
    with get_context('fork').Pool(24) as pool:
        winners = [w for w in pool.map(_claim, [(str(tmp_path), a) for a in attempts]) if w]
    assert len(winners) == 1
    lease = Lease(tmp_path, 'P1-V0-s429').read()
    assert lease['attempt'] == winners[0] and lease['state'] == 'CLAIMED'
    with pytest.raises(RuntimeError, match='not owned'):
        Lease(tmp_path, 'P1-V0-s429').update(next(a for a in attempts if a != winners[0]), state='ACTIVE')


def _scheduler(states, identity=None, cancelled=None):
    def run(command, **kwargs):
        if command[0] == 'squeue':
            return SimpleNamespace(returncode=0, stdout=states.get(command[3], '') + '\n', stderr='')
        if command[0] == 'sacct':
            return SimpleNamespace(returncode=0, stdout=states.get(command[4], 'COMPLETED') + '\n', stderr='')
        if command[0] == 'scontrol':
            fields = (identity or {}).get(command[3])
            text = ' '.join(f'{k}={v}' for k, v in fields.items()) if fields else ''
            return SimpleNamespace(returncode=0 if fields else 1, stdout=text, stderr='')
        if command[0] == 'scancel':
            cancelled.append(command[1])
            return SimpleNamespace(returncode=0, stdout='', stderr='')
        raise AssertionError(command)
    return run


def test_stale_lease_needs_dead_owner_old_heartbeat_and_no_verification(tmp_path, monkeypatch):
    live = _scheduler({'101': 'RUNNING'})
    lease = Lease(tmp_path, 'run', runner=live)
    assert lease.claim('run-H4-a1', '101', 'H4')[0]
    ok, evidence = lease.takeover_if_stale('run-H2-a1', '202', 'H2')
    assert not ok and not evidence['stale']                       # owner job still running
    dead = Lease(tmp_path, 'run', runner=_scheduler({}))           # squeue empty, sacct COMPLETED
    ok, evidence = dead.takeover_if_stale('run-H2-a1', '202', 'H2', max_age=3600)
    assert not ok and not evidence['stale']                       # heartbeat too recent
    record = json.loads(dead.path.read_text()); record['heartbeat'] -= 7200
    dead.path.write_text(json.dumps(record))
    record['state'] = 'TRAINING_VERIFIED'; dead.path.write_text(json.dumps(record))
    ok, evidence = dead.takeover_if_stale('run-H2-a1', '202', 'H2')
    assert not ok                                                  # verified owners are never taken over
    record['state'] = 'CLAIMED'; dead.path.write_text(json.dumps(record))
    ok, result = dead.takeover_if_stale('run-H2-a1', '202', 'H2')
    assert ok and dead.read()['attempt'] == 'run-H2-a1'
    assert len(list((tmp_path / 'leases/run/history').glob('LEASE.*.json'))) == 1   # archived, not deleted


def test_cancellation_only_for_exact_recorded_live_sibling(tmp_path, monkeypatch):
    ledger = tmp_path / 'submission_ledger.jsonl'
    for attempt, job in (('run-H4-a1', '11'), ('run-H2-a1', '22'), ('run-H2-a2', '33')):
        append_jsonl(ledger, dict(event='submitted', run_id='run', attempt=attempt, job_id=job,
                                  job_name=f'epi-run-{attempt[4:6]}', comment=f'c/run/{attempt}'))
    append_jsonl(ledger, dict(event='submitted', run_id='other', attempt='other-H2-a1', job_id='44',
                              job_name='epi-other-H2', comment='c/other/other-H2-a1'))
    cancelled = []
    identity = {'22': dict(JobName='epi-run-H2', UserId='me(1)', Comment='c/run/run-H2-a1'),
                '33': dict(JobName='epi-run-H2', UserId='me(1)', Comment='tampered')}
    runner = _scheduler({'22': 'PENDING', '33': 'PENDING', '44': 'PENDING'}, identity, cancelled)
    actions = cancel_recorded_sibling(tmp_path, 'run', 'run-H4-a1', runner=runner, user='me')
    assert cancelled == ['22']                                      # never the winner, other runs or mismatches
    assert {a['job_id']: a['action'] for a in actions} == {'22': 'cancelled', '33': 'refused_identity_mismatch'}


def test_attempt_directories_are_never_reused(tmp_path):
    from scripts.vln.profile_episodic import run_gate
    (tmp_path / 'H4' / 'v0-smoke').mkdir(parents=True)
    with pytest.raises(ValueError, match='Refusing to reuse'):
        run_gate('H4', 'v0', tmp_path / 'smoke.jsonl', tmp_path, 'model', 3)


def test_two_contenders_cannot_both_take_over_one_stale_lease(tmp_path):
    dead = _scheduler({})
    first = Lease(tmp_path, 'run', runner=dead)
    assert first.claim('run-H4-a1', '101', 'H4')[0]
    record = json.loads(first.path.read_text()); record['state'] = 'FAILED'
    first.path.write_text(json.dumps(record))
    a, b = Lease(tmp_path, 'run', runner=dead), Lease(tmp_path, 'run', runner=dead)
    evidence_b = b.stale_evidence()                                  # B judges X stale ...
    ok_a, _ = a.takeover_if_stale('run-H2-a1', '202', 'H2')          # ... A takes over first
    assert ok_a and evidence_b['stale']
    b.stale_evidence = lambda max_age=900: evidence_b                # B acts on outdated evidence
    ok_b, _ = b.takeover_if_stale('run-H4-a2', '303', 'H4')
    assert not ok_b and Lease(tmp_path, 'run').read()['attempt'] == 'run-H2-a1'
    assert len(list((tmp_path / 'leases/run/history').glob('LEASE.*.json'))) == 1


def test_ever_verified_owner_is_never_stale(tmp_path):
    lease = Lease(tmp_path, 'run', runner=_scheduler({}))
    lease.claim('run-H4-a1', '101', 'H4')
    lease.update('run-H4-a1', state='FAILED', ever_verified=True)
    assert not lease.stale_evidence()['stale']
