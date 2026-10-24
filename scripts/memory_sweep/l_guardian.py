"""Unattended guardian for the L-campaign queue (login node, CPU only, standard library).

Every two minutes, for each L lane:
- if a step named ``l-lane-<L>`` is alive in the lane's allocation, do nothing;
- if no such step has existed for five minutes and an arm of the lane is not yet complete, relaunch the
  lane controller with its original command (``srun --overlap ... lanes/run_lane_<L>.sh``).

Controllers are identity-bound and resume from their last checkpoint, so a relaunch repeats nothing that
finished. Before a relaunch, an interrupted smoke directory (no ``result.json``) is renamed aside, never deleted,
because the controller refuses to reuse one. A lane is never relaunched after ``stopped-by-request`` or with
``STOP_AFTER_CURRENT`` present. Relaunches are capped at three per lane, at least 15 minutes apart.

Publication watchers (M5_L, M5_S) whose process has died before finishing are restarted, also at most three
times each. The guardian never touches a running step or process, and never deletes anything. State is in
``<root>/guardian/status.json`` and the log in ``guardian.log``; create ``<root>/guardian/STOP`` to end it.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

BASE = Path('/mnt/data/vmo-ai-task/anhdh35')
ROOT = BASE / 'm5-gdn-l-sweep-20261023'
OPS = ROOT / 'guardian'
PYTHON = BASE / 'SimpleMemVLN/.venv/bin/python'
WATCHERS = {'M5_L': (BASE / 'm5-gdn-l-publication-20261023', 'watch_publish_l.py'),
            'M5_S': (BASE / 'm5-gdn-s-sweep-publication-20261021', 'watch_publish.py')}
POLL, GRACE, SPACING, MAX_RELAUNCH = 120, 300, 900, 3


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def log(message):
    with (OPS / 'guardian.log').open('a') as f:
        f.write(f'{now()} {message}\n')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def history(lane):
    """History rows; a partially written or corrupt line is skipped, never fatal."""
    rows = []
    try:
        text = (ROOT / 'lanes' / lane / 'history.jsonl').read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def lane_steps(job, lane):
    """Step IDs named l-lane-<lane> in the allocation, or None when squeue itself fails."""
    try:
        out = subprocess.run(['squeue', '--steps', '-h', '-j', job, '-o', '%i %j'], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [s for s, name in (line.split(None, 1) for line in out.stdout.splitlines() if ' ' in line)
            if name.strip() == f'l-lane-{lane}']


def unfinished_arms(arms, rows):
    """Arms with neither an ``arm-complete`` record nor a complete/skipped lane-complete outcome."""
    done = {r.get('arm') for r in rows if r.get('status') == 'arm-complete'}
    for r in rows:
        if r.get('status') == 'lane-complete':
            done |= {a for a, outcome in (r.get('outcomes') or {}).items() if outcome in ('complete', 'skipped')}
    return [a for a in arms if a not in done]


def undeclared_arms(arms, rows):
    """Complete per the controller's validation but without the arm-complete record the M5_L watcher reads."""
    declared = {r.get('arm') for r in rows if r.get('status') == 'arm-complete'}
    return [a for a in arms if a not in declared and a not in unfinished_arms(arms, rows)]


def set_aside_interrupted_smokes(arms):
    moved = []
    for arm in arms:
        run = ROOT / 'runs' / f'{arm}_seed429_e1'
        for smoke in sorted(run.glob('smoke-omp[0-9]')):
            if smoke.is_dir() and not (smoke / 'result.json').exists():
                target = smoke.with_name(f'{smoke.name}.interrupted-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}')
                smoke.rename(target)
                moved.append(str(target))
    return moved


def relaunch_lane(lane, job, dry_run):
    command = ['srun', f'--jobid={job}', '--overlap', '--gpus=4', '--nodelist=worker-2',
               f'--job-name=l-lane-{lane}', str(ROOT / 'lanes' / f'run_lane_{lane}.sh')]
    if dry_run:
        return dict(command=command, pid=None)
    out = (ROOT / 'lanes' / f'lane_{lane}.out').open('a')
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                               start_new_session=True)
    return dict(command=command, pid=process.pid)


def process_alive(pid, marker):
    try:
        return marker in Path(f'/proc/{int(pid)}/cmdline').read_bytes().decode(errors='replace')
    except (OSError, TypeError, ValueError):
        return False


def watcher_host(ops):
    """Host of the most recent launch of a watcher (its pid is only meaningful there)."""
    launches = [read_json(p, {}) for p in [ops / 'launch.json', *sorted(ops.glob('launch-guardian-*.json'))]]
    hosts = [l.get('host') for l in launches if l.get('host')]
    return hosts[-1] if hosts else None


def check_watcher(name, state, dry_run):
    ops, script = WATCHERS[name]
    status = read_json(ops / 'status.json', {})
    record = state['watchers'].setdefault(name, dict(restarts=0))
    record.update(pid=status.get('pid'), watcher=status.get('watcher'), utc=status.get('utc'))
    if status.get('watcher') in ('finished', 'stopped-by-request') or (ops / 'STOP').exists():
        record['action'] = 'none (finished)'
        return
    host = watcher_host(ops)
    if host != os.uname().nodename:
        record['action'] = f'not checked (watcher runs on {host})'
        return
    if process_alive(status.get('pid'), script):
        record['action'] = 'none (alive)'
        return
    if record['restarts'] >= MAX_RELAUNCH:
        record['action'] = 'dead; restart limit reached'
        return
    record['restarts'] += 1
    if not dry_run:
        out = (ops / 'watcher.log').open('a')
        process = subprocess.Popen([str(PYTHON), script], cwd=ops, stdin=subprocess.DEVNULL, stdout=out,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        (ops / f'launch-guardian-{record["restarts"]}.json').write_text(json.dumps(dict(
            pid=process.pid, host=os.uname().nodename, utc=now(), command=[str(PYTHON), script])) + '\n')
    record['action'] = f'restarted (restart {record["restarts"]})'
    log(f'{name} watcher pid {status.get("pid")} was dead; restarted (restart {record["restarts"]})')


def check_lane(lane, spec, state, dry_run):
    job, arms = spec['job'], spec['arms']
    record = state['lanes'].setdefault(lane, dict(relaunches=0, missing_since=None, last_relaunch=0))
    rows = history(lane)
    controller = read_json(ROOT / 'lanes' / lane / 'state.json', {})
    unfinished = unfinished_arms(arms, rows)
    record.update(status=controller.get('status'), arm=controller.get('arm'), unfinished=unfinished,
                  undeclared=undeclared_arms(arms, rows))
    if not unfinished:
        record.update(action='none (all arms complete)', missing_since=None)
        return
    steps = lane_steps(job, lane)
    if steps is None:
        record['action'] = 'squeue failed; retry next cycle'
        return
    record['steps'] = steps
    if steps:
        record.update(action='none (controller alive)', missing_since=None)
        return
    if controller.get('status') == 'stopped-by-request' or (ROOT / 'lanes' / lane / 'STOP_AFTER_CURRENT').exists():
        record['action'] = 'none (stopped by request)'
        return
    record['missing_since'] = record['missing_since'] or time.time()
    if time.time() - record['missing_since'] < GRACE:
        record['action'] = 'controller step absent; waiting out the grace period'
        return
    if record['relaunches'] >= MAX_RELAUNCH:
        record['action'] = 'controller step absent; relaunch limit reached'
        return
    if time.time() - record['last_relaunch'] < SPACING:
        record['action'] = 'controller step absent; waiting for relaunch spacing'
        return
    moved = [] if dry_run else set_aside_interrupted_smokes(unfinished)
    launched = relaunch_lane(lane, job, dry_run)
    record['relaunches'] += 1
    record.update(last_relaunch=time.time(), missing_since=None,
                  action=f'relaunched (relaunch {record["relaunches"]})', moved_smokes=moved)
    log(f'lane {lane}: controller step absent with unfinished {unfinished} (last status '
        f'{controller.get("status")}); relaunch {record["relaunches"]} {launched} moved {moved}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--once', action='store_true', help='One cycle, print the state, then exit')
    p.add_argument('--dry-run', action='store_true', help='Decide but never launch or rename anything')
    args = p.parse_args()
    OPS.mkdir(exist_ok=True)
    lock = (OPS / 'guardian.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lanes = read_json(ROOT / 'campaign.json')['lanes']
    state = read_json(OPS / 'status.json', {}) if not args.dry_run else {}
    state.update(pid=os.getpid(), host=os.uname().nodename, lanes=state.get('lanes', {}),
                 watchers=state.get('watchers', {}))
    log(f'guardian start pid {os.getpid()} on {os.uname().nodename} dry_run={args.dry_run}')
    while True:
        for lane, spec in sorted(lanes.items()):
            try:
                check_lane(lane, spec, state, args.dry_run)
            except Exception as exc:                      # never let one lane's check end the guardian
                state['lanes'].setdefault(lane, {})['action'] = f'check error: {exc!r}'[:300]
                log(f'lane {lane} check error {exc!r}')
        for name in WATCHERS:
            try:
                check_watcher(name, state, args.dry_run)
            except Exception as exc:
                state['watchers'].setdefault(name, {})['action'] = f'check error: {exc!r}'[:300]
                log(f'watcher {name} check error {exc!r}')
        state['utc'] = now()
        trained = all(not v.get('unfinished') for v in state['lanes'].values()) and len(state['lanes']) == len(lanes)
        published = all(w.get('action') in ('none (finished)', 'dead; restart limit reached')
                        for w in state['watchers'].values())
        done = trained and published
        state['guardian'] = 'finished' if done else 'running'
        if args.once:
            print(json.dumps(state, indent=1, default=str))
            return
        temporary = OPS / 'status.tmp'
        temporary.write_text(json.dumps(state, indent=1, default=str) + '\n')
        temporary.replace(OPS / 'status.json')
        if done:
            log('every lane arm is complete and both watchers have finished; guardian finished')
            return
        if (OPS / 'STOP').exists():
            log('STOP present; guardian ended')
            return
        time.sleep(POLL)


if __name__ == '__main__':
    main()
