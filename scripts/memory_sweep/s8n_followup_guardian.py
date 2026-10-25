"""Unattended launcher and guardian for the S8N follow-up queue (login node, CPU only, standard library).

Every two minutes, for each follow-up lane:
- while its L-campaign predecessor step is alive or has not reached ``lane-complete``, do nothing, so the
  L guardian can still relaunch an interrupted L controller into an allocation this queue has not entered;
- if a step named ``s8n-lane-<L>`` is alive in the lane's allocation, do nothing;
- otherwise, with an arm of the lane unfinished, launch the lane controller
  (``srun --overlap ... lanes/run_lane_<L>.sh``).  The first launch follows the predecessor at once;
  relaunches wait out a five-minute grace period, are at least 15 minutes apart and capped at three.

If a predecessor step has been absent for six hours without reaching ``lane-complete`` (its own guardian
has given up), the lane's ``OVERRIDE_PREDECESSOR`` control is created and the lane is launched.

Controllers are identity-bound and resume from their last checkpoint, so a relaunch repeats nothing that
finished.  Before a relaunch, interrupted smoke and qualification directories (no recorded result) are
renamed aside, never deleted, because the controller refuses to reuse them.  A lane is never relaunched
after ``stopped-by-request`` or with ``STOP_AFTER_CURRENT`` present.  The publication watcher, once
launched, is restarted at most three times if it dies before finishing.  The guardian never touches a
running step or process and never deletes anything.  State is in ``<root>/guardian/status.json``, the log
in ``guardian.log``; create ``<root>/guardian/STOP`` to end it.
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
ROOT = BASE / 'm5-s8n-followup-20261025'
OPS = ROOT / 'guardian'
PYTHON = BASE / 'SimpleMemVLN/.venv/bin/python'
WATCHERS = {'M5_S_followup': (BASE / 'm5-s8n-followup-publication-20261025', 'watch_publish_followup.py')}
POLL, GRACE, SPACING, MAX_RELAUNCH, ABANDONED = 120, 300, 900, 3, 6 * 3600
FINAL = 'lane-complete'


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


def steps(job):
    """(step id, name) pairs of the allocation, or None when squeue itself fails."""
    try:
        out = subprocess.run(['squeue', '--steps', '-h', '-j', job, '-o', '%i %j'], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [tuple(line.split(None, 1)) for line in out.stdout.splitlines() if ' ' in line]


def unfinished_arms(arms, rows):
    """Arms with neither an ``arm-complete`` record nor a complete/skipped lane-complete outcome."""
    done = {r.get('arm') for r in rows if r.get('status') == 'arm-complete'}
    for r in rows:
        if r.get('status') == 'lane-complete':
            done |= {a for a, outcome in (r.get('outcomes') or {}).items() if outcome in ('complete', 'skipped')}
    return [a for a in arms if a not in done]


def set_aside(path, moved):
    target = path.with_name(f'{path.name}.interrupted-{time.strftime("%Y%m%dT%H%M%S", time.gmtime())}')
    path.rename(target)
    moved.append(str(target))


def set_aside_interrupted(arms):
    """Rename (never delete) smoke and gate directories an interrupted controller left without a result."""
    moved = []
    for arm in arms:
        for smoke in sorted((ROOT / 'runs' / f'{arm}_seed429_e1').glob('smoke-omp[0-9]')):
            if smoke.is_dir() and not (smoke / 'result.json').exists():
                set_aside(smoke, moved)
    shared = ROOT / 'shared'
    gates = read_json(shared / 'gates.json', {})
    if not (shared / 'QUALIFIED.json').exists():
        if not gates.get('zero-muon-ds', {}).get('passed') and (shared / 'zero_muon_ds').is_dir():
            set_aside(shared / 'zero_muon_ds', moved)
        if not gates.get('reproduction', {}).get('passed'):
            for run in sorted((shared / 'reproduction').glob('S8N_*')):
                if run.is_dir() and '.interrupted-' not in run.name and not (run / 'result.json').exists():
                    set_aside(run, moved)
        if not gates.get('recovery', {}).get('passed') and (shared / 'recovery').is_dir():
            set_aside(shared / 'recovery', moved)
    return moved


def launch_lane(lane, job, dry_run):
    command = ['srun', f'--jobid={job}', '--overlap', '--gpus=4', '--nodelist=worker-2',
               f'--job-name=s8n-lane-{lane}', str(ROOT / 'lanes' / f'run_lane_{lane}.sh')]
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
    if host is None:
        record['action'] = 'not launched yet'
        return
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


def predecessor_state(lane, spec, record, active, dry_run):
    """'finished', or why the lane must keep waiting for its L-campaign predecessor."""
    predecessor = spec['predecessor']
    status = read_json(predecessor['state'], {}).get('status')
    alive = predecessor['step'] in {s for s, _ in active}
    record.update(predecessor_status=status, predecessor_alive=alive)
    if alive:
        record['predecessor_absent_since'] = None
        return 'predecessor step alive'
    if status == predecessor.get('final_status', FINAL):
        return 'finished'
    if any(name.strip().startswith('l-lane-') for _, name in active):
        record['predecessor_absent_since'] = None
        return 'an L controller step is alive'
    record['predecessor_absent_since'] = record.get('predecessor_absent_since') or time.time()
    absent = time.time() - record['predecessor_absent_since']
    if absent < ABANDONED:
        return f'predecessor {status} and absent for {absent / 60:.0f} min'
    override = ROOT / 'lanes' / lane / 'OVERRIDE_PREDECESSOR'
    if not override.exists() and not dry_run:
        override.parent.mkdir(parents=True, exist_ok=True)
        override.write_text(f'{now()} guardian: predecessor {predecessor["step"]} absent {absent / 3600:.1f} h '
                            f'with status {status}\n')
        log(f'lane {lane}: predecessor {predecessor["step"]} abandoned ({status}); OVERRIDE_PREDECESSOR created')
    return 'finished'


def check_lane(lane, spec, state, dry_run):
    job, arms = spec['job'], spec['arms']
    record = state['lanes'].setdefault(lane, dict(launches=0, relaunches=0, missing_since=None, last_launch=0))
    rows = history(lane)
    controller = read_json(ROOT / 'lanes' / lane / 'state.json', {})
    unfinished = unfinished_arms(arms, rows)
    record.update(status=controller.get('status'), arm=controller.get('arm'), unfinished=unfinished)
    if not unfinished:
        record.update(action='none (all arms complete)', missing_since=None)
        return
    active = steps(job)
    if active is None:
        record['action'] = 'squeue failed; retry next cycle'
        return
    own = [s for s, name in active if name.strip() == f's8n-lane-{lane}']
    record['steps'] = own
    if own:
        record.update(action='none (controller alive)', missing_since=None)
        return
    if record['launches'] == 0:
        waiting = predecessor_state(lane, spec, record, active, dry_run)
        if waiting != 'finished':
            record['action'] = f'waiting: {waiting}'
            return
        launched = launch_lane(lane, job, dry_run)
        record.update(launches=1, last_launch=time.time(), missing_since=None, action='launched')
        log(f'lane {lane}: predecessor finished ({record.get("predecessor_status")}); launched {launched}')
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
    if time.time() - record['last_launch'] < SPACING:
        record['action'] = 'controller step absent; waiting for relaunch spacing'
        return
    moved = [] if dry_run else set_aside_interrupted(unfinished)
    launched = launch_lane(lane, job, dry_run)
    record['relaunches'] += 1
    record.update(last_launch=time.time(), missing_since=None,
                  action=f'relaunched (relaunch {record["relaunches"]})', moved=moved)
    log(f'lane {lane}: controller step absent with unfinished {unfinished} (last status '
        f'{controller.get("status")}); relaunch {record["relaunches"]} {launched} moved {moved}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--once', action='store_true', help='One cycle, print the state, then exit')
    p.add_argument('--dry-run', action='store_true', help='Decide but never launch, create or rename anything')
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
            log('every lane arm is complete and the watcher has finished; guardian finished')
            return
        if (OPS / 'STOP').exists():
            log('STOP present; guardian ended')
            return
        time.sleep(POLL)


if __name__ == '__main__':
    main()
