"""Login-node watcher of the S8N R/B training ablation (one instance; stdlib only, no torch).

Every minute it lists the user's RUNNING four-GPU Slurm allocations and assigns each idle one a task that
``rb_ablation`` runs as an ``srun --overlap`` step: ``qualify`` until the campaign is sealed (one allocation
at a time), then arms (a resumable arm first, the most advanced first; then fresh arms in the user's order).
Lustre is mounted ``localflock``, so this single process makes every assignment.

An allocation is idle when no LoopWAM RoboTwin trainer runs or is assigned there and no LoopWAM evaluation
agent is alive there (their own status files), and no rbabl step runs there; the lane itself re-checks that
its GPUs have no compute process.  Work already running is never interrupted.  Before launching on an
allocation it marks that job ``agent_<job>.EXIT`` for the LoopWAM eval dispatcher (which otherwise starts an
eval agent with a GPU T5 server in every running job without training) and removes its ``.EXIT.auto``;
every marker it creates is recorded and removed when the campaign completes.

Controls (files in <root>/watcher): ``STOP`` ends the watcher (running lanes continue); ``PAUSE`` stops new
launches; ``SKIP_<ARM>`` never assigns that arm; ``EXCLUDE_<JOB>`` never uses that allocation.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.memory_sweep.rb_ablation import ARMS, EXPECTED_STEPS, VERSION

LOOPWAM = Path('/groups/yshang/an221229/checkpoints/LoopWAM_NT')
LOOPWAM_TRAIN = (LOOPWAM / 'robotwin10_work/control', LOOPWAM / 'robotwin10_e10_work/control')
LOOPWAM_EVAL = LOOPWAM / 'robotwin10_eval/control'
STEP_PREFIX = 'rbabl'


def now():
    return time.time()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def parse_end(text):
    try:
        return time.mktime(time.strptime(text.strip(), '%Y-%m-%dT%H:%M:%S'))
    except ValueError:
        return None


def running_jobs():
    out = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER', 'an221229'), '-t', 'RUNNING',
                          '--Format=JobID:20,NodeList:20,EndTime:24,tres-alloc:120,NumCPUs:8'],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    if out.returncode:
        raise RuntimeError(f'squeue failed: {out.stderr[:300]}')
    jobs = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or 'gres/gpu=4' not in parts[3]:
            continue
        jobs[parts[0]] = dict(node=parts[1], end=parse_end(parts[2]), cpus=int(parts[4]))
    return jobs


def job_steps(job):
    out = subprocess.run(['squeue', '-h', '--steps', '-j', job, '-o', '%i|%j'], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, universal_newlines=True, timeout=120)
    if out.returncode:
        raise RuntimeError(f'squeue --steps failed: {out.stderr[:300]}')
    return [tuple(line.split('|', 1)) for line in out.stdout.splitlines() if '|' in line]


def loopwam_busy(job):
    """Reasons LoopWAM work runs or is about to run on ``job`` (its own fresh status files)."""
    reasons = []
    for control in LOOPWAM_TRAIN:
        status = read_json(control / f'agent_{job}.status.json', {}) or {}
        fresh = now() - status.get('heartbeat', 0) < 300 and status.get('state') != 'exited'
        runs = list((status.get('runs') or {}).keys())
        assigned = list(((read_json(control / f'agent_{job}.json', {}) or {}).get('assign') or {}).keys())
        if fresh and (runs or assigned):
            reasons.append(f'LoopWAM training {sorted(set(runs + assigned))}')
    status = read_json(LOOPWAM_EVAL / f'agent_{job}.status.json', {}) or {}
    if now() - status.get('heartbeat', 0) < 300 and status.get('state') != 'exited':
        reasons.append(f'LoopWAM eval agent running={sorted((status.get("running") or {}).keys())[:3]}')
    return reasons


def latest_step(run):
    best = 0
    for marker in Path(run, 'train').glob('checkpoint-*/RECOVERY_COMPLETE.json'):
        record = read_json(marker, {}) or {}
        best = max(best, int(record.get('step', 0)))
    return best


def last_profile(run):
    path = Path(run, 'train', 'profile_rank0.jsonl')
    try:
        with path.open('rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 8192))
            lines = f.read().decode(errors='ignore').splitlines()
        return json.loads(lines[-1])
    except (OSError, ValueError, IndexError):
        return None


class Watcher:
    def __init__(self, campaign_path):
        self.campaign_path = Path(campaign_path)
        self.c = json.loads(self.campaign_path.read_text())
        if self.c['version'] != VERSION:
            raise ValueError('Unknown campaign version')
        self.root = Path(self.c['root'])
        self.dir = self.root / 'watcher'
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state = read_json(self.dir / 'state.json', None) or dict(launches=[], backoff={}, errors={},
                                                                     exit_markers=[])

    def log(self, msg):
        line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        with (self.dir / 'watcher.log').open('a') as f:
            f.write(line + '\n')

    def save(self):
        write_json(self.dir / 'state.json', self.state)

    def control(self, name):
        return (self.dir / name).exists()

    def run_dir(self, arm):
        return self.root / 'runs' / f'{arm}_seed{self.c["seed"]}_e1'

    def arm_complete(self, arm):
        audit = read_json(self.run_dir(arm) / 'completion_audit.json', {}) or {}
        return bool(audit.get('passed'))

    def sealed(self):
        seal = read_json(self.root / 'shared/QUALIFIED.json', {}) or {}
        return bool(seal.get('passed')) and seal.get('source_commit') == self.c['source_commit']

    def lane_state(self, job):
        return read_json(self.root / 'lanes' / job / 'state.json', {}) or {}

    # -- LoopWAM eval dispatcher markers --
    def hold_eval(self, job):
        exit_path, auto = LOOPWAM_EVAL / f'agent_{job}.EXIT', LOOPWAM_EVAL / f'agent_{job}.EXIT.auto'
        if not LOOPWAM_EVAL.is_dir():
            return
        if not exit_path.exists():
            exit_path.touch()
            self.state['exit_markers'].append(str(exit_path))
            self.log(f'marked {exit_path.name} for the LoopWAM eval dispatcher')
        if auto.exists():
            auto.unlink()
            self.state.setdefault('removed_auto', []).append(str(auto))
            self.log(f'removed {auto.name} so the eval dispatcher keeps job {job} for training')
        self.save()

    def release_eval(self):
        # EXIT files this watcher created, and the dispatcher's own EXIT where the watcher removed its .auto
        # partner (the dispatcher would have lifted it once LoopWAM training left the job)
        auto_exits = [p[:-len('.auto')] for p in self.state.get('removed_auto', [])]
        for path in [*self.state.get('exit_markers', []), *auto_exits]:
            try:
                Path(path).unlink()
                self.log(f'removed {Path(path).name}')
            except FileNotFoundError:
                pass
        self.state['exit_markers'] = []
        self.state['removed_auto'] = []
        self.save()

    # -- assignment --
    def choose(self, left, busy_tasks):
        if not self.sealed():
            if ('qualify' in busy_tasks or left < self.c['min_qualify_seconds']
                    or self.state['errors'].get('qualify', 0) >= 2):
                return None
            return 'qualify'
        candidates = [a for a in ARMS if a not in busy_tasks and not self.arm_complete(a)
                      and not self.control(f'SKIP_{a}') and self.state['errors'].get(a, 0) < 3]
        resumable = sorted((a for a in candidates if latest_step(self.run_dir(a)) > 0),
                           key=lambda a: -latest_step(self.run_dir(a)))
        fresh = [a for a in candidates if latest_step(self.run_dir(a)) == 0]
        if resumable and left >= self.c['min_resume_seconds']:
            return resumable[0]
        if fresh and left >= self.c['min_fresh_seconds']:
            return fresh[0]
        return None

    def launch(self, job, info, task):
        lanes = self.root / 'lanes' / job
        lanes.mkdir(parents=True, exist_ok=True)
        source = self.c['source']
        command = ['srun', f'--jobid={job}', '--overlap', '-n1', '--gres=gpu:4', f'--cpus-per-task={info["cpus"]}',
                   '--mem=0', '--cpu-bind=none', f'--job-name={STEP_PREFIX}-{task}',
                   'env', f'RB_SOURCE={source}', 'bash', f'{source}/train/rb_newton_env.sh',
                   'python', '-u', '-m', 'scripts.memory_sweep.rb_ablation', '--campaign', str(self.campaign_path),
                   '--job', job, '--task', task]
        log = (lanes / f'srun-{time.strftime("%Y%m%dT%H%M%S")}-{task}.log').open('a')
        log.write(f'# {time.ctime()} {" ".join(command)}\n')
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
        self.state['launches'].append(dict(job=job, task=task, pid=process.pid, unix=now(), node=info['node'],
                                           left_hours=round((info['end'] - now()) / 3600, 2)))
        self.state.setdefault('active', {})[job] = dict(task=task, pid=process.pid, unix=now())
        self.save()
        self.log(f'launched {task} in job {job} on {info["node"]} '
                 f'({(info["end"] - now()) / 3600:.1f} h left, srun pid {process.pid})')

    def settle(self, job, steps):
        """Account for a lane whose step has ended."""
        active = self.state.get('active', {}).get(job)
        if not active or any(name.startswith(STEP_PREFIX) for _, name in steps):
            return
        if now() - active['unix'] < 180:            # step not visible yet
            return
        state = self.lane_state(job)
        task, status = active['task'], state.get('status')
        if state.get('unix', 0) < active['unix']:
            status = 'no-lane-record'
        outcome = state.get('outcome')
        self.log(f'lane {task} on job {job} ended: {status} {outcome or state.get("error", "")[:300]}')
        if status == 'task-finished' and outcome == 'busy' or status == 'busy':
            self.state['backoff'][job] = now() + 600
        elif status != 'task-finished' or outcome in ('failed',):
            self.state['errors'][task] = self.state['errors'].get(task, 0) + 1
            self.state['backoff'][job] = now() + 300
        del self.state['active'][job]
        self.save()

    def status_report(self, jobs, notes):
        lines = [f'S8N R/B ablation watcher {time.strftime("%m-%d %H:%M")}  sealed={self.sealed()}', '', 'Allocations:']
        for job, info in sorted(jobs.items()):
            active = self.state.get('active', {}).get(job, {})
            lines.append(f'  {job} {info["node"]:7s} {(info["end"] - now()) / 3600:5.1f} h left  '
                         f'task={active.get("task", "-"):12s} {notes.get(job, "")}')
        lines += ['', 'Arms:']
        for arm in ARMS:
            run = self.run_dir(arm)
            last = last_profile(run) or {}
            update = last.get('update', 0)
            if self.arm_complete(arm):
                state = 'COMPLETE'
            elif self.control(f'SKIP_{arm}'):
                state = 'skipped'
            elif self.state['errors'].get(arm, 0) >= 3:
                state = 'BLOCKED (3 lane errors)'
            else:
                state = 'running' if arm in [a['task'] for a in self.state.get('active', {}).values()] else 'waiting'
            lines.append(f'  {arm:13s} {update:5d}/{EXPECTED_STEPS} ckpt {latest_step(run):5d}  {state}')
        (self.dir / 'STATUS.txt').write_text('\n'.join(lines) + '\n')

    def loop(self):
        jobs = running_jobs()
        jobs = {j: i for j, i in jobs.items() if j not in self.c.get('exclude_jobs', [])
                and not self.control(f'EXCLUDE_{j}')}
        notes = {}
        busy_tasks = set()
        steps_by_job = {}
        for job in jobs:
            steps_by_job[job] = job_steps(job)
            self.settle(job, steps_by_job[job])
            if job in self.state.get('active', {}):
                busy_tasks.add(self.state['active'][job]['task'])
        for job in list(self.state.get('active', {})):
            if job not in jobs:                      # allocation ended
                self.log(f'job {job} is gone; its lane task {self.state["active"][job]["task"]} is released')
                del self.state['active'][job]
                self.save()
        if all(self.arm_complete(a) or self.control(f'SKIP_{a}') for a in ARMS):
            return 'done'
        for job, info in sorted(jobs.items(), key=lambda kv: -(kv[1]['end'] or 0)):
            if job in self.state.get('active', {}):
                notes[job] = 'lane running'
                continue
            if any(name.startswith(STEP_PREFIX) for _, name in steps_by_job[job]):
                notes[job] = 'unknown rbabl step present'
                continue
            if self.state['backoff'].get(job, 0) > now():
                notes[job] = 'backoff'
                continue
            reasons = loopwam_busy(job)
            if reasons:
                notes[job] = '; '.join(reasons)[:120]
                continue
            if info['end'] is None:
                continue
            task = None if self.control('PAUSE') else self.choose(info['end'] - now(), busy_tasks)
            if task is None:
                notes[job] = 'idle (nothing to assign / too little time / paused)'
                continue
            self.hold_eval(job)
            self.launch(job, info, task)
            busy_tasks.add(task)
            notes[job] = f'launched {task}'
        self.status_report(jobs, notes)
        return 'running'

    def main(self):
        pid_path = self.dir / 'watcher.pid.json'
        old = read_json(pid_path, {}) or {}
        if old.get('host') == os.uname().nodename and Path(f'/proc/{old.get("pid")}').exists():
            cmd = Path(f'/proc/{old["pid"]}/cmdline').read_bytes().decode(errors='ignore')
            if 'rb_watcher' in cmd and old['pid'] != os.getpid():
                raise SystemExit(f'watcher already running: {old}')
        write_json(pid_path, dict(pid=os.getpid(), host=os.uname().nodename, unix=now()))
        self.log(f'watcher start pid {os.getpid()} on {os.uname().nodename}')
        while not self.control('STOP'):
            try:
                outcome = self.loop()
                if outcome == 'done':
                    self.log('every arm is complete (or skipped); releasing eval markers and exiting')
                    self.release_eval()
                    break
            except Exception as exc:                 # keep watching
                self.log(f'error: {type(exc).__name__}: {exc}')
            write_json(self.dir / 'heartbeat.json', dict(unix=now(), pid=os.getpid()))
            time.sleep(60)
        self.log('watcher exit')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--once', action='store_true', help='one dry status pass without launching anything')
    p.add_argument('--hold-eval', nargs='*', default=[], help='mark these jobs EXIT for the LoopWAM eval dispatcher')
    args = p.parse_args()
    watcher = Watcher(args.campaign)
    for job in args.hold_eval:
        watcher.hold_eval(job)
    if args.hold_eval:
        return
    if args.once:
        jobs = running_jobs()
        for job in jobs:
            print(job, jobs[job], job_steps(job), loopwam_busy(job))
        print('sealed', watcher.sealed(), {a: (latest_step(watcher.run_dir(a)), watcher.arm_complete(a)) for a in ARMS})
        return
    watcher.main()


if __name__ == '__main__':
    main()
