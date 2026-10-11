"""One Slurm candidate attempt of one logical episodic run (runs inside the job).

SUBMITTED -> ALLOCATED -> PREFLIGHT_PASSED -> CLAIMED -> TRAINING_VERIFIED -> ACTIVE -> COMPLETE

1. Preflight on the actual allocation: Slurm identity, GPUs, source/manifest/
   model identity, cgroup RAM limit, disk reservation.
2. Early standby check: an existing live/verified lease means this candidate
   must not train (DUPLICATE_STANDBY).
3. Qualification on this allocation (never reusing a previous PASS):
   ``full`` = profile stress suite (V0 + M2); ``startup`` = longest-episode gate
   for this run's own arm.  Any FAIL stops before the lease.
4. Atomic lease claim (exactly one writer per logical run).
5. Fresh training from the pretrained base under the owned supervisor
   (host guard, OOM counters), recovery saves at 50 and every 250 updates,
   stop at the epoch-1 boundary of the declared two-epoch schedule.  An
   agent thread heartbeats the lease, verifies the winner (>=50 healthy
   updates on every rank + complete hashed recovery checkpoint + resource
   limits), then cancels only the recorded redundant sibling and submits the
   matched ablations on this same profile.
6. Upload epoch-1 export + complete training state to the Hub folder <run>/<attempt>/.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from qwen_vl.train.campaign_lease import (  # noqa: E402
    LIVE_SLURM, VERIFIED, Lease, append_jsonl, atomic_write_json, cancel_recorded_sibling, slurm_state)
from qwen_vl.train.host_memory import cgroup_sample  # noqa: E402


class Attempt:
    def __init__(self, attempt_dir):
        self.dir = Path(attempt_dir)
        self.launch = json.loads((self.dir / 'launch_manifest.json').read_text())
        self.campaign = Path(self.launch['campaign_root'])
        self.manifest = json.loads((self.campaign / 'campaign_manifest.json').read_text())
        self.job = os.environ.get('SLURM_JOB_ID', 'local')
        self.lease = Lease(self.campaign, self.launch['run_id'])

    def status(self, state, **fields):
        payload = dict(state=state, job_id=self.job, attempt=self.launch['attempt'], time=time.time(), **fields)
        atomic_write_json(self.dir / 'status.json', payload)
        append_jsonl(self.dir / 'events.jsonl', payload)
        append_jsonl(self.campaign / 'submission_ledger.jsonl', dict(
            event='state', run_id=self.launch['run_id'], **payload))
        print(f'STATE {state} {json.dumps(fields)}', flush=True)


def preflight(a):
    import hashlib
    import torch
    spec = a.launch['resources']
    problems = []
    if a.job == 'local' or not os.environ.get('SLURM_JOB_ID'):
        problems.append('must run inside a Slurm allocation')
    if torch.cuda.device_count() != spec['world']:
        problems.append(f'{torch.cuda.device_count()} visible GPUs != {spec["world"]}')
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 2**30
        if 'H100' not in name or total < 79:
            problems.append(f'GPU {i}: {name} {total:.1f} GiB')
        free, _ = torch.cuda.mem_get_info(i)
        if free / 2**30 < 75:
            problems.append(f'GPU {i} not idle: {free / 2**30:.1f} GiB free')
    saved = json.loads((a.campaign / 'source_hashes.json').read_text())
    source = Path(a.launch['source_dir'])
    for name, digest in saved['files'].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            problems.append(f'source changed: {name}')
            break
    if Path(__file__).resolve().parents[2] != source.resolve():
        problems.append('job is not executing the immutable campaign source')
    from scripts.vln.submit_episodic import sha256_file
    if sha256_file(a.manifest['manifest']) != a.manifest['manifest_sha256']:
        problems.append('manifest hash changed')
    model = Path(a.manifest['model_path'])
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'preprocessor_config.json'):
        if not (model / name).exists():
            problems.append(f'model file missing {name}')
    if a.manifest['model_revision'] not in str(model):
        problems.append('model path is not the pinned revision snapshot')
    memory = cgroup_sample()
    requested = float(spec['mem'].rstrip('G'))
    if memory['job_limit_gib'] < 0.98 * requested:
        problems.append(f'cgroup limit {memory["job_limit_gib"]:.1f} < requested {requested}')
    free_disk = shutil.disk_usage(a.dir).free / 2**30
    if free_disk < a.manifest['disk_reservation_gib']:
        problems.append(f'free disk {free_disk:.0f} GiB < {a.manifest["disk_reservation_gib"]}')
    record = dict(problems=problems, host=os.uname().nodename, gpus=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  cgroup=memory, free_disk_gib=free_disk)
    atomic_write_json(a.dir / 'preflight.json', record)
    return record


def standby_reason(a):
    current = a.lease.read()
    if current is None:
        return None
    if current.get('attempt') == a.launch['attempt']:
        return None
    evidence = a.lease.stale_evidence()
    if evidence and evidence['stale']:
        return None
    return f'lease held by {current.get("attempt")} ({current.get("state")}, job {current.get("job_id")})'


def qualify(a, mode):
    fixtures = a.manifest['fixtures']
    out = a.dir / f'qualification-{a.job}'          # never reuse a previous allocation's PASS
    profile, arm = a.launch['profile'], a.launch['arm']
    if mode == 'full':
        gates = ['smoke', 'longest_r2r', 'below_offload', 'above_offload', 'longest_rxr', 'mixed_tail',
                 'checkpoint', 'speed']
        arms = ['v0', 'm2']
    else:
        gates, arms = ['longest_rxr'], [arm]
    command = [sys.executable, '-m', 'scripts.vln.profile_episodic', 'suite', '--profile', profile,
               '--fixtures', fixtures, '--out', str(out), '--model-path', a.manifest['model_path'],
               '--gates', *gates, '--arms', *arms, '--speed-manifest', a.manifest['manifest']]
    result = subprocess.run(command, cwd=ROOT)
    report_path = out / f'{profile}_suite.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(passed=False)
    report['returncode'] = result.returncode
    return report


def healthy_updates(train, world, needed):
    """Every rank logged >= needed contiguous finite updates with correct exposure counts."""
    counts = []
    for rank in range(world):
        rows = []
        for name in (f'profile_rank{rank}.jsonl', f'exposures_rank{rank}.jsonl'):
            path = train / name
            rows.append([json.loads(x) for x in path.read_text().splitlines()] if path.exists() else [])
        profile, exposures = rows
        updates = sorted({r['update'] for r in profile})
        if updates[:needed] != list(range(1, needed + 1)):
            return False, f'rank {rank} has updates {updates[:5]}..{len(updates)}'
        if any(r['peak_reserved_gib'] > 78 or r.get('cuda_ooms', 0) for r in profile[:needed]):
            return False, f'rank {rank} exceeded GPU limit'
        early = [e for e in exposures if e['update'] < needed]
        if len(early) < needed * a_gas(train) or not all(math.isfinite(e['loss_sum']) for e in early):
            return False, f'rank {rank} exposures incomplete/nonfinite'
        counts.append(len(updates))
    return True, min(counts)


def a_gas(train):
    cfg = json.loads((train / 'resolved_config.json').read_text())
    return cfg['training']['gradient_accumulation_steps']


def verify_checkpoint(checkpoint):
    import hashlib
    from qwen_vl.train.recovery import validate_complete
    record = validate_complete(checkpoint)
    hashes = {}
    for name in record['files']:
        digest = hashlib.sha256()
        with open(checkpoint / name, 'rb') as handle:
            for block in iter(lambda: handle.read(1 << 24), b''):
                digest.update(block)
        hashes[name] = digest.hexdigest()
    atomic_write_json(checkpoint.parent / f'{checkpoint.name}.sha256.json', hashes)
    return record, hashes


class Agent(threading.Thread):
    def __init__(self, attempt, train, supervisor):
        super().__init__(daemon=True)
        self.a, self.train, self.supervisor = attempt, train, supervisor
        self.stop = threading.Event()
        self.verified = (attempt.lease.read() or {}).get('state') in VERIFIED

    def run(self):
        while not self.stop.wait(60):
            try:
                self.step()
            except Exception as exc:  # the agent must never kill training
                append_jsonl(self.a.dir / 'events.jsonl', dict(event='agent_error', error=repr(exc)[:500]))

    def step(self):
        a = self.a
        a.lease.update(a.launch['attempt'], heartbeat_job=a.job)
        if self.verified:
            return
        world = a.launch['resources']['world']
        ok, detail = healthy_updates(self.train, world, a.manifest['verify_updates'])
        checkpoint = self.train / f'checkpoint-{a.manifest["first_recovery_save_step"]}'
        if not ok or not (checkpoint / 'RECOVERY_COMPLETE.json').exists():
            return
        record, hashes = verify_checkpoint(checkpoint)
        memory = [json.loads(x) for x in (self.supervisor / 'host_memory.jsonl').read_text().splitlines()[-600:]]
        guard = json.loads((self.a.dir / 'preflight.json').read_text())['cgroup']['job_limit_gib']
        peak = max(m['working_set_gib'] for m in memory)
        if peak > min(0.9 * guard, guard - 48) or any(m['oom_kill'] for m in memory):
            a.status('CLAIMED', verification='host memory outside guard', peak_working_set_gib=peak)
            return
        self.verified = True
        a.lease.update(a.launch['attempt'], state='TRAINING_VERIFIED')
        a.status('TRAINING_VERIFIED', updates=detail, checkpoint=str(checkpoint),
                 checkpoint_files=len(hashes), recent_peak_working_set_gib=peak)
        actions = cancel_recorded_sibling(a.campaign, a.launch['run_id'], a.launch['attempt'])
        append_jsonl(a.dir / 'events.jsonl', dict(event='sibling_actions', actions=actions))
        for run in a.launch.get('submit_ablations', []):
            if Lease(a.campaign, run).read() or any(
                    json.loads(x).get('run_id') == run and json.loads(x).get('event') == 'submitted'
                    for x in (a.campaign / 'submission_ledger.jsonl').read_text().splitlines()):
                continue
            result = subprocess.run([sys.executable, '-m', 'scripts.vln.submit_episodic', 'submit',
                                     '--campaign-root', str(a.campaign), '--run', run,
                                     '--profiles', a.launch['profile'], '--qualify', 'startup',
                                     '--mem', f'{a.launch["profile"]}={a.launch["resources"]["mem"]}'],
                                    cwd=ROOT, capture_output=True, text=True)
            append_jsonl(a.dir / 'events.jsonl', dict(event='ablation_submitted', run=run,
                                                      returncode=result.returncode,
                                                      stdout=result.stdout[-2000:], stderr=result.stderr[-2000:]))
        a.lease.update(a.launch['attempt'], state='ACTIVE')
        a.status('ACTIVE')


def train(a):
    spec = a.launch['resources']
    out = a.dir / 'train'
    supervisor = a.dir / 'supervisor'
    resume = []
    if out.exists() and list(out.glob('checkpoint-*')):
        resume = ['--resume_from_checkpoint', 'auto']   # same attempt, profile and source only
    command = [sys.executable, '-m', 'scripts.vln.run_episodic_gate', '--out', str(supervisor),
               '--disk-path', str(a.dir), '--',
               'torchrun', '--standalone', f'--nproc_per_node={spec["world"]}', '-m', 'qwen_vl.train.train_qwen',
               '--vln_config', 'configs/episodic/science_common.yaml',
               '--output_config', f'configs/episodic/arm_{a.launch["arm"]}.yaml',
               '--runtime_profile', spec['runtime'], '--deepspeed', spec['deepspeed'],
               '--manifest', a.manifest['manifest'], '--model_name_or_path', a.manifest['model_path'],
               '--output_dir', str(out), '--run-name', a.launch['attempt'],
               '--recovery-save-steps', str(a.manifest['recovery_save_steps']),
               '--first-recovery-save-step', str(a.manifest['first_recovery_save_step']),
               '--stop-after-epochs', str(a.manifest['epochs_to_train']), *resume]
    if supervisor.exists():
        supervisor = a.dir / f'supervisor-{a.job}'
        command[command.index('--out') + 1] = str(supervisor)
    agent = Agent(a, out, supervisor)
    process = subprocess.Popen(command, cwd=ROOT)

    def forward(signum, frame):
        # Slurm warning: request a checkpoint at the next optimizer boundary.
        (out / 'REQUEST_CHECKPOINT_AND_STOP').touch()
    signal.signal(signal.SIGUSR1, forward)
    agent.start()
    code = process.wait()
    agent.stop.set()
    return code, out


def upload(a, out):
    from qwen_vl.train.recovery import completed_checkpoints
    epoch = [p for step, p, r in completed_checkpoints(out) if r['epoch_boundary']]
    if not epoch or not (out / 'final' / 'pytorch_model.bin').exists():
        raise RuntimeError('No complete epoch checkpoint/export to publish')
    command = [sys.executable, '-m', 'scripts.vln.upload_episodic', '--repo', a.manifest['hf_repo'],
               '--folder', f'{a.launch["run_id"]}/{a.launch["attempt"]}', '--checkpoint', str(epoch[-1]),
               '--export', str(out / 'final'), '--attempt-dir', str(a.dir)]
    return subprocess.run(command, cwd=ROOT).returncode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt_dir')
    args = p.parse_args()
    a = Attempt(args.attempt_dir)
    a.status('ALLOCATED', host=os.uname().nodename, gpus=os.environ.get('CUDA_VISIBLE_DEVICES'))
    reason = standby_reason(a)
    if reason:
        a.status('DUPLICATE_STANDBY', reason=reason)
        return 0
    record = preflight(a)
    if record['problems']:
        a.status('FAILED', stage='preflight', problems=record['problems'])
        return 1
    a.status('PREFLIGHT_PASSED')
    reason = standby_reason(a)
    if reason:
        a.status('DUPLICATE_STANDBY', reason=reason)
        return 0
    current = a.lease.read()
    resuming = current is not None and current.get('attempt') == a.launch['attempt']
    # A resumed attempt (same profile/source) still qualifies its new allocation.
    report = qualify(a, 'startup' if resuming else a.launch['qualification'])
    if not report.get('passed'):
        a.status('FAILED', stage='qualification', report=str(a.dir / f'qualification-{a.job}'))
        return 1
    if not resuming:
        a.status('PREFLIGHT_PASSED', qualification='passed')
        ok, record = a.lease.claim(a.launch['attempt'], a.job, a.launch['profile'])
        if not ok:
            ok, record = a.lease.takeover_if_stale(a.launch['attempt'], a.job, a.launch['profile'])
        if not ok:
            a.status('DUPLICATE_STANDBY', reason=f'lost lease race: {record}')
            return 0
        a.status('CLAIMED')
    else:
        a.lease.update(a.launch['attempt'], job_id=a.job, resumed_by=a.job)
        a.status('CLAIMED', resumed=True)
    code, out = train(a)
    if code != 0:
        a.lease.update(a.launch['attempt'], state='FAILED')
        a.status('FAILED', stage='training', exit_code=code)
        return code
    from qwen_vl.train.recovery import completed_checkpoints
    state = json.loads((out / 'trainer_state.json').read_text())
    per_epoch = state['max_steps'] // 2
    boundary = [p for step, p, r in completed_checkpoints(out)
                if r['epoch_boundary'] and step == per_epoch * a.manifest['epochs_to_train']]
    if not boundary:
        # e.g. Slurm pre-termination checkpoint: resumable, never published as an epoch.
        a.status('FAILED', stage='interrupted_before_epoch_boundary', global_step=state['global_step'])
        return 3
    a.lease.update(a.launch['attempt'], state='COMPLETE')
    a.status('COMPLETE', epochs=a.manifest['epochs_to_train'], checkpoint=str(boundary[-1]))
    code = upload(a, out)
    a.status('COMPLETE', uploaded=code == 0)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
