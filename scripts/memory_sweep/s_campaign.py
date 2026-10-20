"""Matched one-epoch training queue for the GDN-only S arms on two 4xH100 lanes.

Every arm reuses the completed M5_E1 recipe exactly (fresh original Qwen3.5-4B,
joint R2R + English RxR15deg, seed 429 and its data order, 3,852 updates,
116 warmup, cosine to 10%, LR 5e-6, global batch 8 = 4 ranks x 1 episode x
GAS2, full BPTT, canonical text actions, frozen vision).  The resolved config
is asserted equal to M5_E1's own resolved config except the memory section
and two loader-only fields; S0 *is* M5_E1 and is never retrained.

Lane A qualifies once for everyone (CPU suite, provenance, training-only
calibration, real-model serving replay of every arm, fresh-process recovery);
lane B waits for its predecessor to finish successfully and for that seal.
Per arm: longest-RxR smoke under the arm's own scorer, then production with
identity-bound automatic resume, completion validation, and pruning of
redundant rolling checkpoints (the epoch checkpoint and export are kept).
Each lane runs as a worker-resident Slurm step and survives SSH loss; it still
depends on its allocation staying alive.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

VERSION = 'm5_gdn_s_sweep_campaign_v1'
EXPECTED_STEPS, WORLD = 3852, 4
LOADER = dict(dataloader_num_workers=4, dataloader_prefetch_factor=1)  # proven safe at 480 GiB
CALIBRATED = ('S6', 'S8N', 'S8W')
QUALIFICATION_GATES = ('cpu-suite', 'provenance', 'calibration', 'serving-replay', 'recovery')


# ---- small pure helpers (unit-tested) ------------------------------------
def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def expected_config(reference, memory):
    """M5_E1's resolved config with only the memory section and loader fields replaced."""
    if reference['memory'].get('episodic_score') != 'committed_mean_l2_v1' or 'research' in reference:
        raise ValueError('Reference must be plain M5 (S0)')
    result = copy.deepcopy(reference)
    result['memory'] = copy.deepcopy(memory)
    result['training'].update(LOADER)
    return result


def ready(active, own, allocation):
    """Only this controller and the allocation's own shell/extern/batch steps may exist."""
    return not (set(active) - {own, *[allocation + s for s in ('.0', '.extern', '.batch')]})


def choose_threads(baseline, candidate):
    """Use more CPU-Adam threads only if the optimizer step clearly improves and nothing regresses."""
    if candidate['step_seconds'] < 0.75 * baseline['step_seconds'] and (
            candidate['update_seconds'] <= 1.03 * baseline['update_seconds']):
        return candidate['threads']
    return baseline['threads']


def optimizer_step_estimate(train_dir, world=WORLD, update=2):
    """Median over ranks of second-minus-first microstep backward time in ``update``.

    Smoke fixtures repeat one episode, so both microsteps of an update have the
    same forward/backward work and the difference is the CPU-Adam step.
    """
    estimates, seconds = [], []
    for rank in range(world):
        rows = [json.loads(line) for line in (Path(train_dir) / f'microtimes_rank{rank}.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r['update'] == update]
        if len(rows) != 2:
            raise ValueError(f'rank {rank}: expected two microsteps in update {update}')
        estimates.append(rows[1]['backward_and_step_overhead_seconds'] - rows[0]['backward_and_step_overhead_seconds'])
        profile = [json.loads(line) for line in (Path(train_dir) / f'profile_rank{rank}.jsonl').read_text().splitlines()]
        seconds += [p['seconds'] for p in profile if p['update'] == update]
    estimates.sort()
    return dict(step_seconds=(estimates[1] + estimates[2]) / 2, update_seconds=max(seconds),
                per_rank_step_seconds=estimates)


def prune_rolling(train_dir, steps=EXPECTED_STEPS):
    """After a validated epoch, delete only this run's completed rolling checkpoints."""
    from qwen_vl.train.recovery import completed_checkpoints, validate_complete
    train_dir = Path(train_dir).resolve()
    final = validate_complete(train_dir / f'checkpoint-{steps}')
    if final['step'] != steps or not final['epoch_boundary'] or not (train_dir / 'final').is_dir():
        raise ValueError('Refusing to prune before a complete epoch checkpoint and export')
    removed = []
    for step, path, record in completed_checkpoints(train_dir):
        if step < steps and not record['epoch_boundary']:
            if path.parent.resolve() != train_dir or path.is_symlink():
                raise ValueError(f'Unsafe pruning target {path}')
            shutil.rmtree(path)
            removed.append(path.name)
    return removed


# ---- campaign definition ---------------------------------------------------
class Campaign:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        if self.data['version'] != VERSION:
            raise ValueError('Unknown campaign version')
        self.root = Path(self.data['root'])
        self.source = Path(self.data['source'])
        self.base = Path(self.data['base'])
        self.shared = self.root / 'shared'
        self.configs = self.root / 'configs'

    def arm_memory_path(self, arm, role='training'):
        folder = self.configs if role == 'training' else self.configs / 'screening'
        return folder / f'arm_{arm}.yaml'

    def resolve(self, arm, runtime=None):
        from qwen_vl.train.vln_runtime import resolve_config
        s = self.source
        return resolve_config(s / 'configs/episodic/science_common.yaml', s / 'configs/vln_empty.yaml',
                              self.arm_memory_path(arm), s / 'configs/vggt/g3_joint_one_epoch.yaml',
                              runtime or self.configs / 'runtime.yaml')

    def source_identity(self):
        head = subprocess.check_output(['git', '-C', str(self.source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(self.source), 'status', '--porcelain'], text=True).strip()
        if head != self.data['source_commit'] or dirty:
            raise ValueError(f'Source {head} dirty={bool(dirty)} differs from {self.data["source_commit"]}')
        return head

    def run_identity(self, arm):
        """Everything that defines one arm's training; lane assignment is excluded."""
        import yaml
        from qwen_vl.contracts import validate_config
        from qwen_vl.research.episodic_contract import parse_episodic_spec
        reference = json.loads(Path(self.data['m5_e1_reference_config']).read_text())
        memory = yaml.safe_load(self.arm_memory_path(arm).read_text())['memory']
        cfg = self.resolve(arm)
        if cfg != expected_config(reference, memory):
            raise ValueError(f'{arm}: resolved config differs from the M5_E1 recipe beyond memory/loader')
        validate_config(copy.deepcopy(cfg), WORLD)
        spec = parse_episodic_spec(cfg)
        if spec.score != arm or spec.version != 'episodic_window_gdn_sweep_v1':
            raise ValueError(f'{arm}: config declares {spec.score}')
        seal = self.shared / 'QUALIFIED.json'
        return dict(campaign=VERSION, arm=arm, config=cfg, source_commit=self.source_identity(),
                    manifest_sha256=self.data['manifest_sha256'], model=self.data['model'],
                    memory_sha256=sha(self.arm_memory_path(arm)), runtime_sha256=sha(self.configs / 'runtime.yaml'),
                    qualification_sha256=sha(seal) if seal.exists() else None)

    def run_dir(self, arm):
        return self.root / 'runs' / f'{arm}_seed{self.data["seed"]}_e1'


# ---- commands ----------------------------------------------------------------
def train_command(campaign, arm, out, *, manifest=None, updates=None, profile_only=True, save_every=None,
                  resume=None, first_save=None, runtime=None, production=False, host_guard=432.0):
    c = campaign.data
    command = ['torchrun', '--standalone', f'--nproc_per_node={WORLD}', '-m', 'qwen_vl.train.train_qwen',
               '--vln_config', 'configs/episodic/science_common.yaml', '--output_config', 'configs/vln_empty.yaml',
               '--memory_config', str(campaign.arm_memory_path(arm)),
               '--policy_config', 'configs/vggt/g3_joint_one_epoch.yaml',
               '--runtime_profile', str(runtime or campaign.configs / 'runtime.yaml'),
               '--deepspeed', 'configs/deepspeed_zero2_cpu_optimizer.json',
               '--manifest', str(manifest or c['manifest']), '--model_name_or_path', c['model'],
               '--output_dir', str(Path(out) / 'train'), '--run-name', f'{arm.lower()}_seed{c["seed"]}_e1']
    if production:
        command += ['--recovery-save-steps', '100', '--first-recovery-save-step', '10', '--resume_from_checkpoint', 'auto']
        supervisor = Path(out) / ('supervisor-' + time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + f'-{os.getpid()}')
    else:
        command += ['--max-optimizer-updates', str(updates), '--debug-repeat-episodes']
        command += ['--recovery-save-steps', str(save_every)] if save_every else (['--profile-only'] if profile_only else [])
        if resume:
            command += ['--resume_from_checkpoint', str(resume)]
        if first_save:
            command += ['--first-recovery-save-step', str(first_save)]
        supervisor = Path(out) / 'supervisor'
    return [sys.executable, '-m', 'scripts.vln.run_episodic_gate', '--out', str(supervisor),
            '--host-guard-gib', str(host_guard), '--disk-path', str(campaign.base),
            '--min-free-disk-gib', str(c['min_free_disk_gib']), '--', *command]


def latest_resources(run):
    supervisors = sorted(Path(run).glob('supervisor-*'))
    if not supervisors or not (supervisors[-1] / 'resources.json').exists():
        return None
    return json.loads((supervisors[-1] / 'resources.json').read_text())


def run_complete(run, cfg, steps=EXPECTED_STEPS):
    """Full epoch on 4 ranks, complete checkpoint, identical navigation config and an export."""
    from qwen_vl.train.recovery import validate_complete
    run = Path(run)
    final = run / 'train' / f'checkpoint-{steps}'
    if not final.exists():
        return False
    resources = latest_resources(run)
    if resources is None or resources.get('status') != 'PASS':
        return False
    record = validate_complete(final)
    state = json.loads((final / 'trainer_state.json').read_text())
    if record['step'] != steps or record['world_size'] != WORLD or state['global_step'] != steps or state['epoch'] < .999:
        raise ValueError(f'{run}: incomplete or wrong-topology epoch checkpoint')
    for path in (final / 'navigation.json', run / 'train/final/navigation.json'):
        if json.loads(path.read_text())['config'] != cfg:
            raise ValueError(f'{path}: navigation contract differs from the bound identity')
    return True


# ---- lane controller ---------------------------------------------------------
class Lane:
    def __init__(self, campaign, name):
        self.campaign, self.name = campaign, name
        self.spec = campaign.data['lanes'][name]
        self.dir = campaign.root / 'lanes' / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.job = self.spec['job']
        self.host_guard = self.spec['host_guard_gib']
        self.own = None

    # -- bookkeeping --
    def record(self, status, **fields):
        data = dict(status=status, unix=time.time(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    lane=self.name, step=self.own, pid=os.getpid(), **fields)
        write(self.dir / 'state.json', data)
        with (self.dir / 'history.jsonl').open('a') as f:
            f.write(json.dumps(data, sort_keys=True) + '\n')
        print(status, json.dumps(fields, sort_keys=True), flush=True)

    def active_steps(self):
        return subprocess.check_output(['squeue', '--steps', '-h', '-j', self.job, '-o', '%i'], text=True).split()

    def gpu_busy(self):
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=60)
        return bool(out.stdout.strip()) or out.returncode != 0

    def run_logged(self, name, command, env_extra=None, check=True):
        env = dict(os.environ, **(env_extra or {}))
        log = self.dir / f'{name}-{time.time_ns()}.log'
        with log.open('x') as f:
            status = subprocess.run(command, cwd=self.campaign.source, env=env, stdout=f,
                                    stderr=subprocess.STDOUT).returncode
        if check and status:
            raise RuntimeError(f'{name} failed with exit {status}; see {log}')
        return status

    def control(self, name):
        return (self.dir / name).exists()

    # -- waiting --
    def wait_predecessor(self):
        predecessor = self.spec.get('predecessor')
        if not predecessor:
            return
        while True:
            try:
                status = json.loads(Path(predecessor['state']).read_text()).get('status')
            except (OSError, ValueError):
                status = 'unreadable'
            alive = predecessor['step'] in self.active_steps()
            if self.control('OVERRIDE_PREDECESSOR'):
                self.record('predecessor-overridden', predecessor_status=status)
                break
            if status == 'complete' and not alive:
                break
            self.record('waiting-predecessor', predecessor_status=status, predecessor_alive=alive)
            time.sleep(120)
        while not (ready(self.active_steps(), self.own, self.job) and not self.gpu_busy()):
            self.record('waiting-for-idle-allocation', steps=self.active_steps())
            time.sleep(60)

    def wait_qualification(self):
        seal = self.campaign.shared / 'QUALIFIED.json'
        while not seal.exists():
            self.record('waiting-shared-qualification')
            time.sleep(120)
        data = json.loads(seal.read_text())
        if not data.get('passed') or data.get('source_commit') != self.campaign.source_identity():
            raise RuntimeError('Shared qualification failed or belongs to another source')
        return data

    # -- shared qualification (lane A) --
    def qualify(self):
        c, shared = self.campaign, self.campaign.shared
        seal = shared / 'QUALIFIED.json'
        if seal.exists():
            return self.wait_qualification()
        shared.mkdir(parents=True, exist_ok=True)
        gates = json.loads((shared / 'gates.json').read_text()) if (shared / 'gates.json').exists() else {}

        def gate(name, fn):
            if gates.get(name, {}).get('passed'):
                return
            self.record('qualifying', gate=name)
            result = fn()
            gates[name] = dict(passed=True, result=result, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            write(shared / 'gates.json', gates)
        source = c.source_identity()
        py = sys.executable
        gate('cpu-suite', lambda: self.run_logged('cpu-suite', [
            py, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/memory_sweep',
            'tests/vln/test_episodic_contract.py', 'tests/vln/test_episodic_selector.py',
            'tests/vln/test_episodic_attention.py', 'tests/vln/test_episodic_forward.py',
            'tests/vln/test_committed_write_reference.py']) or 'pass')

        def provenance():
            self.run_logged('provenance', [py, '-m', 'scripts.memory_sweep.audit_manifest', '--campaign', str(c.path),
                                           '--out', str(shared / 'provenance.json')])
            return json.loads((shared / 'provenance.json').read_text())['checks']
        gate('provenance', provenance)

        def calibration():
            out = shared / 'calibration'
            self.run_logged('calibration', [
                'torchrun', '--standalone', f'--nproc_per_node={WORLD}', '-m', 'scripts.memory_sweep.calibrate',
                '--manifest', c.data['manifest'], '--model-path', c.data['model'], '--m5-e1', c.data['m5_e1'],
                '--memory-config', str(c.source / 'configs/episodic/arm_m5.yaml'),
                '--runtime-profile', str(c.configs / 'runtime.yaml'), '--out', str(out)])
            record = out / 'calibration.json'
            self.run_logged('arm-configs-training', [py, '-m', 'scripts.memory_sweep.make_arm_configs', '--calibration',
                                                     str(record), '--role', 'base', '--out', str(c.configs)])
            self.run_logged('arm-configs-screening', [py, '-m', 'scripts.memory_sweep.make_arm_configs', '--calibration',
                                                      str(record), '--role', 'm5_e1', '--out', str(c.configs / 'screening')])
            return dict(calibration_sha256=sha(record), configs={a: sha(c.arm_memory_path(a)) for a in CALIBRATED})
        gate('calibration', calibration)

        def serving_replay():
            arms = c.data['screening_arms']
            shards = [arms[i::WORLD] for i in range(WORLD)]
            processes = []
            for index, shard in enumerate(shards):
                command = [py, '-m', 'scripts.memory_sweep.replay_check', '--checkpoint', c.data['m5_e1'],
                           '--model-path', c.data['model'], '--manifest', c.data['manifest'],
                           '--out', str(shared / 'replay' / f'shard{index}.json')]
                for arm in shard:
                    command += ['--arm', arm, str(c.arm_memory_path(arm, 'screening'))]
                log = (self.dir / f'replay-shard{index}-{time.time_ns()}.log').open('x')
                processes.append((subprocess.Popen(command, cwd=c.source, stdout=log, stderr=subprocess.STDOUT,
                                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(index))), log))
            codes = [p.wait() for p, _ in processes]
            for _, log in processes:
                log.close()
            summary = {}
            for index in range(WORLD):
                path = shared / 'replay' / f'shard{index}.json'
                if not path.exists():           # a crash, not a tolerance result
                    raise RuntimeError(f'Serving replay shard {index} crashed (exit {codes[index]})')
                for r in json.loads(path.read_text())['replay']:
                    summary[f'{r["arm"]}/{r["dataset"]}'] = {k: r.get(k) for k in (
                        'passed', 'rms_logit_error', 'max_logit_error', 'plan_id_mismatches', 'mean_selected_overlap',
                        'max_relative_score_error', 'max_resident_kv', 'action_nll', 'overlap_vs_S0_legacy')}
            # The harness itself must reproduce qualified S0 under the unchanged BF16 gates.
            for key, record in summary.items():
                if key.split('/')[0] in ('S0_legacy', 'S0') and not record['passed']:
                    raise RuntimeError(f'S0 serving replay failed: {key} {record}')
            # A new arm that misses the unchanged serving tolerance is reported, never relaxed;
            # it blocks closed-loop serving qualification of that arm, not its offline training.
            flagged = sorted({k.split('/')[0] for k, r in summary.items() if not r['passed']})
            return dict(records=summary, serving_tolerance_failed=flagged)
        gate('serving-replay', serving_replay)

        def recovery():
            arm, root = c.data['recovery_arm'], shared / 'recovery'
            fixture = Path(c.data['fixtures']) / 'mixed_tail.jsonl'
            diagnostic = self.diagnostic_runtime()
            for name, extra in (('reference', dict(save_every=2)),
                                ('resumed', dict(save_every=2, resume=root / 'reference/train/checkpoint-2', first_save=3))):
                out = root / name
                if out.exists():
                    raise RuntimeError(f'Preserve interrupted recovery evidence {out} before retry')
                self.run_logged(f'recovery-{name}', train_command(c, arm, out, manifest=fixture, updates=3,
                                                                  runtime=diagnostic, host_guard=self.host_guard, **extra))
                self.check_profile(out / 'train', 3, 1 if name == 'reference' else 3)
                cfg = json.loads((out / 'train/resolved_config.json').read_text())
                expected = c.resolve(arm, diagnostic)
                if cfg != expected:
                    raise RuntimeError('Recovery diagnostic config differs')
            self.run_logged('recovery-compare', [py, '-m', 'scripts.memory_sweep.compare_recovery',
                                                 '--root', str(root), '--out', str(root / 'comparison.json')])
            return json.loads((root / 'comparison.json').read_text())
        gate('recovery', recovery)
        if set(gates) != set(QUALIFICATION_GATES) or not all(g['passed'] for g in gates.values()):
            raise RuntimeError('Incomplete shared qualification')
        write(seal, dict(passed=True, source_commit=source, gates=gates,
                         calibration_sha256=gates['calibration']['result']['calibration_sha256']))
        self.record('qualified')
        return json.loads(seal.read_text())

    def diagnostic_runtime(self):
        """Production runtime with epoch saves disabled for short diagnostic fixtures."""
        import yaml
        path = self.campaign.shared / 'diagnostic-runtime.yaml'
        runtime = yaml.safe_load((self.campaign.configs / 'runtime.yaml').read_text())
        runtime['training']['save_strategy'] = 'no'
        text = yaml.safe_dump(runtime)
        if path.exists():
            if path.read_text() != text:
                raise RuntimeError('Diagnostic runtime changed')
        else:
            path.write_text(text)
        return path

    def check_profile(self, train_dir, updates, first_update=1, max_reserved=78.0):
        from scripts.vln.profile_episodic import verify
        check = verify(train_dir, WORLD, updates, 8, max_reserved, first_update=first_update)
        resources = json.loads((Path(train_dir).parent / 'supervisor/resources.json').read_text())
        if not check['passed'] or resources['status'] != 'PASS':
            raise RuntimeError(f'Profile gate failed: {check} {resources}')
        return dict(check=check, resources=resources)

    # -- per arm --
    def smoke(self, arm, run, threads):
        """Two updates of the complete longest RxR episode under this arm's scorer."""
        out = run / f'smoke-omp{threads}'
        result_path = out / 'result.json'
        if result_path.exists():
            return json.loads(result_path.read_text())
        if out.exists():
            raise RuntimeError(f'Preserve interrupted smoke {out} before retry')
        fixture = Path(self.campaign.data['fixtures']) / 'rxrce_longest.jsonl'
        self.record('smoke', arm=arm, threads=threads)
        self.run_logged(f'{arm}-smoke-omp{threads}', train_command(
            self.campaign, arm, out, manifest=fixture, updates=2, profile_only=True, host_guard=self.host_guard),
            env_extra=dict(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)))
        result = self.check_profile(out / 'train', 2)
        cfg = json.loads((out / 'train/resolved_config.json').read_text())
        if cfg != self.campaign.resolve(arm):
            raise RuntimeError(f'{arm}: smoke config differs from production')
        result.update(threads=threads, **optimizer_step_estimate(out / 'train'))
        write(result_path, result)
        return result

    def lane_threads(self, arm, run):
        path = self.dir / 'threads.json'
        if path.exists():
            return json.loads(path.read_text())['threads']
        baseline = self.smoke(arm, run, 4)
        candidate = self.smoke(arm, run, 8)
        threads = choose_threads(baseline, candidate)
        write(path, dict(threads=threads, baseline=baseline, candidate=candidate, arm=arm))
        self.record('threads-selected', threads=threads, baseline_step=baseline['step_seconds'],
                    candidate_step=candidate['step_seconds'])
        return threads

    def train_arm(self, arm):
        c = self.campaign
        identity = c.run_identity(arm)
        run = c.run_dir(arm)
        run.mkdir(parents=True, exist_ok=True)
        bound = run / 'run_identity.json'
        if bound.exists():
            if json.loads(bound.read_text()) != identity:
                raise RuntimeError(f'{arm}: bound run identity changed')
        else:
            if any(p.name != 'run_identity.json' for p in run.iterdir()):
                raise RuntimeError(f'{arm}: refusing to adopt unbound outputs')
            write(bound, identity)
        cfg = identity['config']
        if run_complete(run, cfg):
            return 'complete'
        threads = self.lane_threads(arm, run)
        if threads not in (4, 8):
            raise RuntimeError('Unexpected thread choice')
        if not (run / f'smoke-omp{threads}/result.json').exists():
            self.smoke(arm, run, threads)
        for attempt in range(1, 4):
            if not ready(self.active_steps(), self.own, self.job):
                raise RuntimeError('Foreign compute step in this allocation')
            with (run / 'launcher.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.record('training', arm=arm, attempt=attempt, threads=threads, run=str(run))
                self.run_logged(f'{arm}-production', train_command(
                    c, arm, run, production=True, host_guard=self.host_guard),
                    env_extra=dict(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads)), check=False)
            if run_complete(run, cfg):
                removed = prune_rolling(run / 'train')
                self.record('arm-complete', arm=arm, pruned=removed)
                return 'complete'
            self.record('production-attempt-failed', arm=arm, attempt=attempt, resources=latest_resources(run))
            time.sleep(120)
        return 'failed'

    def run_arms(self):
        """Arms in lane order; control files ``STOP_AFTER_CURRENT`` and ``SKIP_<ARM>``."""
        outcomes = {}
        for arm in self.spec['arms']:
            if self.control('STOP_AFTER_CURRENT'):
                self.record('stopped-by-request', outcomes=outcomes)
                return outcomes
            if self.control(f'SKIP_{arm}'):
                outcomes[arm] = 'skipped'
                continue
            try:
                outcomes[arm] = self.train_arm(arm)
            except Exception as exc:          # one arm's failure must not idle the lane
                outcomes[arm] = f'error: {exc!r}'[:500]
                self.record('arm-error', arm=arm, error=repr(exc)[:2000])
        self.record('lane-complete', outcomes=outcomes)
        return outcomes

    def main(self):
        if os.environ.get('SLURM_JOB_ID') != self.job or os.environ.get('SLURMD_NODENAME') != 'worker-2':
            raise RuntimeError('Lane must run inside its own worker-2 allocation')
        self.own = self.job + '.' + os.environ['SLURM_STEP_ID']
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(15):
                write(self.dir / 'heartbeat.json', dict(unix=time.time(), step=self.own, pid=os.getpid()))
        with (self.dir / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                self.campaign.source_identity()
                self.wait_predecessor()
                if not ready(self.active_steps(), self.own, self.job) or self.gpu_busy():
                    raise RuntimeError('Allocation is not idle')
                if self.spec.get('qualifies'):
                    self.qualify()
                else:
                    self.wait_qualification()
                self.run_arms()
            except BaseException as exc:
                self.record('blocked', error=repr(exc)[:2000])
                raise
            finally:
                stop.set()
                thread.join()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--lane', required=True)
    p.add_argument('--preflight', action='store_true', help='Check identities of every arm without GPU work')
    args = p.parse_args()
    campaign = Campaign(args.campaign)
    if args.preflight:
        arms = [a for lane in campaign.data['lanes'].values() for a in lane['arms']]
        report = {}
        for arm in arms:
            if arm in CALIBRATED and not campaign.arm_memory_path(arm).exists():
                report[arm] = 'pending-calibration'
                continue
            identity = campaign.run_identity(arm)
            report[arm] = dict(memory_sha256=identity['memory_sha256'], kv_guard=identity['config']['memory']['kv_guard_tokens'])
        print(json.dumps(report, indent=2))
        print('PREFLIGHT_PASS', flush=True)
        return
    Lane(campaign, args.lane).main()


if __name__ == '__main__':
    main()
