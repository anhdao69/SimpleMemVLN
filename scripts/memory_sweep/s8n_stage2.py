"""S8N stage 2: one epoch of S8N's own DAgger rollouts from S8N's final weights, on two 4xH100 lanes of one job.

S8N_DAGGER         the DAgger rows as collected: every step supervises the executed action.
S8N_DAGGER_ORACLE  the same rollouts, frames and executed history in the context; every step supervises the
                   oracle's action at that state (``target_action_name``).

Both arms start strictly from S8N's final weights (SHA-256 bound; the file published as anhdao69/M5_S/S8N) with a
fresh AdamW and S8N's own resolved config (byte copies of its memory and runtime files) except the epoch schedule
of the DAgger manifest: ceil(ceil(N / 4) / GAS) updates and ceil(3%) warmup, the rule that gives S8N's 3,852/116.
The data order is the seed's first epoch; both arms' episodes encode to identical lengths, so their orders are
identical.  Lane A trains on GPUs 0-3 with socket-0 CPUs, lane B on GPUs 4-7 with socket-1 CPUs.

Gates per lane: parent (S8N identity, final weights, manifests, configs), cpu-suite, init (three updates of the
full manifest from the bound weights; both arms must see the same data order, the as-collected arm's update-1 loss
must come from trained weights and the oracle targets must raise it).  Then production with identity-bound
automatic resume, completion validation and pruning of rolling checkpoints.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from scripts.memory_sweep import s_campaign as base
from scripts.memory_sweep.s8n_followup import CPU_SUITE as FOLLOWUP_SUITE, cached_paths, config_diff, \
    drop_cache, exposure_plan, thread_env, working_set_gib
from scripts.memory_sweep.s_campaign import WORLD, latest_resources, prune_rolling, run_complete, sha, write

VERSION = 's8n_stage2_dagger_campaign_v1'
PARENT = 'S8N'
ARMS = {'S8N_DAGGER': 'executed_recovery_actions', 'S8N_DAGGER_ORACLE': 'oracle_actions_on_executed_rollout'}
SCHEDULE_KEYS = ('expected_total_steps', 'warmup_steps', 'expected_warmup_steps')
INIT_UPDATES = 3
STEP_PREFIX = 's8n-stage2-lane-'
CPU_SUITE = (*FOLLOWUP_SUITE, 'tests/vln/test_contracts.py', 'tests/vln/test_serializer.py',
             'tests/vln/test_stream_core.py')


# ---- small pure helpers (unit-tested) ------------------------------------
def epoch_schedule(episodes, accumulation, world=WORLD, warmup_fraction=0.03):
    """One epoch: ceil(N / world) episodes per rank (accelerate_repeat tail), an update every ``accumulation``
    of them, warmup ceil(3%).  S8N's 30,815 episodes give its 3,852 updates / 116 warmup."""
    updates = math.ceil(math.ceil(episodes / world) / accumulation)
    warmup = math.ceil(round(warmup_fraction * updates, 9))
    return dict(expected_total_steps=updates, warmup_steps=warmup, expected_warmup_steps=warmup)


def policy_text(source, schedule):
    """The one-epoch policy file with the stage-2 manifest's epoch schedule."""
    import yaml
    policy = yaml.safe_load((Path(source) / 'configs/vggt/g3_joint_one_epoch.yaml').read_text())
    if set(policy) != {'training'} or set(policy['training']) != {'epochs', *SCHEDULE_KEYS} \
            or policy['training']['epochs'] != 1:
        raise ValueError('Unexpected one-epoch policy')
    policy['training'].update(schedule)
    return (f'# {VERSION}: configs/vggt/g3_joint_one_epoch.yaml with the epoch schedule of the stage-2 manifest\n'
            + yaml.safe_dump(policy, sort_keys=False))


def foreign_steps(lines, own, allocation):
    """`squeue -o '%i %j'` lines other than this controller, the allocation's own steps and the campaign lanes."""
    allowed = {own, *[allocation + s for s in ('.0', '.extern', '.batch')]}
    foreign = []
    for line in lines:
        fields = line.split(None, 1)
        if fields and fields[0] not in allowed and not (len(fields) > 1 and fields[1].strip().startswith(STEP_PREFIX)):
            foreign.append(line.strip())
    return foreign


def cpu_list(text):
    cpus = set()
    for part in text.split(','):
        first, _, last = part.partition('-')
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def compare_init(results, original_update1_loss):
    """First updates of both arms from the bound weights (pure; unit-tested).

    ``results``: arm -> dict(losses, grad_norms, plan).  Both arms must see the same data order; the as-collected
    arm supervises S8N's own choices on most steps, so its update-1 loss must be far below the original model's
    (S8N's logged update 1), and the oracle targets must raise it.
    """
    problems = []
    for arm, run in results.items():
        values = run['losses'] + run['grad_norms']
        if len(run['losses']) != INIT_UPDATES or not all(math.isfinite(v) for v in values):
            problems.append(f'{arm}: losses {run["losses"]} grad norms {run["grad_norms"]}')
    if problems:
        return dict(passed=False, problems=problems)
    executed, oracle = results['S8N_DAGGER'], results['S8N_DAGGER_ORACLE']
    if executed['plan'] != oracle['plan']:
        problems.append('the arms see different data orders')
    if not executed['losses'][0] < 0.5 * original_update1_loss:
        problems.append(f'as-collected update-1 loss {executed["losses"][0]} is not from trained weights')
    if not oracle['losses'][0] > executed['losses'][0]:
        problems.append('oracle targets do not raise the update-1 loss over the executed actions')
    return dict(passed=not problems, problems=problems, original_model_update1_loss=original_update1_loss,
                losses={a: r['losses'] for a, r in results.items()},
                grad_norms={a: r['grad_norms'] for a, r in results.items()})


# ---- campaign definition ---------------------------------------------------
class Campaign(base.Campaign):
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
        self.parent = self.data['parent']
        self.arms = [arm for lane in self.data['lanes'].values() for arm in lane['arms']]
        if sorted(self.arms) != sorted(ARMS) or set(self.data['manifests']) != set(ARMS):
            raise ValueError('The campaign must train each stage-2 arm exactly once')
        episodes = {m['episodes'] for m in self.data['manifests'].values()}
        if len(episodes) != 1:
            raise ValueError('Both arms must train the same episodes')
        self.schedule = epoch_schedule(episodes.pop(), self.data['accumulation'])

    def manifest(self, arm):
        return self.data['manifests'][arm]['path']

    def arm_memory_path(self, arm, role='training'):
        return self.configs / f'arm_{PARENT}.yaml'

    def policy_path(self, arm):
        return self.configs / 'policy_stage2.yaml'

    def resolve(self, arm, runtime=None):
        from qwen_vl.train.vln_runtime import resolve_config
        s = self.source
        return resolve_config(s / 'configs/episodic/science_common.yaml', s / 'configs/vln_empty.yaml',
                              self.arm_memory_path(arm), self.policy_path(arm),
                              runtime or self.configs / 'runtime.yaml')

    def parent_identity(self):
        path = Path(self.parent['run']) / 'run_identity.json'
        if sha(path) != self.parent['identity_sha256']:
            raise ValueError('S8N run identity changed')
        identity = json.loads(path.read_text())
        if (identity['arm'] != PARENT or identity['model'] != self.data['model']
                or identity['manifest_sha256'] != self.parent['manifest_sha256']):
            raise ValueError('S8N identity differs from the declared parent')
        return identity

    def expected_config(self):
        cfg = copy.deepcopy(self.parent_identity()['config'])
        cfg['training'].update(self.schedule)
        return cfg

    def run_identity(self, arm):
        """Everything that defines one arm's training; lane assignment is excluded."""
        from qwen_vl.contracts import validate_config
        from qwen_vl.research.episodic_contract import parse_episodic_spec
        from qwen_vl.research.matrix_optimizer_contract import parse_matrix_optimizer
        parent = self.parent_identity()
        for name, path in (('memory', self.arm_memory_path(arm)), ('runtime', self.configs / 'runtime.yaml')):
            if sha(path) != parent[f'{name}_sha256']:
                raise ValueError(f'{path} is not S8N\'s {name} file')
        if self.policy_path(arm).read_text() != policy_text(self.source, self.schedule):
            raise ValueError('Policy file differs from its declaration')
        cfg = self.resolve(arm)
        expected = self.expected_config()
        if cfg != expected:
            raise ValueError(f'{arm}: resolved config differs from S8N at {config_diff(cfg, expected)}')
        if cfg['training']['gradient_accumulation_steps'] != self.data['accumulation']:
            raise ValueError('Declared accumulation differs from S8N')
        validate_config(copy.deepcopy(cfg), WORLD)
        spec = parse_episodic_spec(cfg)
        if spec.score != PARENT or spec.version != 'episodic_window_gdn_sweep_v1':
            raise ValueError(f'{arm}: config declares {spec.score}')
        if parse_matrix_optimizer(cfg['training']) is not None:
            raise ValueError(f'{arm}: S8N trains with AdamW only')
        manifest = self.data['manifests'][arm]
        return dict(campaign=VERSION, arm=arm, parent=PARENT, config=cfg, source_commit=self.source_identity(),
                    manifest=manifest['path'], manifest_sha256=manifest['sha256'], supervision=ARMS[arm],
                    model=self.data['model'], memory_sha256=sha(self.arm_memory_path(arm)),
                    runtime_sha256=sha(self.configs / 'runtime.yaml'), policy_sha256=sha(self.policy_path(arm)),
                    parent_identity_sha256=self.parent['identity_sha256'],
                    init_weights_checkpoint=self.parent['weights_checkpoint'],
                    init_weights_sha256=self.parent['weights_sha256'], schedule=self.schedule, sampler_epoch_offset=0)


def prepare(campaign):
    """Byte copies of S8N's memory and runtime files and the stage-2 policy file (idempotent)."""
    import shutil
    c, parent = campaign, campaign.parent_identity()
    c.configs.mkdir(parents=True, exist_ok=True)
    for name, source, target in (('memory', c.parent['memory'], c.arm_memory_path(PARENT)),
                                 ('runtime', c.parent['runtime'], c.configs / 'runtime.yaml')):
        if not target.exists():
            shutil.copy2(source, target)
        if sha(target) != parent[f'{name}_sha256'] or sha(source) != parent[f'{name}_sha256']:
            raise ValueError(f'{target} is not S8N\'s {name} file')
    path, text = c.policy_path(None), policy_text(c.source, c.schedule)
    if path.exists():
        if path.read_text() != text:
            raise ValueError(f'{path} exists with different content')
    else:
        path.write_text(text)
    return {p.name: sha(p) for p in sorted(c.configs.iterdir()) if p.is_file()}


# ---- commands ----------------------------------------------------------------
def train_command(campaign, arm, out, *, manifest=None, **kwargs):
    """The S-campaign command with the arm's manifest, the stage-2 policy and S8N's bound weights."""
    command = base.train_command(campaign, arm, out, manifest=manifest or campaign.manifest(arm), **kwargs)
    index = command.index('--policy_config') + 1
    if command[index] != 'configs/vggt/g3_joint_one_epoch.yaml':
        raise RuntimeError('Unexpected policy argument')
    command[index] = str(campaign.policy_path(arm))
    return command + ['--init-weights-checkpoint', campaign.parent['weights_checkpoint'],
                      '--init-weights-sha256', campaign.parent['weights_sha256'], '--init-weights-new-schedule']


# ---- lane controller ---------------------------------------------------------
class Lane(base.Lane):
    def __init__(self, campaign, name):
        super().__init__(campaign, name)
        (self.arm,) = self.spec['arms']
        self.gpus, self.cpus = self.spec['gpus'], self.spec['cpus']

    def foreign(self):
        lines = subprocess.check_output(['squeue', '--steps', '-h', '-j', self.job, '-o', '%i %j'],
                                        text=True).splitlines()
        return foreign_steps(lines, self.own, self.job)

    def gpu_busy(self):
        out = subprocess.run(['nvidia-smi', '-i', self.gpus, '--query-compute-apps=pid', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=60)
        return bool(out.stdout.strip()) or out.returncode != 0

    def relieve_cache(self, stop):
        """Drop clean frame/tree cache whenever the job's working set nears the host guard (as in s8n_followup:
        frames both lanes read become active file cache of this job and count toward the guard)."""
        relief = self.campaign.data['cache_relief']
        limit, last = relief['fraction'] * self.host_guard, 0.0
        cgroup = next(Path('/sys/fs/cgroup').glob(f'**/job_{self.job}'), None)
        log = self.dir / 'cache_relief.jsonl'
        while cgroup is not None and not stop.wait(60):
            try:
                before = working_set_gib(cgroup)
                if before < limit or time.time() - last < relief['cooldown_seconds']:
                    continue
                last = started = time.time()
                paths = cached_paths(self.campaign.manifest(self.arm), relief['trees'])
                files = 0
                with ThreadPoolExecutor(relief['threads']) as pool:
                    while batch := list(itertools.islice(paths, 4096)):
                        files += sum(1 for _ in pool.map(drop_cache, batch))
                row = dict(before_gib=before, after_gib=working_set_gib(cgroup), limit_gib=limit, files=files,
                           seconds=time.time() - started)
            except Exception as exc:              # relief is best effort; never disturb the controller
                row = dict(error=repr(exc)[:500])
            with log.open('a') as f:
                f.write(json.dumps(dict(row, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))) + '\n')

    def qualify(self):
        c = self.campaign
        seal = self.dir / 'QUALIFIED.json'
        if seal.exists():
            data = json.loads(seal.read_text())
            if not data.get('passed') or data.get('source_commit') != c.source_identity():
                raise RuntimeError('Lane qualification failed or belongs to another source')
            return data
        path = self.dir / 'gates.json'
        gates = json.loads(path.read_text()) if path.exists() else {}

        def gate(name, fn):
            if gates.get(name, {}).get('passed'):
                return
            self.record('qualifying', gate=name)
            result = fn()
            gates[name] = dict(passed=True, result=result, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            write(path, gates)
        py = sys.executable

        def parent():
            identity = c.parent_identity()
            checkpoint = Path(c.parent['weights_checkpoint'])
            digest = sha(checkpoint / 'pytorch_model.bin')
            if digest != c.parent['weights_sha256'] or digest != c.parent['huggingface']['lfs_sha256']:
                raise RuntimeError(f'S8N final weights {digest} differ from the declared/published SHA-256')
            if json.loads((checkpoint / 'navigation.json').read_text())['config'] != identity['config']:
                raise RuntimeError('S8N final navigation config differs from its bound identity')
            state = json.loads((Path(c.parent['run']) / 'train/checkpoint-3852/trainer_state.json').read_text())
            if state['global_step'] != state['max_steps'] or state['global_step'] != base.EXPECTED_STEPS:
                raise RuntimeError('S8N did not complete its epoch')
            manifests = {}
            for arm, m in c.data['manifests'].items():
                with open(m['path']) as f:
                    episodes = sum(1 for line in f if line.strip())
                if sha(m['path']) != m['sha256'] or episodes != m['episodes']:
                    raise RuntimeError(f'{arm}: manifest differs from its declaration')
                manifests[arm] = dict(sha256=m['sha256'], episodes=episodes)
            return dict(weights_sha256=digest, configs=prepare(c), manifests=manifests, schedule=c.schedule,
                        policy_sha256=c.run_identity(self.arm)['policy_sha256'])
        gate('parent', parent)
        gate('cpu-suite', lambda: self.run_logged('cpu-suite', [py, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                                                 *CPU_SUITE]) or 'pass')

        def init():
            out = c.shared / 'init' / self.arm
            result_path = out / 'result.json'
            if not result_path.exists():
                if out.exists():
                    raise RuntimeError(f'Preserve interrupted init run {out} before retry')
                self.run_logged(f'init-{self.arm}', train_command(
                    c, self.arm, out, updates=INIT_UPDATES, profile_only=True, host_guard=self.host_guard),
                    env_extra=thread_env(c.data['threads']))
                check = self.check_profile(out / 'train', INIT_UPDATES)['check']
                if json.loads((out / 'train/resolved_config.json').read_text()) != c.resolve(self.arm):
                    raise RuntimeError(f'{self.arm}: init config differs from production')
                report = json.loads((out / 'train/weights_initialization.json').read_text())
                parent_training = c.parent_identity()['config']['training']
                expected = dict(initialization='strict_weights_only', weights_sha256=c.parent['weights_sha256'],
                                optimizer='fresh', scheduler='fresh', resumed_from=None,
                                schedule_changes={k: [parent_training[k], c.schedule[k]] for k in SCHEDULE_KEYS})
                for key, value in expected.items():
                    if report.get(key) != value:
                        raise RuntimeError(f'weights initialization {key}={report.get(key)!r}, expected {value!r}')
                write(result_path, dict(losses=check['losses'], grad_norms=check['grad_norms'],
                                        peak_reserved_gib=check['peak_reserved_gib'], weights_initialization=report))
            results, deadline = {}, time.time() + 3 * 3600
            for arm in ARMS:
                other = c.shared / 'init' / arm
                while not (other / 'result.json').exists():
                    if time.time() > deadline:
                        raise RuntimeError(f'{arm}: no init result')
                    self.record('waiting-init', arm=arm)
                    time.sleep(60)
                results[arm] = dict(json.loads((other / 'result.json').read_text()),
                                    plan=exposure_plan(other / 'train', range(INIT_UPDATES)))
            log = json.loads((Path(c.parent['run']) / f'train/checkpoint-{base.EXPECTED_STEPS}/trainer_state.json')
                             .read_text())
            comparison = compare_init(results, {r['step']: r for r in log['log_history'] if 'loss' in r}[1]['loss'])
            write(self.dir / 'init_comparison.json', comparison)
            if not comparison['passed']:
                raise RuntimeError(f'Init check failed: {comparison["problems"]}')
            return comparison
        gate('init', init)
        write(seal, dict(passed=True, source_commit=c.source_identity(), gates=gates))
        self.record('qualified')
        return json.loads(seal.read_text())

    def train_arm(self, arm):
        c = self.campaign
        identity = dict(c.run_identity(arm), qualification_sha256=sha(self.dir / 'QUALIFIED.json'))
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
        cfg, steps = identity['config'], c.schedule['expected_total_steps']
        if run_complete(run, cfg, steps):
            return 'complete'
        threads = c.data['threads']
        for attempt in range(1, 4):
            if self.foreign():
                raise RuntimeError(f'Foreign compute step in this allocation: {self.foreign()}')
            with (run / 'launcher.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.record('training', arm=arm, attempt=attempt, threads=threads, run=str(run))
                self.run_logged(f'{arm}-production', train_command(
                    c, arm, run, production=True, host_guard=self.host_guard),
                    env_extra=thread_env(threads), check=False)
            if run_complete(run, cfg, steps):
                removed = prune_rolling(run / 'train', steps)
                self.record('arm-complete', arm=arm, pruned=removed)
                return 'complete'
            self.record('production-attempt-failed', arm=arm, attempt=attempt, resources=latest_resources(run))
            time.sleep(120)
        return 'failed'

    def main(self):
        if os.environ.get('SLURM_JOB_ID') != self.job or os.environ.get('SLURMD_NODENAME') != 'worker-2':
            raise RuntimeError('Lane must run inside its worker-2 allocation')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != self.gpus or os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
            raise RuntimeError(f'Lane {self.name} must see exactly GPUs {self.gpus} in PCI order')
        if os.sched_getaffinity(0) != cpu_list(self.cpus):
            raise RuntimeError(f'Lane {self.name} must run on CPUs {self.cpus}')
        self.own = self.job + '.' + os.environ['SLURM_STEP_ID']
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(15):
                write(self.dir / 'heartbeat.json', dict(unix=time.time(), step=self.own, pid=os.getpid()))
        with (self.dir / 'controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            threading.Thread(target=heartbeat, daemon=True).start()
            threading.Thread(target=self.relieve_cache, args=(stop,), daemon=True).start()
            try:
                self.campaign.source_identity()
                while self.foreign() or self.gpu_busy():
                    self.record('waiting-for-idle-gpus', foreign=self.foreign())
                    time.sleep(60)
                self.qualify()
                self.record('lane-complete', outcomes={self.arm: self.train_arm(self.arm)})
            except BaseException as exc:
                self.record('blocked', error=repr(exc)[:2000])
                raise
            finally:
                stop.set()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--lane')
    p.add_argument('--prepare', action='store_true', help='Write the configs, then check every identity (no GPU)')
    args = p.parse_args()
    campaign = Campaign(args.campaign)
    if args.prepare:
        configs = prepare(campaign)
        report = {arm: dict(policy_sha256=campaign.run_identity(arm)['policy_sha256'],
                            manifest_sha256=campaign.data['manifests'][arm]['sha256']) for arm in ARMS}
        print(json.dumps(dict(configs=configs, schedule=campaign.schedule, arms=report), indent=2))
        print('PREPARE_PASS', flush=True)
        return
    Lane(campaign, args.lane).main()


if __name__ == '__main__':
    main()
