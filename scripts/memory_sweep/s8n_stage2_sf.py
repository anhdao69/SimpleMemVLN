"""S8N stage 2 with the STOP-label fix: three one-epoch fine-tunes from S8N's final weights on two 4xH100 lanes.

S8N_DAGGER_SF            the 14,195 oracle-relabeled DAgger rollouts with the STOP-label fix (tau 1.5 m).
S8N_DAGGER_SF_REMAINING  those plus the 16,620 R2R/RxR15 training rows whose instruction DAgger did not re-collect.
S8N_DAGGER_SF_FULL       those plus all 30,815 R2R/RxR15 training rows S8N was trained on.

As in s8n_stage2, every arm starts strictly from S8N's final weights (SHA-256 bound) with a fresh AdamW and S8N's
own resolved config except the epoch schedule of its manifest: ceil(ceil(N / 4) / GAS) updates and ceil(3%)
warmup, i.e. 1,775/54, 3,852/116 (S8N's own) and 5,627/169.  Lane A trains S8N_DAGGER_SF and then
S8N_DAGGER_SF_REMAINING on GPUs 0-3 with socket-0 CPUs; lane B trains S8N_DAGGER_SF_FULL on GPUs 4-7 with socket-1
CPUs.

A second campaign version, ``s8n_stage2_stopfix_scalevln_campaign_v1``, trains one more arm the same way:

S8N_DAGGER_SF_REMAINING_SCALEVLN50K  S8N_DAGGER_SF_REMAINING's 30,815 rows plus the 50,000 best-ranked clean
                                     ScaleVLN episodes (selection ranks 0-49,999); 80,815 episodes, 10,102/304.

Gates per lane, all before its first production: parent (S8N identity, final weights, manifests, configs),
cpu-suite, and an init run of every arm of the lane (three updates of the arm's full manifest from the bound
weights: finite losses and gradients, the production config, a strict weights-only start that changes nothing but
the schedule, and episodes drawn from the arm's own manifest).  Then production arm by arm with identity-bound
automatic resume, completion validation and pruning of rolling checkpoints; a queued arm starts once the lane's
GPUs are idle.  Control files in the lane directory: ``STOP_AFTER_CURRENT`` and ``SKIP_<ARM>``.
"""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from scripts.memory_sweep import s_campaign as base
from scripts.memory_sweep import s8n_stage2 as stage2
from scripts.memory_sweep.s8n_followup import config_diff, exposure_plan, thread_env
from scripts.memory_sweep.s8n_stage2 import INIT_UPDATES, PARENT, SCHEDULE_KEYS, cpu_list, epoch_schedule, \
    train_command
from scripts.memory_sweep.s_campaign import WORLD, latest_resources, prune_rolling, run_complete, sha, write

VERSION = 's8n_stage2_stopfix_campaign_v1'
ARMS = {'S8N_DAGGER_SF': 'dagger_oracle_stopfix',
        'S8N_DAGGER_SF_REMAINING': 'dagger_oracle_stopfix+r2r_rxr15_not_recollected',
        'S8N_DAGGER_SF_FULL': 'dagger_oracle_stopfix+r2r_rxr15_full'}
SCALEVLN_VERSION = 's8n_stage2_stopfix_scalevln_campaign_v1'
SCALEVLN_ARMS = {'S8N_DAGGER_SF_REMAINING_SCALEVLN50K':
                 'dagger_oracle_stopfix+r2r_rxr15_not_recollected+scalevln_ranks_0_50000'}
CAMPAIGN_ARMS = {VERSION: ARMS, SCALEVLN_VERSION: SCALEVLN_ARMS}   # campaign version -> the arms it trains once
STEP_PREFIX = 's8n-stage2-sf-lane-'


# ---- small pure helpers (unit-tested) ------------------------------------
def policy_text(source, arm, schedule, version=VERSION):
    """The one-epoch policy file with the epoch schedule of the arm's manifest."""
    import yaml
    policy = yaml.safe_load((Path(source) / 'configs/vggt/g3_joint_one_epoch.yaml').read_text())
    if set(policy) != {'training'} or set(policy['training']) != {'epochs', *SCHEDULE_KEYS} \
            or policy['training']['epochs'] != 1:
        raise ValueError('Unexpected one-epoch policy')
    if arm not in CAMPAIGN_ARMS[version]:
        raise ValueError(f'{arm} is not an arm of {version}')
    policy['training'].update(schedule)
    return (f'# {version} {arm}: configs/vggt/g3_joint_one_epoch.yaml with the epoch schedule of its manifest\n'
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


def schedule_changes(parent_training, schedule):
    """What the weights-only start must report: the schedule keys whose value differs from S8N's."""
    return {k: [parent_training[k], schedule[k]] for k in SCHEDULE_KEYS if parent_training[k] != schedule[k]}


def check_init(result, plan, uids):
    """Three updates of one arm from the bound weights (pure; unit-tested).

    ``result``: losses and grad_norms per update; ``plan``: rank -> update -> episode uids (exposures);
    ``uids``: the arm's manifest.  Every rank must train its GAS episodes in each update, all from the manifest.
    """
    problems = []
    values = result['losses'] + result['grad_norms']
    if len(result['losses']) != INIT_UPDATES or len(result['grad_norms']) != INIT_UPDATES \
            or not all(math.isfinite(v) for v in values):
        problems.append(f'losses {result["losses"]} grad norms {result["grad_norms"]}')
    if sorted(plan) != list(range(WORLD)) or any(sorted(updates) != list(range(INIT_UPDATES))
                                                 for updates in plan.values()):
        problems.append(f'exposures cover ranks/updates {[(r, sorted(u)) for r, u in sorted(plan.items())]}')
    exposed = [uid for updates in plan.values() for episodes in updates.values() for uid in episodes]
    if any(len(episodes) != result['accumulation'] for updates in plan.values() for episodes in updates.values()):
        problems.append('an update did not train GAS episodes on every rank')
    foreign = sorted(set(exposed) - set(uids))
    if foreign:
        problems.append(f'episodes outside the manifest: {foreign[:5]}')
    return dict(passed=not problems, problems=problems, episodes=len(exposed))


# ---- campaign definition ---------------------------------------------------
class Campaign(stage2.Campaign):
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        if self.data['version'] not in CAMPAIGN_ARMS:
            raise ValueError('Unknown campaign version')
        self.version = self.data['version']
        self.arm_data = CAMPAIGN_ARMS[self.version]
        self.root = Path(self.data['root'])
        self.source = Path(self.data['source'])
        self.base = Path(self.data['base'])
        self.shared = self.root / 'shared'
        self.configs = self.root / 'configs'
        self.parent = self.data['parent']
        self.arms = [arm for lane in self.data['lanes'].values() for arm in lane['arms']]
        if sorted(self.arms) != sorted(self.arm_data) or set(self.data['manifests']) != set(self.arm_data):
            raise ValueError('The campaign must train each stopfix arm exactly once')
        self.schedules = {arm: epoch_schedule(m['episodes'], self.data['accumulation'])
                          for arm, m in self.data['manifests'].items()}

    def policy_path(self, arm):
        return self.configs / f'policy_{arm}.yaml'

    def expected_config(self, arm):
        cfg = copy.deepcopy(self.parent_identity()['config'])
        cfg['training'].update(self.schedules[arm])
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
        schedule = self.schedules[arm]
        if self.policy_path(arm).read_text() != policy_text(self.source, arm, schedule, self.version):
            raise ValueError(f'{arm}: policy file differs from its declaration')
        cfg = self.resolve(arm)
        expected = self.expected_config(arm)
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
        return dict(campaign=self.version, arm=arm, parent=PARENT, config=cfg, source_commit=self.source_identity(),
                    manifest=manifest['path'], manifest_sha256=manifest['sha256'], episodes=manifest['episodes'],
                    data=self.arm_data[arm], model=self.data['model'], memory_sha256=sha(self.arm_memory_path(arm)),
                    runtime_sha256=sha(self.configs / 'runtime.yaml'), policy_sha256=sha(self.policy_path(arm)),
                    parent_identity_sha256=self.parent['identity_sha256'],
                    init_weights_checkpoint=self.parent['weights_checkpoint'],
                    init_weights_sha256=self.parent['weights_sha256'], schedule=schedule, sampler_epoch_offset=0)


def prepare(campaign):
    """Byte copies of S8N's memory and runtime files and one policy file per arm (idempotent)."""
    import shutil
    c, parent = campaign, campaign.parent_identity()
    c.configs.mkdir(parents=True, exist_ok=True)
    for name, source, target in (('memory', c.parent['memory'], c.arm_memory_path(PARENT)),
                                 ('runtime', c.parent['runtime'], c.configs / 'runtime.yaml')):
        if not target.exists():
            shutil.copy2(source, target)
        if sha(target) != parent[f'{name}_sha256'] or sha(source) != parent[f'{name}_sha256']:
            raise ValueError(f'{target} is not S8N\'s {name} file')
    for arm in c.arm_data:
        path, text = c.policy_path(arm), policy_text(c.source, arm, c.schedules[arm], c.version)
        if path.exists():
            if path.read_text() != text:
                raise ValueError(f'{path} exists with different content')
        else:
            path.write_text(text)
    return {p.name: sha(p) for p in sorted(c.configs.iterdir()) if p.is_file()}


def manifest_uids(path):
    with open(path) as f:
        return [json.loads(line)['episode_uid'] for line in f if line.strip()]


# ---- lane controller ---------------------------------------------------------
class Lane(stage2.Lane):
    def __init__(self, campaign, name):
        base.Lane.__init__(self, campaign, name)
        self.arms = list(self.spec['arms'])
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError(f'Lane {name} must queue distinct arms')
        self.arm = self.arms[0]                      # the arm whose frames cache relief drops
        self.gpus, self.cpus = self.spec['gpus'], self.spec['cpus']

    def foreign(self):
        lines = subprocess.check_output(['squeue', '--steps', '-h', '-j', self.job, '-o', '%i %j'],
                                        text=True).splitlines()
        return foreign_steps(lines, self.own, self.job)

    def wait_idle(self, **fields):
        """Other work in the allocation or on this lane's GPUs (e.g. the previous arm's ranks) delays a start."""
        while (foreign := self.foreign()) or self.gpu_busy():
            self.record('waiting-for-idle-gpus', foreign=foreign, **fields)
            time.sleep(60)

    def qualify(self):
        c = self.campaign
        seal = self.dir / 'QUALIFIED.json'
        if seal.exists():
            data = json.loads(seal.read_text())
            if not data.get('passed') or data.get('source_commit') != c.source_identity() \
                    or data.get('arms') != self.arms:
                raise RuntimeError('Lane qualification failed or belongs to another source or queue')
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
            for arm in self.arms:
                m = c.data['manifests'][arm]
                uids = manifest_uids(m['path'])
                if sha(m['path']) != m['sha256'] or len(uids) != m['episodes'] or len(set(uids)) != len(uids):
                    raise RuntimeError(f'{arm}: manifest differs from its declaration')
                manifests[arm] = dict(sha256=m['sha256'], episodes=len(uids))
            return dict(weights_sha256=digest, configs=prepare(c), manifests=manifests,
                        schedules={arm: c.schedules[arm] for arm in self.arms},
                        policy_sha256={arm: c.run_identity(arm)['policy_sha256'] for arm in self.arms})
        gate('parent', parent)
        gate('cpu-suite', lambda: self.run_logged('cpu-suite', [py, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                                                 *stage2.CPU_SUITE]) or 'pass')

        def init(arm):
            out = c.shared / 'init' / arm
            result_path = out / 'result.json'
            if not result_path.exists():
                if out.exists():
                    raise RuntimeError(f'Preserve interrupted init run {out} before retry')
                self.arm = arm
                self.run_logged(f'init-{arm}', train_command(
                    c, arm, out, updates=INIT_UPDATES, profile_only=True, host_guard=self.host_guard),
                    env_extra=thread_env(c.data['threads']))
                check = self.check_profile(out / 'train', INIT_UPDATES)['check']
                if json.loads((out / 'train/resolved_config.json').read_text()) != c.resolve(arm):
                    raise RuntimeError(f'{arm}: init config differs from production')
                report = json.loads((out / 'train/weights_initialization.json').read_text())
                expected = dict(initialization='strict_weights_only', weights_sha256=c.parent['weights_sha256'],
                                optimizer='fresh', scheduler='fresh', resumed_from=None,
                                schedule_changes=schedule_changes(c.parent_identity()['config']['training'],
                                                                  c.schedules[arm]))
                for key, value in expected.items():
                    if report.get(key) != value:
                        raise RuntimeError(f'{arm}: weights initialization {key}={report.get(key)!r}, '
                                           f'expected {value!r}')
                write(result_path, dict(losses=check['losses'], grad_norms=check['grad_norms'],
                                        peak_reserved_gib=check['peak_reserved_gib'], weights_initialization=report))
            result = dict(json.loads(result_path.read_text()), accumulation=c.data['accumulation'])
            verdict = check_init(result, exposure_plan(out / 'train', range(INIT_UPDATES)),
                                 manifest_uids(c.manifest(arm)))
            if not verdict['passed']:
                raise RuntimeError(f'{arm}: init check failed: {verdict["problems"]}')
            return dict(verdict, losses=result['losses'], grad_norms=result['grad_norms'],
                        peak_reserved_gib=result['peak_reserved_gib'])
        for arm in self.arms:
            gate(f'init-{arm}', lambda arm=arm: init(arm))
        write(seal, dict(passed=True, source_commit=c.source_identity(), arms=self.arms, gates=gates))
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
        cfg, steps = identity['config'], c.schedules[arm]['expected_total_steps']
        if run_complete(run, cfg, steps):
            return 'complete'
        self.arm, threads = arm, c.data['threads']
        for attempt in range(1, 4):
            self.wait_idle(arm=arm, attempt=attempt)
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

    def run_arms(self):
        """Arms in lane order; one arm's failure does not idle the lane."""
        outcomes = {}
        for arm in self.arms:
            if self.control('STOP_AFTER_CURRENT'):
                self.record('stopped-by-request', outcomes=outcomes)
                return outcomes
            if self.control(f'SKIP_{arm}'):
                outcomes[arm] = 'skipped'
                continue
            try:
                outcomes[arm] = self.train_arm(arm)
            except Exception as exc:
                outcomes[arm] = f'error: {exc!r}'[:500]
                self.record('arm-error', arm=arm, error=repr(exc)[:2000])
        self.record('lane-complete', outcomes=outcomes)
        return outcomes

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
                self.wait_idle()
                self.qualify()
                self.run_arms()
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
        report = {arm: dict(schedule=campaign.schedules[arm], policy_sha256=campaign.run_identity(arm)['policy_sha256'],
                            manifest_sha256=campaign.data['manifests'][arm]['sha256']) for arm in campaign.arm_data}
        print(json.dumps(dict(configs=configs, arms=report), indent=2))
        print('PREPARE_PASS', flush=True)
        return
    Lane(campaign, args.lane).main()


if __name__ == '__main__':
    main()
