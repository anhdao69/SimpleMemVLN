"""Queued follow-ups of the completed S8N arm (M5 GDN S sweep) on the two 4xH100 lanes.

S8N_E2     one more epoch from S8N's final weights (SHA-256 bound, strict weights-only start):
           fresh AdamW, the identical warmup/cosine schedule and the seed's second-epoch data order.
S8N_LR1E5  S8N from the original Qwen3.5-4B with backbone/classifier LR 1e-5.
S8N_MUON   S8N from the original Qwen3.5-4B at LR 5e-6 with ZeRO-partitioned Muon on the hidden-layer
           matrices; embeddings, LM head, norms, gates and conv filters keep the same CPU AdamW.

Each arm's resolved config must equal S8N's bound config except the declared training keys, built
from S8N's own memory and runtime files (byte copies, SHA-256 checked).  Lane B qualifies once and
seals: CPU suite, a real-DeepSpeed ZeRO-2 offload Muon check, first-update reproduction of S8N's
logged production on this source (every arm), and fresh-process Muon recovery including the
optimizer state.  Lane A waits for the seal.  Both lanes wait for their L-campaign predecessor.
Per arm: longest-RxR smoke, then production with identity-bound automatic resume, completion
validation and pruning of rolling checkpoints, exactly as for the S arms.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import itertools
import json
import os
from pathlib import Path
import sys
import threading
import time

from scripts.memory_sweep import l_campaign
from scripts.memory_sweep import s_campaign as base
from scripts.memory_sweep.s_campaign import WORLD, latest_resources, prune_rolling, ready, run_complete, sha, write

VERSION = 'm5_s8n_followup_campaign_v1'
PARENT = 'S8N'
MUON = dict(name='muon_zero_offload_v1', scope='language_model_layers_2d', momentum=0.95, nesterov=True,
            ns_steps=5, ns_coefficients=[3.4445, -4.775, 2.0315], eps=1.0e-07, adjust_lr_fn='match_rms_adamw',
            group_numel=268435456)
# The training keys each arm changes relative to S8N.  Threads only set CPU parallelism (S8N ran with 4).
ARMS = {
    'S8N_E2': dict(training={}, continue_parent=True, threads=4),
    'S8N_LR1E5': dict(training=dict(backbone_lr=1.0e-05, classifier_lr=1.0e-05), threads=4),
    'S8N_MUON': dict(training=dict(matrix_optimizer=MUON), threads=8),
}
REFERENCE = 'S8N_REF'           # S8N's own config on this source (reproduction gate only)
REFERENCE_SPEC = dict(training={}, threads=4)
QUALIFICATION_GATES = ('parent', 'cpu-suite', 'zero-muon-ds', 'reproduction', 'recovery')
CPU_SUITE = ('tests/train', *l_campaign.CPU_SUITE, 'tests/vln/test_training.py', 'tests/vln/test_campaign.py')
# Updates 1-2 hold identical weights (LR 0 at update 1): forward is bitwise, backward varies ~5e-4 run to run.
# Update 3 follows Adam's first, sign-like step, which amplifies that noise: two identical S8N_REF runs gave
# losses 0.51269/0.50925 and S8N 0.50726 (2026-10-25), so update 3 only bounds gross optimizer errors.
LOSS_RTOL, GRAD_NORM_RTOL, STEPPED_RTOL, REPRODUCED_UPDATES = 2e-4, 2e-3, 3e-2, 3


def arm_spec(arm):
    return REFERENCE_SPEC if arm == REFERENCE else ARMS[arm]


def thread_env(threads):
    return dict(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))


def config_diff(actual, expected, prefix=''):
    """Dotted paths where two resolved configs differ."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return [d for k in sorted(set(actual) | set(expected), key=str)
                for d in config_diff(actual.get(k, '<absent>'), expected.get(k, '<absent>'), f'{prefix}{k}.')]
    return [] if actual == expected else [prefix.rstrip('.')]


def policy_text(source, arm):
    """The one-epoch policy file plus the arm's declared training keys."""
    import yaml
    policy = yaml.safe_load((Path(source) / 'configs/vggt/g3_joint_one_epoch.yaml').read_text())
    if set(policy) != {'training'}:
        raise ValueError('Unexpected sections in the one-epoch policy')
    policy['training'].update(copy.deepcopy(arm_spec(arm)['training']))
    return (f'# {VERSION} {arm}: configs/vggt/g3_joint_one_epoch.yaml plus the arm\'s declared training keys\n'
            + yaml.safe_dump(policy, sort_keys=False))


def working_set_gib(cgroup):
    """The supervisor's host measure: job cgroup usage minus inactive file cache."""
    stat = dict(line.split() for line in (Path(cgroup) / 'memory.stat').read_text().splitlines())
    return (int((Path(cgroup) / 'memory.current').read_text()) - int(stat['inactive_file'])) / 2**30


def drop_cache(path):
    """posix_fadvise DONTNEED: clean cached pages are dropped, the file is untouched."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return os.fstat(fd).st_size
    finally:
        os.close(fd)


def cached_paths(manifest, trees):
    """Every training frame of the manifest, then every regular file under the given trees."""
    with open(manifest) as f:
        for line in f:
            for step in json.loads(line)['steps']:
                yield step['rgb_path']
    for root in trees:
        for directory, _, names in os.walk(root):
            for name in names:
                path = os.path.join(directory, name)
                if not os.path.islink(path):
                    yield path


def exposure_plan(train_dir, updates=None):
    """rank -> update -> ordered episode uids; rows replayed after a resume collapse onto the originals."""
    plan = {}
    for rank in range(WORLD):
        rows = {}
        for line in (Path(train_dir) / f'exposures_rank{rank}.jsonl').read_text().splitlines():
            row = json.loads(line)
            if updates is None or row['update'] in updates:
                rows.setdefault(row['update'], {})[row['episode_uid']] = None
        plan[rank] = {update: list(uids) for update, uids in rows.items()}
    return plan


def compare_reproduction(parent_log, parent_plan, runs):
    """First updates on this source against S8N's logged production (pure; unit-tested).

    ``parent_log``: step -> HF log row of S8N; ``parent_plan``: S8N's exposure plan of the same updates;
    ``runs``: arm -> dict(losses, grad_norms, plan).  S8N_REF must reproduce S8N's losses and gradient
    norms at updates 1-2 (identical weights) and stay within a band at update 3; every first-epoch arm
    must reproduce its data order.  The LR is 0 at update 1, so the LR1E5 and Muon arms still hold the
    original weights at update 2 and must match S8N_REF there.  S8N_E2 must start from the trained
    weights (loss far below the original model's) on a different data order.
    """
    def close(a, b, rtol):
        return abs(a - b) <= rtol * abs(b)
    problems = []
    reference = runs[REFERENCE]
    for i in range(REPRODUCED_UPDATES):
        row = parent_log[i + 1]
        loss_rtol, norm_rtol = (LOSS_RTOL, GRAD_NORM_RTOL) if i < 2 else (STEPPED_RTOL, STEPPED_RTOL)
        if not close(reference['losses'][i], row['loss'], loss_rtol):
            problems.append(f'update {i + 1} loss {reference["losses"][i]} != S8N {row["loss"]}')
        if not close(reference['grad_norms'][i], row['grad_norm'], norm_rtol):
            problems.append(f'update {i + 1} grad norm {reference["grad_norms"][i]} != S8N {row["grad_norm"]}')
    for arm, run in runs.items():
        if len(run['losses']) != REPRODUCED_UPDATES:
            problems.append(f'{arm}: {len(run["losses"])} logged updates')
            continue
        if ARMS.get(arm, {}).get('continue_parent'):
            if run['plan'] == parent_plan:
                problems.append(f'{arm}: repeats the first-epoch data order')
            if not run['losses'][0] < 0.5 * parent_log[1]['loss']:
                problems.append(f'{arm}: update-1 loss {run["losses"][0]} is not from trained weights')
            continue
        if run['plan'] != parent_plan:
            problems.append(f'{arm}: data order differs from S8N')
        if arm != REFERENCE:
            for i in (0, 1):
                if not close(run['losses'][i], reference['losses'][i], LOSS_RTOL):
                    problems.append(f'{arm}: update {i + 1} loss {run["losses"][i]} != S8N_REF')
                if not close(run['grad_norms'][i], reference['grad_norms'][i], GRAD_NORM_RTOL):
                    problems.append(f'{arm}: update {i + 1} grad norm {run["grad_norms"][i]} != S8N_REF')
    return dict(passed=not problems, problems=problems,
                parent={s: dict(loss=parent_log[s]['loss'], grad_norm=parent_log[s]['grad_norm'])
                        for s in range(1, REPRODUCED_UPDATES + 1)},
                runs={arm: dict(losses=r['losses'], grad_norms=r['grad_norms']) for arm, r in runs.items()})


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

    def arm_memory_path(self, arm, role='training'):
        return self.configs / f'arm_{PARENT}.yaml'

    def policy_path(self, arm):
        return self.configs / f'policy_{arm}.yaml'

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
        if (identity['arm'] != PARENT or identity['manifest_sha256'] != self.data['manifest_sha256']
                or identity['model'] != self.data['model']):
            raise ValueError('S8N identity differs from the campaign manifest/model')
        return identity

    def expected_config(self, arm):
        cfg = copy.deepcopy(self.parent_identity()['config'])
        cfg['training'].update(copy.deepcopy(arm_spec(arm)['training']))
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
        if self.policy_path(arm).read_text() != policy_text(self.source, arm):
            raise ValueError(f'{arm}: policy file differs from its declaration')
        cfg = self.resolve(arm)
        expected = self.expected_config(arm)
        if cfg != expected:
            raise ValueError(f'{arm}: resolved config differs from S8N at {config_diff(cfg, expected)}')
        validate_config(copy.deepcopy(cfg), WORLD)
        spec = parse_episodic_spec(cfg)
        if spec.score != PARENT or spec.version != 'episodic_window_gdn_sweep_v1':
            raise ValueError(f'{arm}: config declares {spec.score}')
        muon = parse_matrix_optimizer(cfg['training'])
        if muon != (MUON if arm == 'S8N_MUON' else None):
            raise ValueError(f'{arm}: unexpected matrix optimizer {muon}')
        seal = self.shared / 'QUALIFIED.json'
        identity = dict(campaign=VERSION, arm=arm, parent=PARENT, config=cfg, source_commit=self.source_identity(),
                        manifest_sha256=self.data['manifest_sha256'], model=self.data['model'],
                        memory_sha256=sha(self.arm_memory_path(arm)), runtime_sha256=sha(self.configs / 'runtime.yaml'),
                        policy_sha256=sha(self.policy_path(arm)), parent_identity_sha256=self.parent['identity_sha256'],
                        qualification_sha256=sha(seal) if seal.exists() else None)
        if arm_spec(arm).get('continue_parent'):
            identity.update(init_weights_checkpoint=self.parent['weights_checkpoint'],
                            init_weights_sha256=self.parent['weights_sha256'], sampler_epoch_offset=1)
        return identity


def prepare(campaign):
    """Byte copies of S8N's memory and runtime files and the declared policy files (idempotent)."""
    import shutil
    c, parent = campaign, campaign.parent_identity()
    c.configs.mkdir(parents=True, exist_ok=True)
    for name, source, target in (('memory', c.parent['memory'], c.arm_memory_path(PARENT)),
                                 ('runtime', c.parent['runtime'], c.configs / 'runtime.yaml')):
        if not target.exists():
            shutil.copy2(source, target)
        if sha(target) != parent[f'{name}_sha256'] or sha(source) != parent[f'{name}_sha256']:
            raise ValueError(f'{target} is not S8N\'s {name} file')
    for arm in (REFERENCE, *ARMS):
        path, text = c.policy_path(arm), policy_text(c.source, arm)
        if path.exists():
            if path.read_text() != text:
                raise ValueError(f'{path} exists with different content')
        else:
            path.write_text(text)
    return {p.name: sha(p) for p in sorted(c.configs.iterdir()) if p.is_file()}


# ---- commands ----------------------------------------------------------------
def train_command(campaign, arm, out, **kwargs):
    """The S-campaign command with the arm's policy file and, for a continuation, the parent weights."""
    command = base.train_command(campaign, arm, out, **kwargs)
    index = command.index('--policy_config') + 1
    if command[index] != 'configs/vggt/g3_joint_one_epoch.yaml':
        raise RuntimeError('Unexpected policy argument')
    command[index] = str(campaign.policy_path(arm))
    if arm_spec(arm).get('continue_parent'):
        command += ['--init-weights-checkpoint', campaign.parent['weights_checkpoint'],
                    '--init-weights-sha256', campaign.parent['weights_sha256'], '--sampler-epoch-offset', '1']
    return command


# ---- lane controller ---------------------------------------------------------
class Lane(l_campaign.Lane):
    def relieve_cache(self, stop):
        """Drop clean frame/tree cache whenever the job's working set nears the host guard.

        The supervisor stops a run once the job's working set exceeds the guard. Frames that the other
        lane re-reads become active file cache of this job and count toward it (L campaign, 2026-10-25).
        Dropping them only costs a later re-read.
        """
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
                paths, files = cached_paths(self.campaign.data['manifest'], relief['trees']), 0
                with ThreadPoolExecutor(relief['threads']) as pool:
                    while batch := list(itertools.islice(paths, 4096)):   # bounded: no 3M-future backlog
                        files += sum(1 for _ in pool.map(drop_cache, batch))
                row = dict(before_gib=before, after_gib=working_set_gib(cgroup), limit_gib=limit, files=files,
                           seconds=time.time() - started)
            except Exception as exc:              # relief is best effort; never disturb the controller
                row = dict(error=repr(exc)[:500])
            with log.open('a') as f:
                f.write(json.dumps(dict(row, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))) + '\n')

    def main(self):
        stop = threading.Event()
        threading.Thread(target=self.relieve_cache, args=(stop,), daemon=True).start()
        try:
            super().main()
        finally:
            stop.set()

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

        def parent():
            identity = c.parent_identity()
            checkpoint = Path(c.parent['weights_checkpoint'])
            if sha(checkpoint / 'pytorch_model.bin') != c.parent['weights_sha256']:
                raise RuntimeError('S8N final weights differ from the declared SHA-256')
            if json.loads((checkpoint / 'navigation.json').read_text())['config'] != identity['config']:
                raise RuntimeError('S8N final navigation config differs from its bound identity')
            state = json.loads((Path(c.parent['run']) / 'train/checkpoint-3852/trainer_state.json').read_text())
            if state['global_step'] != state['max_steps'] or state['global_step'] != base.EXPECTED_STEPS:
                raise RuntimeError('S8N did not complete its epoch')
            if sha(c.data['manifest']) != c.data['manifest_sha256']:
                raise RuntimeError('Manifest differs')
            arms = {arm: dict(policy_sha256=c.run_identity(arm)['policy_sha256'],
                              changed=sorted(arm_spec(arm)['training'])) for arm in (REFERENCE, *ARMS)}
            return dict(weights_sha256=c.parent['weights_sha256'], manifest_sha256=c.data['manifest_sha256'],
                        configs=prepare(c), arms=arms)
        gate('parent', parent)
        gate('cpu-suite', lambda: self.run_logged('cpu-suite', [py, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                                                                 *CPU_SUITE]) or 'pass')

        def zero_muon_ds():
            out = shared / 'zero_muon_ds'
            if out.exists():
                raise RuntimeError(f'Preserve interrupted ZeRO Muon check {out} before retry')
            self.run_logged('zero-muon-ds', ['torchrun', '--standalone', f'--nproc_per_node={WORLD}', '-m',
                                             'scripts.memory_sweep.zero_muon_ds_check', '--out', str(out)],
                            env_extra=thread_env(8))
            report = json.loads((out / 'zero_muon_ds_check.json').read_text())
            if not report['passed']:
                raise RuntimeError(f'ZeRO Muon check failed: {report}')
            return {k: report[k] for k in ('muon_groups', 'steps_ok', 'clipping_exercised', 'resume_ok', 'resume')}
        gate('zero-muon-ds', zero_muon_ds)

        def reproduction():
            parent_run = Path(c.parent['run']) / 'train'
            log = json.loads((parent_run / f'checkpoint-{base.EXPECTED_STEPS}/trainer_state.json').read_text())
            parent_log = {r['step']: r for r in log['log_history'] if 'loss' in r}
            first = range(REPRODUCED_UPDATES)
            parent_plan = exposure_plan(parent_run, first)
            runs = {}
            for arm in (REFERENCE, *ARMS):
                out = shared / 'reproduction' / arm
                result_path = out / 'result.json'
                if not result_path.exists():
                    if out.exists():
                        raise RuntimeError(f'Preserve interrupted reproduction {out} before retry')
                    self.run_logged(f'reproduction-{arm}', train_command(
                        c, arm, out, updates=REPRODUCED_UPDATES, profile_only=True, host_guard=self.host_guard),
                        env_extra=thread_env(arm_spec(arm)['threads']))
                    check = self.check_profile(out / 'train', REPRODUCED_UPDATES)['check']
                    if json.loads((out / 'train/resolved_config.json').read_text()) != c.resolve(arm):
                        raise RuntimeError(f'{arm}: reproduction config differs from production')
                    result = dict(losses=check['losses'], grad_norms=check['grad_norms'],
                                  peak_reserved_gib=check['peak_reserved_gib'])
                    if arm_spec(arm).get('continue_parent'):
                        report = json.loads((out / 'train/weights_initialization.json').read_text())
                        if (report['weights_sha256'] != c.parent['weights_sha256'] or report['optimizer'] != 'fresh'
                                or report['initialization'] != 'strict_weights_only'):
                            raise RuntimeError(f'{arm}: unexpected weights initialization {report}')
                        result['weights_initialization'] = report
                    write(result_path, result)
                runs[arm] = dict(json.loads(result_path.read_text()), plan=exposure_plan(out / 'train', first))
            comparison = compare_reproduction(parent_log, parent_plan, runs)
            write(shared / 'reproduction/comparison.json', comparison)
            if not comparison['passed']:
                raise RuntimeError(f'Reproduction failed: {comparison["problems"]}')
            return comparison
        gate('reproduction', reproduction)

        def recovery():
            arm, root = 'S8N_MUON', shared / 'recovery'
            fixture = Path(c.data['fixtures']) / 'mixed_tail.jsonl'
            diagnostic = self.diagnostic_runtime()
            for name, extra in (('reference', dict(save_every=2, first_save=3)),
                                ('resumed', dict(save_every=2, resume=root / 'reference/train/checkpoint-2',
                                                 first_save=3))):
                out = root / name
                if out.exists():
                    raise RuntimeError(f'Preserve interrupted recovery evidence {out} before retry')
                self.run_logged(f'recovery-{name}', train_command(c, arm, out, manifest=fixture, updates=3,
                                                                  runtime=diagnostic, host_guard=self.host_guard,
                                                                  **extra),
                                env_extra=thread_env(ARMS[arm]['threads']))
                self.check_profile(out / 'train', 3, 1 if name == 'reference' else 3)
                if json.loads((out / 'train/resolved_config.json').read_text()) != c.resolve(arm, diagnostic):
                    raise RuntimeError('Recovery diagnostic config differs')
            self.run_logged('recovery-compare', [py, '-m', 'scripts.memory_sweep.compare_recovery',
                                                 '--root', str(root), '--out', str(root / 'comparison.json')])
            self.run_logged('recovery-optimizer', [py, '-m', 'scripts.memory_sweep.compare_optimizer_states',
                                                   '--reference', str(root / 'reference/train/checkpoint-3'),
                                                   '--resumed', str(root / 'resumed/train/checkpoint-3'),
                                                   '--out', str(root / 'optimizer_states.json')])
            return dict(weights=json.loads((root / 'comparison.json').read_text()),
                        optimizer=json.loads((root / 'optimizer_states.json').read_text())['summary'])
        gate('recovery', recovery)
        if set(gates) != set(QUALIFICATION_GATES) or not all(g['passed'] for g in gates.values()):
            raise RuntimeError('Incomplete shared qualification')
        write(seal, dict(passed=True, source_commit=source, gates=gates))
        self.record('qualified')
        return json.loads(seal.read_text())

    # -- per arm --
    def smoke(self, arm, run, threads):
        """Two updates of the complete longest RxR episode under the arm's exact configuration."""
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
            env_extra=thread_env(threads))
        result = self.check_profile(out / 'train', 2)
        if json.loads((out / 'train/resolved_config.json').read_text()) != self.campaign.resolve(arm):
            raise RuntimeError(f'{arm}: smoke config differs from production')
        result.update(threads=threads, **base.optimizer_step_estimate(out / 'train'))
        write(result_path, result)
        return result

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
        threads = ARMS[arm]['threads']
        self.smoke(arm, run, threads)
        for attempt in range(1, 4):
            if not ready(self.active_steps(), self.own, self.job):
                raise RuntimeError('Foreign compute step in this allocation')
            with (run / 'launcher.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.record('training', arm=arm, attempt=attempt, threads=threads, run=str(run))
                self.run_logged(f'{arm}-production', train_command(
                    c, arm, run, production=True, host_guard=self.host_guard),
                    env_extra=thread_env(threads), check=False)
            if run_complete(run, cfg):
                removed = prune_rolling(run / 'train')
                self.record('arm-complete', arm=arm, pruned=removed)
                return 'complete'
            self.record('production-attempt-failed', arm=arm, attempt=attempt, resources=latest_resources(run))
            time.sleep(120)
        return 'failed'


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
                            changed=sorted(arm_spec(arm)['training'])) for arm in (REFERENCE, *ARMS)}
        print(json.dumps(dict(configs=configs, arms=report), indent=2))
        print('PREPARE_PASS', flush=True)
        return
    Lane(campaign, args.lane).main()


if __name__ == '__main__':
    main()
