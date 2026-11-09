"""S8N recent-window (R) / episodic-bank (B) training ablation on newton (4xH100 interactive allocations).

Six arms are retrained for one epoch with exactly the recipe of the completed S8N (R4/B1200) and R4B0
(R4/B0) arms of the M5 GDN S sweep (training source ``dabf4ad``): fresh Qwen3.5-4B@851bf6e8, joint R2R +
English RxR15deg, seed 429 and its data order, 3,852 updates, 116 warmup, cosine to 10%, LR 5e-6,
global batch 8 = 4 ranks x 1 episode x GAS2, ZeRO-2 CPU Adam, the same runtime file byte for byte.

  S8N_R4B600   S8N scorer/selector, R4,  B600  (2 frames)   guard 2048 + 4 x 384 +  600 = 4184
  S8N_R4B2400  S8N scorer/selector, R4,  B2400 (8 frames)   guard 2048 + 4 x 384 + 2400 = 5984
  R8B0         R4B0's bank-free episodic_window_v1, R8      guard 2048 + 8 x 384        = 5120
  R12B0        R4B0's bank-free episodic_window_v1, R12     guard 2048 + 12 x 384       = 6656
  S8N_R2B1200  S8N scorer/selector, R2,  B1200 (4 frames)   guard 2048 + 2 x 384 + 1200 = 4016
  S8N_R8B1200  S8N scorer/selector, R8,  B1200 (4 frames)   guard 2048 + 8 x 384 + 1200 = 6320

Every arm's memory equals its family's published memory (S8N or R4B0) except
``recent_groups_including_current``, ``episodic_budget_tokens`` and the derived ``kv_guard_tokens``; the S8N
drift floors stay the frozen training calibration of S8N, so the scorer is identical and only R/B change.
Every resolved config equals the family's published resolved config except the memory section.

One login-node watcher (``rb_watcher.py``) assigns one task per allocation and launches this module as an
``srun --overlap`` step; a lane runs exactly that task and exits.  (Lustre is mounted ``localflock``:
flock does not exclude across nodes, so all assignment decisions live in the single watcher.)
Tasks: ``qualify`` (provenance, CPU suite, configs, S8N/R4B0 first-update reproduction) and one arm
(longest-RxR smoke once, then production with automatic resume; before the allocation ends the lane asks
the trainer for a complete recovery checkpoint and stops, and another allocation resumes it).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import threading
import time

from scripts.memory_sweep import s_campaign as base
from scripts.memory_sweep.s_campaign import WORLD, sha, write

VERSION = 'm5_s8n_rb_ablation_newton_v1'
TRAINING_CODE_COMMIT = 'dabf4ad085d20512640fe0ba3bda72ad4724dc76'
EXPECTED_STEPS = base.EXPECTED_STEPS
# Worker-2 manifest r2r_rxr15_train_20260929.jsonl (5deb425d...) has this path-independent fingerprint
# (reports/vggt_five/G3_M5/implementation_report.md); newton's copy differs only in rgb_path roots.
MANIFEST_FINGERPRINT = 'f11fcceac19724e586f1b0a97828c388f619a88ee14d513bd8498856e9c2b0a9'
MANIFEST_COUNTS = dict(episodes=30815, steps=3128624, r2rce=10819, rxrce=19996)
FAMILIES = ('S8N', 'R4B0')
ARMS = {                                  # in the user's order (2026-11-09)
    'S8N_R4B600': dict(family='S8N', recent=4, budget=600),
    'S8N_R4B2400': dict(family='S8N', recent=4, budget=2400),
    'R8B0': dict(family='R4B0', recent=8, budget=0),
    'R12B0': dict(family='R4B0', recent=12, budget=0),
    'S8N_R2B1200': dict(family='S8N', recent=2, budget=1200),
    'S8N_R8B1200': dict(family='S8N', recent=8, budget=1200),
}
REFERENCES = {'S8N_REF': 'S8N', 'R4B0_REF': 'R4B0'}    # each family's own published config (gate only)
MEMORY_KEYS_CHANGED = ('recent_groups_including_current', 'episodic_budget_tokens', 'kv_guard_tokens')
PINNED = {'torch': '2.10.0+cu129', 'transformers': '5.11.0', 'fla-core': '0.5.2', 'flash-linear-attention': '0.5.2',
          'flash_attn': '2.8.3+cu.12.9.torch.2.10', 'deepspeed': '0.16.4', 'numpy': '2.4.4', 'triton': '3.6.0',
          'tokenizers': '0.22.2', 'pillow': '12.2.0'}
# Additions allowed on top of the training code commit; nothing else may differ.
ALLOWED_ADDITIONS = ('scripts/memory_sweep/rb_ablation.py', 'scripts/memory_sweep/rb_watcher.py',
                     'scripts/memory_sweep/rb_setup.py', 'tests/memory_sweep/test_rb_ablation.py',
                     'train/rb_newton_env.sh', 'reports/memory_sweep/s8n-rb-ablation-newton/')
# First-update reproduction of the published logs (tolerances of the S8N follow-up gate, a8bb0a2):
# updates 1-2 hold identical weights (LR 0 at update 1), update 3 follows Adam's first sign-like step.
LOSS_RTOL, GRAD_NORM_RTOL, STEPPED_RTOL, REPRODUCED_UPDATES = 2e-4, 2e-3, 3e-2, 3
THREADS = 4                               # S8N's and R4B0's CPU-Adam threads; 4 ranks x 4 = 16 CPUs
MAX_RESERVED_GIB = 78.0
QUALIFICATION_GATES = ('provenance', 'cpu-suite', 'configs', 'reproduction')


# ---- pure helpers (unit-tested) -----------------------------------------------
def guard(observations, recent, budget):
    return observations['max_prefix_tokens'] + recent * observations['max_step_group_tokens'] + budget


def arm_memory(family_memory, observations, recent, budget):
    """The family's memory with only R, B and the derived structural guard changed."""
    memory = copy.deepcopy(family_memory)
    memory['recent_groups_including_current'] = recent
    memory['episodic_budget_tokens'] = budget
    memory['kv_guard_tokens'] = guard(observations, recent, budget)
    changed = sorted(k for k in set(memory) | set(family_memory) if memory.get(k) != family_memory.get(k))
    if not set(changed) <= set(MEMORY_KEYS_CHANGED):
        raise ValueError(f'unexpected memory changes {changed}')
    return memory


def memory_yaml(arm, memory, family):
    import yaml
    spec = ARMS[arm]
    return (f'# {VERSION} {arm}: the published {family} memory with only R={spec["recent"]}, B={spec["budget"]}\n'
            f'# and the derived guard 2048 + {spec["recent"]} x 384 + {spec["budget"]} changed.\n'
            + yaml.safe_dump({'memory': memory}, sort_keys=False))


def config_diff(actual, expected, prefix=''):
    if isinstance(actual, dict) and isinstance(expected, dict):
        return [d for k in sorted(set(actual) | set(expected), key=str)
                for d in config_diff(actual.get(k, '<absent>'), expected.get(k, '<absent>'), f'{prefix}{k}.')]
    return [] if actual == expected else [prefix.rstrip('.')]


def manifest_fingerprint(path):
    """Path-independent fingerprint (rebase_manifest.py / scripts/vln/manifest_fingerprint.py definition)."""
    digest, episodes, steps, domains = hashlib.sha256(), 0, 0, {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            s = [(x['action_name'], PurePosixPath(x['rgb_path']).name, x.get('is_valid', True)) for x in row['steps']]
            digest.update(json.dumps([row['episode_uid'], row['instruction'], s]).encode() + b'\n')
            episodes += 1
            steps += len(s)
            domain = row['episode_uid'].split(':')[0]
            domains[domain] = domains.get(domain, 0) + 1
    return dict(fingerprint_v1=digest.hexdigest(), episodes=episodes, steps=steps, **domains)


def collapse(rows):
    """update -> last row (rows replayed after a resume or a lost tail collapse onto one record)."""
    out = {}
    for row in rows:
        out[row['update']] = row
    return out


def exposure_plan(train_dir, updates=None):
    """rank -> update -> ordered episode uids; rows replayed after a resume collapse onto the originals."""
    plan = {}
    for rank in range(WORLD):
        rows = {}
        for line in (Path(train_dir) / f'exposures_rank{rank}.jsonl').read_text().splitlines():
            row = json.loads(line)
            if updates is None or row['update'] in updates:
                rows.setdefault(row['update'], {})[row['episode_uid']] = None
        plan[rank] = {update: list(uids) for update, uids in sorted(rows.items())}
    return plan


def compare_reproduction(reference_logs, runs):
    """``reference_logs``: family -> step -> published HF log row; ``runs``: REF arm -> losses, grad_norms, plan.

    Each family's own config must reproduce its published losses/grad norms at updates 1-2 (identical weights)
    and stay within a band at update 3; both references must see the same episodes (one data order).
    """
    def close(a, b, rtol):
        return abs(a - b) <= rtol * abs(b)
    problems = []
    for arm, run in runs.items():
        log = reference_logs[REFERENCES[arm]]
        if len(run['losses']) != REPRODUCED_UPDATES:
            problems.append(f'{arm}: {len(run["losses"])} logged updates')
            continue
        for i in range(REPRODUCED_UPDATES):
            row = log[i + 1]
            loss_rtol, norm_rtol = (LOSS_RTOL, GRAD_NORM_RTOL) if i < 2 else (STEPPED_RTOL, STEPPED_RTOL)
            if not close(run['losses'][i], row['loss'], loss_rtol):
                problems.append(f'{arm}: update {i + 1} loss {run["losses"][i]} != published {row["loss"]}')
            if not close(run['grad_norms'][i], row['grad_norm'], norm_rtol):
                problems.append(f'{arm}: update {i + 1} grad norm {run["grad_norms"][i]} != published {row["grad_norm"]}')
    plans = [run['plan'] for run in runs.values()]
    if any(p != plans[0] for p in plans[1:]):
        problems.append('reference runs saw different episodes')
    return dict(passed=not problems, problems=problems,
                runs={arm: dict(losses=r['losses'], grad_norms=r['grad_norms']) for arm, r in runs.items()},
                published={f: {s: dict(loss=reference_logs[f][s]['loss'], grad_norm=reference_logs[f][s]['grad_norm'])
                               for s in range(1, REPRODUCED_UPDATES + 1)} for f in FAMILIES})


def audit_epoch(train_dir, steps=EXPECTED_STEPS, world=WORLD, slots=8, max_reserved=MAX_RESERVED_GIB):
    """Full 4-rank coverage of every update with finite losses, no CUDA OOM, the exposure plan."""
    train_dir = Path(train_dir)
    problems, peaks = [], []
    for rank in range(world):
        profile = collapse(json.loads(l) for l in (train_dir / f'profile_rank{rank}.jsonl').read_text().splitlines())
        if sorted(profile) != list(range(1, steps + 1)):
            problems.append(f'rank {rank}: {len(profile)} profiled updates')
        for row in profile.values():
            peaks.append(row['peak_reserved_gib'])
            if row.get('cuda_ooms', 0):
                problems.append(f'rank {rank} update {row["update"]}: CUDA OOM counter {row["cuda_ooms"]}')
    plan = exposure_plan(train_dir)              # exposure rows use the 0-based update index
    for rank, updates in plan.items():
        if sorted(updates) != list(range(steps)) or any(len(u) != slots // world for u in updates.values()):
            problems.append(f'rank {rank}: exposures do not cover every update with {slots // world} episodes')
    exposures = {}
    for rank in range(world):
        for line in (train_dir / f'exposures_rank{rank}.jsonl').read_text().splitlines():
            row = json.loads(line)
            exposures[(rank, row['update'], row['episode_uid'])] = row
    if not all(math.isfinite(r['loss_sum']) for r in exposures.values()):
        problems.append('nonfinite exposure loss')
    if peaks and max(peaks) > max_reserved:
        problems.append(f'GPU reserved {max(peaks):.3f} GiB > {max_reserved}')
    return dict(passed=not problems, problems=problems, peak_reserved_gib=max(peaks) if peaks else None,
                plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()), plan


def loss_correlation(a, b, start=117):
    """Pearson correlation of per-update losses after warmup (a data-order sanity metric, not a gate)."""
    keys = sorted(k for k in set(a) & set(b) if k >= start)
    if len(keys) < 10:
        return None
    x, y = [a[k] for k in keys], [b[k] for k in keys]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((p - mx) * (q - my) for p, q in zip(x, y))
    sxx, syy = sum((p - mx) ** 2 for p in x), sum((q - my) ** 2 for q in y)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else None


def deadline_actions(now, end, request_margin, kill_margin):
    """What the production watchdog does at ``now`` for an allocation ending at ``end``."""
    if end is None:
        return 'none'
    if now >= end - kill_margin:
        return 'kill'
    if now >= end - request_margin:
        return 'request'
    return 'none'


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
    """Every regular file under the given trees (few large files: staged caches, checkpoints), then every
    training frame of the manifest (3.1M small Lustre files, a slow walk)."""
    for root in trees:
        for directory, _, names in os.walk(root):
            for name in names:
                path = os.path.join(directory, name)
                if not os.path.islink(path):
                    yield path
    with open(manifest) as f:
        for line in f:
            for step in json.loads(line)['steps']:
                yield step['rgb_path']


# ---- campaign ------------------------------------------------------------------
class Campaign:
    def __init__(self, path):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        if self.data['version'] != VERSION:
            raise ValueError('Unknown campaign version')
        self.root = Path(self.data['root'])
        self.source = Path(self.data['source'])
        self.base = Path(self.data['base'])         # disk-space check path of the supervisor
        self.shared = self.root / 'shared'
        self.configs = self.root / 'configs'
        self.reference = self.root / 'reference'

    # files
    def arm_memory_path(self, arm, role='training'):
        if arm in REFERENCES:
            return self.configs / f'arm_{REFERENCES[arm]}.yaml'
        return self.configs / f'arm_{arm}.yaml'

    def family(self, arm):
        return REFERENCES[arm] if arm in REFERENCES else ARMS[arm]['family']

    def family_identity(self, family):
        return json.loads((self.reference / family / 'run_identity.json').read_text())

    def family_log(self, family):
        state = json.loads((self.reference / family / 'trainer_state.json').read_text())
        return {r['step']: r for r in state['log_history'] if 'loss' in r}

    def run_dir(self, arm):
        if arm in REFERENCES:
            return self.shared / 'reproduction' / arm
        return self.root / 'runs' / f'{arm}_seed{self.data["seed"]}_e1'

    # identities
    def source_identity(self):
        head = subprocess.check_output(['git', '-C', str(self.source), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(self.source), 'status', '--porcelain'], text=True).strip()
        if head != self.data['source_commit'] or dirty:
            raise ValueError(f'Source {head} dirty={bool(dirty)} differs from {self.data["source_commit"]}')
        changes = subprocess.check_output(['git', '-C', str(self.source), 'diff', '--name-status',
                                           TRAINING_CODE_COMMIT, head], text=True).split('\n')
        for line in filter(None, changes):
            status, path = line.split('\t', 1)
            if status != 'A' or not path.startswith(ALLOWED_ADDITIONS):
                raise ValueError(f'Source differs from training code {TRAINING_CODE_COMMIT} at {status} {path}')
        return head

    def resolve(self, arm, runtime=None):
        from qwen_vl.train.vln_runtime import resolve_config
        s = self.source
        return resolve_config(s / 'configs/episodic/science_common.yaml', s / 'configs/vln_empty.yaml',
                              self.arm_memory_path(arm), s / 'configs/vggt/g3_joint_one_epoch.yaml',
                              runtime or self.configs / 'runtime.yaml')

    def expected_config(self, arm):
        """The family's published resolved config with only the memory section replaced."""
        import yaml
        family = self.family(arm)
        cfg = json.loads((self.reference / family / 'resolved_config.json').read_text())
        if cfg != self.family_identity(family)['config']:
            raise ValueError(f'{family}: published resolved config differs from its run identity')
        memory = yaml.safe_load(self.arm_memory_path(arm).read_text())['memory']
        if arm in REFERENCES:
            if memory != cfg['memory']:
                raise ValueError(f'{arm}: memory file is not the published {family} memory')
        else:
            spec = ARMS[arm]
            if memory != arm_memory(cfg['memory'], cfg['observations'], spec['recent'], spec['budget']):
                raise ValueError(f'{arm}: memory is not {family} with only R/B/guard changed')
        cfg['memory'] = memory
        return cfg

    def run_identity(self, arm):
        from qwen_vl.contracts import validate_config
        from qwen_vl.research.episodic_contract import parse_episodic_spec
        family = self.family(arm)
        published = self.family_identity(family)
        if sha(self.configs / 'runtime.yaml') != published['runtime_sha256']:
            raise ValueError('runtime.yaml is not the published S8N/R4B0 runtime file')
        if arm in REFERENCES and sha(self.arm_memory_path(arm)) != published['memory_sha256']:
            raise ValueError(f'{arm}: memory file differs from the published {family} file')
        cfg = self.resolve(arm)
        expected = self.expected_config(arm)
        if cfg != expected:
            raise ValueError(f'{arm}: resolved config differs from {family} at {config_diff(cfg, expected)}')
        validate_config(copy.deepcopy(cfg), WORLD)
        spec = parse_episodic_spec(cfg)
        recent, budget = ((4, 1200) if family == 'S8N' else (4, 0)) if arm in REFERENCES else (
            ARMS[arm]['recent'], ARMS[arm]['budget'])
        want = (('episodic_window_gdn_sweep_v1', 'S8N') if family == 'S8N' else ('episodic_window_v1', 'none'))
        if (spec.version, spec.score, spec.recent_groups, spec.budget_tokens) != (*want, recent, budget):
            raise ValueError(f'{arm}: unexpected spec {spec}')
        seal = self.shared / 'QUALIFIED.json'
        return dict(campaign=VERSION, arm=arm, family=family, config=cfg, source_commit=self.source_identity(),
                    training_code_commit=TRAINING_CODE_COMMIT, manifest=self.data['manifest'],
                    manifest_sha256=self.data['manifest_sha256'], manifest_fingerprint_v1=MANIFEST_FINGERPRINT,
                    model=self.data['model'], memory_sha256=sha(self.arm_memory_path(arm)),
                    runtime_sha256=sha(self.configs / 'runtime.yaml'),
                    family_identity_sha256=sha(self.reference / family / 'run_identity.json'),
                    qualification_sha256=sha(seal) if seal.exists() and arm not in REFERENCES else None)

    def sealed(self):
        seal = self.shared / 'QUALIFIED.json'
        if not seal.exists():
            return None
        data = json.loads(seal.read_text())
        return data if data.get('passed') and data.get('source_commit') == self.data['source_commit'] else None


def train_command(campaign, arm, out, **kwargs):
    return base.train_command(campaign, arm, out, **kwargs)


def latest_checkpoint_step(run):
    from qwen_vl.train.recovery import completed_checkpoints
    found = completed_checkpoints(Path(run) / 'train') if (Path(run) / 'train').is_dir() else []
    return found[-1][0] if found else 0


# ---- lane: one task inside one allocation ----------------------------------------
class Lane:
    def __init__(self, campaign, job, task):
        self.campaign, self.job, self.task = campaign, str(job), task
        self.dir = campaign.root / 'lanes' / self.job
        self.dir.mkdir(parents=True, exist_ok=True)
        self.step = None
        self.end = None
        self.host_guard = campaign.data['host_guard_gib']

    def record(self, event, /, **fields):
        if 'status' in fields:                    # a field named status never collides with the event name
            fields['field_status'] = fields.pop('status')
        data = dict(status=event, unix=time.time(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    job=self.job, step=self.step, task=self.task, pid=os.getpid(), **fields)
        write(self.dir / 'state.json', data)
        for path in (self.dir / 'history.jsonl', self.campaign.root / 'events.jsonl'):
            with path.open('a') as f:
                f.write(json.dumps(data, sort_keys=True) + '\n')
        print(event, json.dumps(fields, sort_keys=True), flush=True)

    def job_end(self):
        out = subprocess.run(['squeue', '-h', '-j', self.job, '-o', '%e'], capture_output=True, text=True).stdout.strip()
        try:
            return time.mktime(time.strptime(out, '%Y-%m-%dT%H:%M:%S'))
        except ValueError:
            return self.end

    def relieve_cache(self, stop, initial_limit):
        """Drop clean frame/checkpoint/LoopWAM-stage cache whenever the job's working set nears the host guard.

        The supervisor stops a run once the job's working set (usage minus clean inactive file) exceeds the
        guard. On newton's cgroup v1 nearly all file cache is active (G3A1: 340 GiB working set at 267 GiB
        anon) and allocations arrive holding LoopWAM's staged cache (160 GiB active file on 1156199,
        2026-11-09). Dropping clean pages only costs a later re-read (s8n_followup relief, L campaign).
        """
        from qwen_vl.train.host_memory import cgroup_sample, job_cgroup
        relief = self.campaign.data['cache_relief']
        location = job_cgroup()
        limit, last = initial_limit, 0.0
        log = self.dir / 'cache_relief.jsonl'
        while True:
            try:
                before = cgroup_sample(location)['working_set_gib']
                if before >= limit and time.time() - last >= relief['cooldown_seconds']:
                    last = started = time.time()
                    trees = [*relief['trees'], str(self.campaign.root / 'runs')]
                    paths, files = cached_paths(self.campaign.data['manifest'], trees), 0
                    with ThreadPoolExecutor(relief['threads']) as pool:
                        while batch := list(itertools.islice(paths, 4096)):   # bounded: no 3M-future backlog
                            files += sum(1 for _ in pool.map(drop_cache, batch))
                    row = dict(before_gib=before, after_gib=cgroup_sample(location)['working_set_gib'],
                               limit_gib=limit, files=files, seconds=time.time() - started)
                    with log.open('a') as f:
                        f.write(json.dumps(dict(row, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))) + '\n')
            except Exception as exc:              # relief is best effort; never disturb the lane
                with log.open('a') as f:
                    f.write(json.dumps(dict(error=repr(exc)[:500], unix=time.time())) + '\n')
            limit = relief['fraction'] * self.host_guard
            if stop.wait(60):
                return

    def gpu_attempts(self, name, out, command, attempts=2):
        """Run a short GPU job; a failed attempt's directory is preserved (renamed) and the job retried once
        after a fresh CUDA probe (newton raises transient CUDA 802/999 and NCCL init errors)."""
        for attempt in range(attempts):
            if out.exists():
                failed = out.with_name(f'{out.name}.failed-{time.time_ns()}')
                out.rename(failed)
                self.record('preserved-failed-attempt', name=name, path=str(failed))
            status = self.run_logged(name, command, check=False)
            if status == 0:
                return
            self.record('attempt-failed', name=name, attempt=attempt, exit_code=status)
            if attempt + 1 < attempts:
                time.sleep(60)
                self.cuda_ready()
        raise RuntimeError(f'{name} failed {attempts} times; see {self.dir}')

    def run_logged(self, name, command, threads=THREADS, check=True):
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
        log = self.dir / f'{name}-{time.time_ns()}.log'
        with log.open('x') as f:
            status = subprocess.run(command, cwd=self.campaign.source, env=env, stdout=f,
                                    stderr=subprocess.STDOUT).returncode
        if check and status:
            raise RuntimeError(f'{name} failed with exit {status}; see {log}')
        return status

    # -- allocation checks --
    def gpu_apps(self):
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=60)
        if out.returncode:
            raise RuntimeError(f'nvidia-smi failed: {out.stderr[:300]}')
        return out.stdout.strip()

    def cuda_ready(self, attempts=6):
        """A tiny matmul on every GPU in a child process (fresh newton nodes can raise CUDA error 802)."""
        probe = ('import torch\nassert torch.cuda.device_count() == 4, torch.cuda.device_count()\n'
                 'for i in range(4):\n    x = torch.randn(512, 512, device=i)\n    float((x @ x).sum())\n'
                 'print("CUDA_READY", [torch.cuda.get_device_name(i) for i in range(4)])\n')
        for attempt in range(attempts):
            out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=600)
            if out.returncode == 0 and 'CUDA_READY' in out.stdout and 'H100' in out.stdout:
                return out.stdout.strip()
            self.record('cuda-not-ready', attempt=attempt, error=(out.stderr or out.stdout)[-500:])
            time.sleep(30)
        raise RuntimeError('CUDA did not become ready on all four GPUs')

    # -- qualification --
    def qualify(self):
        c, shared = self.campaign, self.campaign.shared
        if c.sealed():
            return 'already-sealed'
        shared.mkdir(parents=True, exist_ok=True)
        gates_path = shared / 'gates.json'
        gates = json.loads(gates_path.read_text()) if gates_path.exists() else {}

        def gate(name, fn):
            if gates.get(name, {}).get('passed'):
                return
            self.record('qualifying', gate=name)
            result = fn()
            gates[name] = dict(passed=True, result=result, job=self.job,
                               utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            write(gates_path, gates)
        gate('provenance', self.provenance)
        gate('cpu-suite', lambda: self.run_logged('cpu-suite', [
            sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/memory_sweep',
            'tests/vln/test_episodic_contract.py', 'tests/vln/test_episodic_selector.py',
            'tests/vln/test_episodic_attention.py', 'tests/vln/test_episodic_forward.py',
            'tests/vln/test_committed_write_reference.py']) or 'pass')
        gate('configs', self.check_configs)
        gate('reproduction', self.reproduction)
        if set(gates) != set(QUALIFICATION_GATES) or not all(g['passed'] for g in gates.values()):
            raise RuntimeError('Incomplete qualification')
        write(shared / 'QUALIFIED.json', dict(passed=True, source_commit=c.source_identity(), gates=gates))
        self.record('qualified')
        return 'sealed'

    def provenance(self):
        import importlib.metadata as md
        c = self.campaign
        checks = dict(source_commit=c.source_identity())
        versions = {}
        for name in PINNED:
            try:
                versions[name] = md.version(name)
            except md.PackageNotFoundError:
                versions[name] = None
        import torch
        versions['torch'] = torch.__version__
        if versions != PINNED:
            raise RuntimeError(f'Package versions differ from the pinned S8N stack: {versions} vs {PINNED}')
        checks['packages'] = versions
        if sha(c.data['manifest']) != c.data['manifest_sha256']:
            raise RuntimeError('Manifest bytes changed')
        fp = manifest_fingerprint(c.data['manifest'])
        if fp['fingerprint_v1'] != MANIFEST_FINGERPRINT or any(fp[k] != v for k, v in MANIFEST_COUNTS.items()):
            raise RuntimeError(f'Manifest fingerprint/counts differ from the worker-2 manifest: {fp}')
        checks['manifest'] = fp
        model = Path(c.data['model'])
        files = {}
        for path in sorted(model.iterdir()):
            target = path.resolve()
            digest = sha(target)
            files[path.name] = digest
            if len(target.name) == 64 and target.parent.name == 'blobs' and digest != target.name:
                raise RuntimeError(f'Model blob {path.name} is corrupt')
        for name, digest in c.data['base_assets_sha256'].items():
            if files.get(name) != digest:
                raise RuntimeError(f'Base model file {name} differs from the evaluation contract')
        checks['model_files'] = files
        for name, digest in c.data['reference_sha256'].items():
            if sha(c.reference / name) != digest:
                raise RuntimeError(f'Reference file {name} changed')
        # deterministic image sample + fixture images
        import random
        rows = [json.loads(line) for line in open(c.data['manifest'])]
        paths = [s['rgb_path'] for r in rows for s in r['steps']]
        sample = random.Random(429).sample(paths, 4000)
        missing = [p for p in sample if not os.path.isfile(p)]
        fixture = [json.loads(line) for line in open(Path(c.data['fixtures']) / 'rxrce_longest.jsonl')]
        missing += [s['rgb_path'] for r in fixture for s in r['steps'] if not os.path.isfile(s['rgb_path'])]
        if missing:
            raise RuntimeError(f'{len(missing)} images missing, e.g. {missing[:3]}')
        checks['images_present'] = dict(sample=len(sample), fixture_steps=sum(len(r['steps']) for r in fixture))
        write(c.shared / 'provenance.json', checks)
        return dict(source_commit=checks['source_commit'], manifest=fp, packages=versions)

    def check_configs(self):
        c = self.campaign
        out = {}
        for arm in (*REFERENCES, *ARMS):
            identity = c.run_identity(arm)
            out[arm] = dict(memory_sha256=identity['memory_sha256'], memory=identity['config']['memory'])
        # the S8N arms are exactly the registry generator's S8N with R/B changed and S8N's frozen floors
        from qwen_vl.research.gdn_sweep_contract import arm_memory_config
        s8n = json.loads((c.reference / 'S8N/resolved_config.json').read_text())['memory']
        for arm, spec in ARMS.items():
            if spec['family'] == 'S8N':
                generated = arm_memory_config('S8N', recent=spec['recent'], budget=spec['budget'],
                                              artifacts=s8n['artifacts'])['memory']
                if json.loads(json.dumps(generated)) != out[arm]['memory']:
                    raise RuntimeError(f'{arm}: differs from arm_memory_config')
        if json.loads(json.dumps(arm_memory_config('S8N', artifacts=s8n['artifacts'])['memory'])) != s8n:
            raise RuntimeError('arm_memory_config does not reproduce the published S8N memory')
        return {arm: dict(memory_sha256=v['memory_sha256'], kv_guard=v['memory']['kv_guard_tokens'])
                for arm, v in out.items()}

    def check_profile(self, train_dir, updates):
        from scripts.vln.profile_episodic import verify
        check = verify(train_dir, WORLD, updates, 8, MAX_RESERVED_GIB)
        resources = json.loads((Path(train_dir).parent / 'supervisor/resources.json').read_text())
        if not check['passed'] or resources['status'] != 'PASS':
            raise RuntimeError(f'Profile gate failed: {check} {resources}')
        return dict(check=check, resources=resources)

    def reproduction(self):
        c = self.campaign
        runs = {}
        first = range(REPRODUCED_UPDATES)          # exposure rows are 0-based
        for arm in REFERENCES:
            out = c.run_dir(arm)
            result_path = out / 'result.json'
            if not result_path.exists():
                self.record('reproduction', arm=arm)
                self.gpu_attempts(f'reproduction-{arm}', out, train_command(
                    c, arm, out, updates=REPRODUCED_UPDATES, profile_only=True, host_guard=self.host_guard))
                check = self.check_profile(out / 'train', REPRODUCED_UPDATES)
                if json.loads((out / 'train/resolved_config.json').read_text()) != c.resolve(arm):
                    raise RuntimeError(f'{arm}: reproduction config differs')
                write(result_path, dict(losses=check['check']['losses'], grad_norms=check['check']['grad_norms'],
                                        peak_reserved_gib=check['check']['peak_reserved_gib'],
                                        resources=check['resources']))
            runs[arm] = dict(json.loads(result_path.read_text()), plan=exposure_plan(out / 'train', first))
        comparison = compare_reproduction({f: c.family_log(f) for f in FAMILIES}, runs)
        comparison['plan'] = {str(r): {str(u): uids for u, uids in v.items()}
                              for r, v in runs['S8N_REF']['plan'].items()}
        write(c.shared / 'reproduction/comparison.json', comparison)
        if not comparison['passed']:
            raise RuntimeError(f'Reproduction failed: {comparison["problems"]}')
        return {k: v for k, v in comparison.items() if k != 'plan'}

    # -- one arm --
    def smoke(self, arm, run):
        out = run / f'smoke-omp{THREADS}'
        result_path = out / 'result.json'
        if result_path.exists():
            return json.loads(result_path.read_text())
        fixture = Path(self.campaign.data['fixtures']) / 'rxrce_longest.jsonl'
        self.record('smoke', arm=arm)
        self.gpu_attempts(f'{arm}-smoke', out, train_command(self.campaign, arm, out, manifest=fixture, updates=2,
                                                             profile_only=True, host_guard=self.host_guard))
        result = self.check_profile(out / 'train', 2)
        if json.loads((out / 'train/resolved_config.json').read_text()) != self.campaign.resolve(arm):
            raise RuntimeError(f'{arm}: smoke config differs from production')
        result.update(threads=THREADS, job=self.job, host=os.uname().nodename,
                      **base.optimizer_step_estimate(out / 'train'))
        write(result_path, result)
        return result

    def production_segment(self, arm, run):
        """Train until the epoch completes, the process fails or the allocation deadline asks for a stop."""
        c = self.campaign
        command = train_command(c, arm, run, production=True, host_guard=self.host_guard)
        request = run / 'train' / 'REQUEST_CHECKPOINT_AND_STOP'
        request.unlink(missing_ok=True)
        (run / 'train').mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS))
        log = self.dir / f'{arm}-production-{time.time_ns()}.log'
        start_step = latest_checkpoint_step(run)
        self.record('training', arm=arm, resume_from=start_step, run=str(run), end=self.end)
        stop = threading.Event()
        events = dict(requested=None, killed=None)
        supervisors_before = set(run.glob('supervisor-*'))

        def watchdog(process):
            while not stop.wait(30):
                self.end = self.job_end() or self.end
                action = deadline_actions(time.time(), self.end, c.data['request_margin_seconds'],
                                          c.data['kill_margin_seconds'])
                if action == 'request' and not events['requested']:
                    request.write_text(self.job + '\n')
                    events['requested'] = time.time()
                    self.record('checkpoint-requested', arm=arm, end=self.end)
                elif action == 'kill' and not events['killed']:
                    events['killed'] = time.time()
                    self.record('deadline-kill', arm=arm)
                    for sup in set(run.glob('supervisor-*')) - supervisors_before:
                        owned = sup / 'owned_process.json'
                        if owned.exists():
                            try:
                                os.killpg(json.loads(owned.read_text())['pgid'], signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                    try:
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        process.kill()
        with log.open('x') as f:
            process = subprocess.Popen(command, cwd=c.source, env=env, stdout=f, stderr=subprocess.STDOUT)
            thread = threading.Thread(target=watchdog, args=(process,), daemon=True)
            thread.start()
            status = process.wait()
            stop.set()
            thread.join()
        request.unlink(missing_ok=True)
        return dict(exit_code=status, log=str(log), start_step=start_step, end_step=latest_checkpoint_step(run),
                    requested=events['requested'], killed=events['killed'])

    def complete(self, arm, run, cfg):
        audit, plan = audit_epoch(run / 'train')
        reference_plan = json.loads((self.campaign.shared / 'reproduction/comparison.json').read_text())['plan']
        first = {str(r): {str(u): plan[r][u] for u in range(REPRODUCED_UPDATES)} for r in plan}
        if first != reference_plan:
            audit['problems'].append('first updates differ from the reproduction runs (data order)')
        order = self.campaign.shared / 'exposure_plan.json'
        flat = {str(r): {str(u): uids for u, uids in v.items()} for r, v in plan.items()}
        if order.exists():
            if json.loads(order.read_text()) != flat:
                audit['problems'].append('exposure plan differs from the first completed arm')
        elif not audit['problems']:
            write(order, flat)
        state = json.loads((run / f'train/checkpoint-{EXPECTED_STEPS}/trainer_state.json').read_text())
        losses = {r['step']: r['loss'] for r in state['log_history'] if 'loss' in r}
        audit['loss_correlation_vs'] = {f: loss_correlation(losses, {k: v['loss'] for k, v in
                                                                      self.campaign.family_log(f).items()})
                                        for f in FAMILIES}
        audit['passed'] = not audit['problems']
        write(run / 'completion_audit.json', audit)
        if not audit['passed']:
            raise RuntimeError(f'{arm}: completion audit failed: {audit["problems"]}')
        removed = base.prune_rolling(run / 'train')
        self.record('arm-complete', arm=arm, pruned=removed, audit=audit)
        return 'complete'

    def train_arm(self, arm):
        c = self.campaign
        if not c.sealed():
            raise RuntimeError('Campaign is not qualified')
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
        if base.run_complete(run, cfg):
            return self.complete(arm, run, cfg) if not (run / 'completion_audit.json').exists() else 'complete'
        self.smoke(arm, run)
        failures = 0
        while True:
            self.end = self.job_end() or self.end
            if self.end and self.end - time.time() < c.data['min_resume_seconds']:
                self.record('segment-skipped-short-allocation', arm=arm, left=self.end - time.time())
                return 'stopped'
            segment = self.production_segment(arm, run)
            if base.run_complete(run, cfg):
                return self.complete(arm, run, cfg)
            if segment['requested'] or segment['killed']:
                self.record('segment-stopped', arm=arm, **segment)
                return 'stopped'
            progressed = segment['end_step'] > segment['start_step']
            failures = 0 if progressed else failures + 1
            self.record('production-attempt-failed', arm=arm, failures=failures,
                        resources=base.latest_resources(run), **segment)
            if failures >= 2:
                return 'failed'
            time.sleep(120)

    # -- entry --
    def main(self):
        if os.environ.get('SLURM_JOB_ID') != self.job:
            raise RuntimeError(f'Lane for job {self.job} runs in {os.environ.get("SLURM_JOB_ID")}')
        self.step = f'{self.job}.{os.environ.get("SLURM_STEP_ID")}'
        self.end = self.job_end()
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(15):
                write(self.dir / 'heartbeat.json', dict(unix=time.time(), step=self.step, pid=os.getpid(),
                                                        task=self.task, host=os.uname().nodename))
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            self.campaign.source_identity()
            apps = self.gpu_apps()
            if apps:
                self.record('busy', apps=apps[:1000])
                return 'busy'
            ready = self.cuda_ready()
            if self.gpu_apps():
                self.record('busy', apps='appeared during CUDA probe')
                return 'busy'
            self.record('started', host=os.uname().nodename, end=self.end, cuda=ready)
            # first pass at a low threshold clears whatever cache the allocation arrived with
            threading.Thread(target=self.relieve_cache,
                             args=(stop, self.campaign.data['cache_relief']['initial_gib']), daemon=True).start()
            outcome = self.qualify() if self.task == 'qualify' else self.train_arm(self.task)
            self.record('task-finished', outcome=outcome)
            return outcome
        except BaseException as exc:
            self.record('task-error', error=repr(exc)[:3000])
            raise
        finally:
            stop.set()
            thread.join()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--job')
    p.add_argument('--task', help="'qualify' or an arm")
    p.add_argument('--preflight', action='store_true', help='Check every identity without GPU work')
    args = p.parse_args()
    campaign = Campaign(args.campaign)
    if args.preflight:
        report = {}
        for arm in (*REFERENCES, *ARMS):
            identity = campaign.run_identity(arm)
            report[arm] = dict(memory_sha256=identity['memory_sha256'],
                               kv_guard=identity['config']['memory']['kv_guard_tokens'],
                               recent=identity['config']['memory']['recent_groups_including_current'],
                               budget=identity['config']['memory']['episodic_budget_tokens'])
        print(json.dumps(report, indent=2))
        print('PREFLIGHT_PASS', flush=True)
        return
    if args.task not in ('qualify', *ARMS):
        raise SystemExit(f'Unknown task {args.task}')
    Lane(campaign, args.job, args.task).main()


if __name__ == '__main__':
    main()
