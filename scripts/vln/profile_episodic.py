"""Complete-episode resource qualification for one runtime profile (H4 or H2).

``select`` computes exact encoded lengths of the full manifest and writes stress
fixtures (complete real episodes, repeated only to fill global slots; repeats
are recorded).  ``gate`` runs one real Trainer stress test under the owned
supervisor and verifies: every rank finished every update, finite losses and
gradient norms, expected exposures, GPU reserved <= limit on every rank and
update, no host-guard trigger and no new OOM-kill event.  ``suite`` runs the
declared gates sequentially (never concurrently on the same devices) and
writes one profile report.  Smoke fixtures never initialize production runs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {
    'H4': dict(world=4, runtime='configs/episodic/profile_h4.yaml', deepspeed='deepspeed.json'),
    'H2': dict(world=2, runtime='configs/episodic/profile_h2.yaml',
               deepspeed='configs/deepspeed_zero2_cpu_optimizer.json'),
}
OFFLOAD = 65536


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 23), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded_lengths(manifest, model_path, cache):
    cache = Path(cache)
    key = sha256(manifest)
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved['manifest_sha256'] == key:
            return saved
    import transformers
    sys.path.insert(0, str(ROOT / 'src'))
    from qwen_vl.data.episode_dataset import EpisodeDataset
    from qwen_vl.data.episode_serializer import EpisodeSerializer
    from qwen_vl.train.vln_runtime import resolve_config
    cfg = resolve_config(str(ROOT / 'configs/episodic/science_common.yaml'),
                         str(ROOT / 'configs/episodic/arm_m2.yaml'),
                         runtime=str(ROOT / 'configs/episodic/profile_h4.yaml'))
    serializer = EpisodeSerializer(transformers.AutoProcessor.from_pretrained(model_path), cfg)
    dataset = EpisodeDataset(manifest, serializer)
    rows = [dict(uid=e['episode_uid'], dataset=e['dataset'], steps=len(e['steps']), tokens=n,
                 prefix=serializer.encode_prefix(e['instruction'])['input_ids'].numel())
            for e, n in zip(dataset.episodes, dataset.encoded_lengths)]
    saved = dict(manifest_sha256=key, episodes=rows)
    cache.write_text(json.dumps(saved) + '\n')
    return saved


def select(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = encoded_lengths(args.manifest, args.model_path, out / 'encoded_lengths.json')
    rows = info['episodes']
    by_uid = {r['uid']: r for r in rows}
    r2r = [r for r in rows if r['dataset'] == 'r2rce']
    rxr = [r for r in rows if r['dataset'] == 'rxrce']
    longest_r2r = max(r2r, key=lambda r: (r['tokens'], r['uid']))
    longest_rxr = max(rxr, key=lambda r: (r['tokens'], r['uid']))
    below = max((r for r in rows if r['tokens'] < OFFLOAD), key=lambda r: (r['tokens'], r['uid']))
    above = min((r for r in rows if r['tokens'] >= OFFLOAD), key=lambda r: (r['tokens'], r['uid']))
    big_prefix = max((r for r in rows if r['tokens'] >= OFFLOAD), key=lambda r: (r['prefix'], r['uid']))
    ordered = sorted(rows, key=lambda r: (r['tokens'], r['uid']))
    shortest = [r for r in ordered if r['dataset'] == 'r2rce'][:4] + [r for r in ordered if r['dataset'] == 'rxrce'][:4]

    def near(target, exclude):
        candidates = [r for r in rows if r['uid'] not in exclude]
        return min(candidates, key=lambda r: (abs(r['tokens'] - target), r['uid']))
    # Mixed tail: longest + alternating below/above-threshold long/medium episodes.
    mixed, chosen = [longest_rxr], {longest_rxr['uid']}
    for target in (60000, 120000, 40000, 90000, 64000, 160000, 25000):
        pick = near(target, chosen); mixed.append(pick); chosen.add(pick['uid'])
    # Interactive-allocation variant (128 GiB): same pattern without episodes > cap.
    cap = args.interactive_token_cap
    small_mixed, chosen = [], set()
    for target in (64000, 30000, 66000, 45000, 70000, 20000, 58000, 68000):
        pick = near(target, chosen | {r['uid'] for r in rows if r['tokens'] > cap})
        small_mixed.append(pick); chosen.add(pick['uid'])
    fixtures = dict(
        smoke=shortest,
        longest_r2r=[longest_r2r] * 8, below_offload=[below] * 8,
        above_offload=[above] * 4 + [big_prefix] * 4,
        longest_rxr=[longest_rxr] * 8, mixed_tail=mixed, mixed_tail_interactive=small_mixed,
    )
    lookup = {}
    for line in Path(args.manifest).open():
        uid = json.loads(line)['episode_uid'] if '"episode_uid"' in line else None
        if uid in {r['uid'] for f in fixtures.values() for r in f}:
            lookup[uid] = line if line.endswith('\n') else line + '\n'
    record = {}
    for name, items in fixtures.items():
        path = out / f'{name}.jsonl'
        path.write_text(''.join(lookup[r['uid']] for r in items))
        record[name] = dict(path=str(path), sha256=sha256(path),
                            episodes=[dict(uid=r['uid'], dataset=r['dataset'], steps=r['steps'],
                                           tokens=r['tokens'], prefix=r['prefix']) for r in items],
                            repeated=len({r['uid'] for r in items}) != len(items),
                            max_tokens=max(r['tokens'] for r in items))
    summary = dict(manifest=args.manifest, manifest_sha256=info['manifest_sha256'], fixtures=record,
                   corpus=dict(episodes=len(rows), max_tokens=longest_rxr['tokens'],
                               longest_rxr=longest_rxr, longest_r2r=longest_r2r,
                               just_below_offload=below, just_above_offload=above, largest_prefix_above=big_prefix,
                               episodes_at_or_above_offload=sum(r['tokens'] >= OFFLOAD for r in rows)))
    (out / 'fixtures.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['corpus'], indent=2), flush=True)


def _rows(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def verify(train_dir, world, updates, slots, max_reserved, first_update=1):
    train_dir = Path(train_dir)
    problems, peaks, hosts = [], [], []
    for rank in range(world):
        profile = _rows(train_dir / f'profile_rank{rank}.jsonl')
        exposures = _rows(train_dir / f'exposures_rank{rank}.jsonl')
        seen = sorted({r['update'] for r in profile})
        if seen != list(range(first_update, updates + 1)):
            problems.append(f'rank {rank} updates {seen}')
        for r in profile:
            peaks.append(r['peak_reserved_gib'])
            hosts.append(r.get('process_rss_gib', 0))
            if r.get('cuda_ooms', 0):
                problems.append(f'rank {rank} CUDA OOM counter {r["cuda_ooms"]}')
        expected = (updates - first_update + 1) * slots // world
        if len(exposures) != expected:
            problems.append(f'rank {rank} exposures {len(exposures)} != {expected}')
        if not all(math.isfinite(e['loss_sum']) for e in exposures):
            problems.append(f'rank {rank} nonfinite loss')
    state = train_dir / 'trainer_state.json'
    log = json.loads(state.read_text())['log_history'] if state.exists() else []
    steps = [r for r in log if 'loss' in r]
    if len(steps) < updates - first_update + 1 or not all(math.isfinite(r['loss']) and math.isfinite(r.get('grad_norm', 0))
                                       for r in steps):
        problems.append(f'trainer log incomplete/nonfinite: {len(steps)} updates')
    if peaks and max(peaks) > max_reserved:
        problems.append(f'GPU reserved {max(peaks):.3f} GiB > {max_reserved}')
    return dict(passed=not problems, problems=problems, peak_reserved_gib=max(peaks) if peaks else None,
                max_process_rss_gib=max(hosts) if hosts else None,
                losses=[r['loss'] for r in steps], grad_norms=[r.get('grad_norm') for r in steps])


def train_command(profile, arm, manifest, out, model_path, updates, extra=()):
    spec = PROFILES[profile]
    return ['torchrun', '--standalone', f'--nproc_per_node={spec["world"]}', '-m', 'qwen_vl.train.train_qwen',
            '--vln_config', 'configs/episodic/science_common.yaml',
            '--output_config', f'configs/episodic/arm_{arm}.yaml',
            '--runtime_profile', spec['runtime'], '--deepspeed', spec['deepspeed'],
            '--manifest', str(manifest), '--model_name_or_path', model_path,
            '--max-optimizer-updates', str(updates), '--debug-repeat-episodes',
            '--output_dir', str(out), '--run-name', f'qual-{profile}-{arm}', *extra]


def run_gate(profile, arm, fixture, root, model_path, updates, host_guard=None, extra=('--profile-only',),
             slots=8, name=None, max_reserved=78.0, first_update=1):
    name = name or f'{arm}-{Path(fixture).stem}'
    out = Path(root) / profile / name
    if out.exists():
        raise ValueError(f'Refusing to reuse gate output {out}')
    supervisor = ['python', '-m', 'scripts.vln.run_episodic_gate', '--out', str(out / 'supervisor'),
                  '--disk-path', str(root)]
    if host_guard:
        supervisor += ['--host-guard-gib', str(host_guard)]
    command = supervisor + ['--', *train_command(profile, arm, fixture, out / 'train', model_path, updates, extra)]
    started = time.time()
    status = subprocess.run(command, cwd=ROOT).returncode
    resources = json.loads((out / 'supervisor/resources.json').read_text())
    checks = verify(out / 'train', PROFILES[profile]['world'], updates, slots, max_reserved, first_update)
    result = dict(gate=name, profile=profile, arm=arm, fixture=str(fixture), updates=updates,
                  supervisor_status=resources['status'], resources=resources, training=checks,
                  passed=status == 0 and resources['status'] == 'PASS' and checks['passed'],
                  wall_seconds=time.time() - started)
    (out / 'gate.json').write_text(json.dumps(result, indent=2) + '\n')
    print('GATE', json.dumps(dict(gate=name, passed=result['passed'], peak_reserved=checks['peak_reserved_gib'],
                                  host_ws=resources['sampled_peak_working_set_gib'],
                                  problems=checks['problems'])), flush=True)
    return result


def checkpoint_gate(profile, arm, fixture, root, model_path, host_guard=None):
    """Save a complete recovery checkpoint under stress, exit, reload fresh, continue."""
    first = run_gate(profile, arm, fixture, root, model_path, 2, host_guard,
                     extra=('--recovery-save-steps', '1'), name=f'{arm}-checkpoint-save')
    train = Path(root) / profile / f'{arm}-checkpoint-save' / 'train'
    sys.path.insert(0, str(ROOT / 'src'))
    from qwen_vl.train.recovery import validate_complete
    record = validate_complete(train / 'checkpoint-2')
    resumed = run_gate(profile, arm, fixture, root, model_path, 3, host_guard,
                       extra=('--recovery-save-steps', '1', '--resume_from_checkpoint', str(train / 'checkpoint-2')),
                       name=f'{arm}-checkpoint-resume', first_update=3)
    # The resume gate writes into a fresh dir; verify update 3 happened after the reload.
    return dict(save=first, resume=resumed, checkpoint_files=len(record['files']),
                checkpoint_bytes=sum(record['files'].values()),
                passed=first['passed'] and resumed['passed'])


def suite(args):
    fixtures = json.loads((Path(args.fixtures) / 'fixtures.json').read_text())['fixtures']
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    results = []
    for gate in args.gates:
        for arm in args.arms:
            if gate == 'checkpoint':
                results.append(dict(gate=f'{arm}-checkpoint', **checkpoint_gate(
                    args.profile, arm, fixtures[args.checkpoint_fixture]['path'], root, args.model_path,
                    args.host_guard)))
            else:
                slots = len(fixtures[gate]['episodes'])
                results.append(run_gate(args.profile, arm, fixtures[gate]['path'], root, args.model_path,
                                        args.updates, args.host_guard, slots=slots))
            report = dict(profile=args.profile, gates=args.gates, arms=args.arms, results=results,
                          passed=all(r['passed'] for r in results), host=os.uname().nodename,
                          slurm_job=os.environ.get('SLURM_JOB_ID'))
            (root / f'{args.profile}_suite.json').write_text(json.dumps(report, indent=2) + '\n')
            if not results[-1]['passed'] and not args.keep_going:
                print('SUITE_FAIL', results[-1].get('gate'), flush=True)
                raise SystemExit(1)
    print('SUITE_PASS' if all(r['passed'] for r in results) else 'SUITE_FAIL', flush=True)
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select')
    s.add_argument('--manifest', required=True)
    s.add_argument('--model-path', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--interactive-token-cap', type=int, default=72000)
    q = sub.add_parser('suite')
    q.add_argument('--profile', choices=PROFILES, required=True)
    q.add_argument('--fixtures', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--model-path', required=True)
    q.add_argument('--arms', nargs='+', default=['v0', 'm2'])
    q.add_argument('--gates', nargs='+', required=True)
    q.add_argument('--updates', type=int, default=3)
    q.add_argument('--checkpoint-fixture', default='longest_rxr')
    q.add_argument('--host-guard', type=float)
    q.add_argument('--keep-going', action='store_true')
    args = p.parse_args()
    select(args) if args.command == 'select' else suite(args)


if __name__ == '__main__':
    main()
