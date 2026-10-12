"""Episodic campaign coordinator: manifests, immutable source, owned submissions.

``init``     create the campaign manifest and a detached read-only source snapshot.
``submit``   create one isolated attempt directory per runtime profile for ONE
             logical run and submit each with ``sbatch --parsable`` (or only write
             the plan with ``--dry-run``).  Every job ID is recorded in the
             append-only submission ledger, which is the cancellation allowlist.
``status``   summarize leases, attempts and live scheduler state.

No command here cancels anything; sibling cancellation happens only inside a
verified winner (``qwen_vl.train.campaign_lease.cancel_recorded_sibling``).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from qwen_vl.train.campaign_lease import Lease, append_jsonl, atomic_write_json, slurm_state  # noqa: E402

PROFILES = {
    'H4': dict(world=4, gpus=4, cpus=16, mem='512G', time='3-00:00:00', omp=2,
               runtime='configs/episodic/profile_h4.yaml', deepspeed='deepspeed.json'),
    'H2': dict(world=2, gpus=2, cpus=16, mem='512G', time='5-00:00:00', omp=4,
               runtime='configs/episodic/profile_h2.yaml', deepspeed='configs/deepspeed_zero2_cpu_optimizer.json'),
}
RUNS = {  # Phase-1 primary arms; Phase-2 arms require the documented evidence gate.
    'P1-V0-s429': 'v0', 'P1-M1-s429': 'm1', 'P1-M2-s429': 'm2', 'P1-M2c-s429': 'm2c',
}
PHASE2_RUNS = {  # submitted only with --allow-phase2 and a recorded --phase2-reason
    'P2-M5-s429': 'm5', 'P2-M5c-s429': 'm5c',
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 23), b''):
            digest.update(block)
    return digest.hexdigest()


def tree_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def numerical_report_bytes(path, commit):
    raw = Path(path).read_bytes()
    report = json.loads(raw)
    if (not isinstance(report, dict) or report.get('passed') is not True
            or report.get('source_commit') != commit or report.get('source_dirty') is not False):
        raise SystemExit('Numerical evidence must PASS on the clean requested source commit')
    if not isinstance(report.get('runtime_sha256'), str) or len(report['runtime_sha256']) != 64:
        raise SystemExit('Numerical evidence must identify its runtime profile')
    return raw


def per_profile(cast):
    """'32' applies to every profile; 'H4=32,H2=30' sets each profile separately."""
    def parse(text):
        if '=' not in text:
            return {profile: cast(text) for profile in PROFILES}
        values = {k: cast(v) for k, v in (kv.split('=') for kv in text.split(','))}
        if set(values) - set(PROFILES):
            raise argparse.ArgumentTypeError(f'unknown profile in {text!r}')
        return values
    return parse


def store_numerical_reports(campaign, raws):
    """Evidence is bound to one runtime profile hash; keep one report per runtime."""
    refs = {}
    for raw in raws:
        runtime = json.loads(raw)['runtime_sha256']
        if runtime in refs:
            raise SystemExit(f'Two numerical reports for runtime {runtime}')
        refs[runtime] = store_numerical_report(campaign, raw)
    return refs


def numerical_reference(manifest, runtime_path):
    """The stored passing report whose runtime hash equals this profile's runtime file."""
    runtime = sha256_file(runtime_path)
    refs = dict(manifest.get('numerical_reports') or {})
    if not refs and manifest.get('numerical_report'):   # campaigns initialized with one report
        ref = manifest['numerical_report']
        refs[json.loads(Path(ref['path']).read_bytes())['runtime_sha256']] = ref
    if runtime not in refs:
        raise SystemExit(f'No numerical evidence for runtime {runtime_path} ({runtime})')
    return refs[runtime]


def store_numerical_report(campaign, raw):
    digest = hashlib.sha256(raw).hexdigest()
    path = campaign / 'numerical' / f'{digest}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)
        path.chmod(0o444)
    elif path.read_bytes() != raw:
        raise SystemExit('Immutable numerical evidence changed')
    return dict(path=str(path.resolve()), sha256=digest)


def init(args):
    campaign = Path(args.campaign_root)
    manifest_path = campaign / 'campaign_manifest.json'
    if manifest_path.exists():
        raise SystemExit(f'Campaign already initialized: {manifest_path}')
    commit = subprocess.check_output(['git', 'rev-parse', args.source_commit], cwd=ROOT, text=True).strip()
    numerical_raws = [numerical_report_bytes(path, commit) for path in args.numerical_report]
    source = campaign / 'source' / commit
    if source.exists():
        raise SystemExit(f'Source snapshot exists: {source}')
    source.mkdir(parents=True)
    archive = subprocess.Popen(['git', 'archive', commit], cwd=ROOT, stdout=subprocess.PIPE)
    subprocess.run(['tar', '-x', '-C', str(source)], stdin=archive.stdout, check=True)
    if archive.wait() != 0:
        raise SystemExit('git archive failed')
    hashes = tree_hashes(source)
    (campaign / 'source_hashes.json').write_text(json.dumps(dict(commit=commit, files=hashes), indent=2) + '\n')
    for path in source.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    numerical_refs = store_numerical_reports(campaign, numerical_raws)
    payload = dict(
        numerical_reports=numerical_refs, numerical_report=next(iter(numerical_refs.values())),
        campaign_uuid=str(uuid.uuid4()), created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        plan='plans/episodic_mem.md', source_commit=commit, source_dir=str(source),
        source_tree_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        env_dir=args.env_dir, cuda_home=args.cuda_home, cxx=args.cxx,
        model_path=args.model_path, model_revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
        manifest=args.manifest, manifest_sha256=sha256_file(args.manifest),
        manifest_records=sum(1 for line in open(args.manifest) if line.strip()),
        fixtures=args.fixtures, hf_repo=args.hf_repo, hf_home=args.hf_home,
        protected_jobs=str(campaign / 'protected_jobs.json'),
        disk_reservation_gib=args.disk_reservation_gib, epochs_to_train=1,
        recovery_save_steps=250, first_recovery_save_step=50, verify_updates=50,
        publish_checkpoints=not args.no_upload)
    atomic_write_json(manifest_path, payload)
    print(json.dumps(payload, indent=2))


def _snapshot(campaign, commit):
    source = campaign / 'source' / commit
    if source.exists():
        raise SystemExit(f'Source snapshot exists: {source}')
    source.mkdir(parents=True)
    archive = subprocess.Popen(['git', 'archive', commit], cwd=ROOT, stdout=subprocess.PIPE)
    subprocess.run(['tar', '-x', '-C', str(source)], stdin=archive.stdout, check=True)
    if archive.wait() != 0:
        raise SystemExit('git archive failed')
    hashes = tree_hashes(source)
    for path in source.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    return source, hashes


def resnapshot(args):
    """Add a corrected immutable source; existing attempts keep their recorded source."""
    campaign = Path(args.campaign_root)
    manifest = json.loads((campaign / 'campaign_manifest.json').read_text())
    commit = subprocess.check_output(['git', 'rev-parse', args.source_commit], cwd=ROOT, text=True).strip()
    numerical_raws = [numerical_report_bytes(path, commit) for path in args.numerical_report]
    source, hashes = _snapshot(campaign, commit)
    previous = dict(source_commit=manifest['source_commit'], source_dir=manifest['source_dir'],
                    source_tree_sha256=manifest['source_tree_sha256'],
                    numerical_report=manifest.get('numerical_report'),
                    numerical_reports=manifest.get('numerical_reports'))
    shutil.copy(campaign / 'source_hashes.json', campaign / f'source_hashes.{manifest["source_commit"]}.json')
    (campaign / 'source_hashes.json').write_text(json.dumps(dict(commit=commit, files=hashes), indent=2) + '\n')
    manifest.setdefault('source_history', []).append(dict(previous, replaced=time.time(), reason=args.reason))
    numerical_refs = store_numerical_reports(campaign, numerical_raws)
    manifest.update(numerical_reports=numerical_refs, numerical_report=next(iter(numerical_refs.values())),
                    source_commit=commit, source_dir=str(source),
                    source_tree_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest())
    atomic_write_json(campaign / 'campaign_manifest.json', manifest)
    append_jsonl(campaign / 'submission_ledger.jsonl', dict(event='resnapshot', previous=previous,
                                                             source_commit=commit, reason=args.reason))
    print(json.dumps(dict(source_commit=commit, source_dir=str(source)), indent=2))


def _next_attempt(run_dir, profile):
    existing = sorted(run_dir.glob(f'*-{profile}-a*'))
    return len(existing) + 1


def submit(args):
    campaign = Path(args.campaign_root)
    manifest = json.loads((campaign / 'campaign_manifest.json').read_text())
    if args.run not in RUNS and not args.allow_phase2:
        raise SystemExit(f'Unknown or gated run {args.run}')
    if args.run not in RUNS and not (args.phase2_reason or '').strip():
        raise SystemExit('A gated (Phase-2) run needs --phase2-reason, recorded in the ledger')
    arm = RUNS.get(args.run) or PHASE2_RUNS.get(args.run, args.arm)
    if args.arm and args.arm != arm:
        raise SystemExit('Arm does not match the run identity')
    lease = Lease(campaign, args.run).read()
    if lease and lease.get('state') in ('TRAINING_VERIFIED', 'ACTIVE', 'COMPLETE') and not args.allow_existing_lease:
        raise SystemExit(f'{args.run} already has a verified owner: {lease}')
    source = Path(manifest['source_dir'])
    from qwen_vl.train.vln_runtime import resolve_config
    import yaml
    planned = []
    for profile in args.profiles:
        spec = dict(PROFILES[profile])
        if getattr(args, 'cpus', None) and profile in args.cpus:
            spec['cpus'] = args.cpus[profile]
        if getattr(args, 'omp', None) and profile in args.omp:
            spec['omp'] = args.omp[profile]
        if min(spec['cpus'], spec['omp']) < 1:
            raise SystemExit('CPU and OpenMP thread counts must be positive')
        if args.mem and profile in args.mem:
            spec['mem'] = args.mem[profile]
        numerical = numerical_reference(manifest, source / spec['runtime'])
        run_dir = campaign / 'runs' / args.run
        attempt = f'{args.run}-{profile}-a{_next_attempt(run_dir, profile)}'
        attempt_dir = run_dir / attempt
        attempt_dir.mkdir(parents=True)
        cfg = resolve_config(str(source / 'configs/episodic/science_common.yaml'),
                             str(source / f'configs/episodic/arm_{arm}.yaml'),
                             runtime=str(source / spec['runtime']))
        resolved = yaml.safe_dump(cfg, sort_keys=True)
        (attempt_dir / 'resolved_config.yaml').write_text(resolved)
        shutil.copy(campaign / 'source_hashes.json', attempt_dir / 'source_hashes.json')
        job_name = f'epi-{args.run}-{profile}'
        comment = f'{manifest["campaign_uuid"]}/{args.run}/{attempt}'
        launch = dict(
            campaign_uuid=manifest['campaign_uuid'], campaign_root=str(campaign), run_id=args.run, arm=arm,
            profile=profile, attempt=attempt, attempt_dir=str(attempt_dir), job_name=job_name,
            numerical_report=numerical, phase2_reason=getattr(args, 'phase2_reason', None),
            partition=getattr(args, 'partition', 'highgpu'),
            comment=comment, source_dir=str(source), source_commit=manifest['source_commit'],
            config_sha256=hashlib.sha256(resolved.encode()).hexdigest(), resources=spec,
            qualification=args.qualify, race_siblings=[p for p in args.profiles if p != profile],
            submit_ablations=list(args.ablations or []), created=time.time())
        atomic_write_json(attempt_dir / 'launch_manifest.json', launch)
        command = ['sbatch', '--parsable', f'--job-name={job_name}', f'--comment={comment}',
                   f'--partition={getattr(args, "partition", "highgpu")}', '--nodes=1', '--ntasks=1',
                   f'--gres=gpu:nvidia_h100_80gb_hbm3:{spec["gpus"]}', f'--cpus-per-task={spec["cpus"]}',
                   f'--mem={spec["mem"]}', f'--time={spec["time"]}', '--signal=B:USR1@1800',
                   f'--output={attempt_dir}/slurm-%j.log',
                   str(source / 'train/slurm/episodic_candidate.sbatch'), str(attempt_dir)]
        planned.append(dict(attempt=attempt, command=command))
        if args.dry_run:
            append_jsonl(campaign / 'submission_ledger.jsonl', dict(event='planned', run_id=args.run,
                                                                     attempt=attempt, command=command))
            continue
        job = subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip().split(';')[0]
        append_jsonl(campaign / 'submission_ledger.jsonl', dict(
            event='submitted', run_id=args.run, attempt=attempt, profile=profile, job_id=job,
            job_name=job_name, comment=comment, output_dir=str(attempt_dir),
            source_sha=manifest['source_commit'], config_sha=launch['config_sha256'],
            **({'phase2_reason': args.phase2_reason} if args.run not in RUNS else {})))
        atomic_write_json(attempt_dir / 'status.json', dict(state='SUBMITTED', job_id=job, time=time.time()))
        planned[-1]['job_id'] = job
        print(f'SUBMITTED {attempt} job {job}', flush=True)
    print(json.dumps(planned, indent=2))
    return planned


def status(args):
    campaign = Path(args.campaign_root)
    rows = []
    ledger = campaign / 'submission_ledger.jsonl'
    for line in (ledger.read_text().splitlines() if ledger.exists() else []):
        row = json.loads(line)
        if row.get('event') == 'submitted':
            attempt_status = Path(row['output_dir']) / 'status.json'
            rows.append(dict(run=row['run_id'], attempt=row['attempt'], job=row['job_id'],
                             slurm=slurm_state(row['job_id']),
                             status=json.loads(attempt_status.read_text()) if attempt_status.exists() else None))
    leases = {p.parent.name: json.loads(p.read_text()) for p in (campaign / 'leases').glob('*/LEASE.json')}
    print(json.dumps(dict(attempts=rows, leases=leases), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('init')
    i.add_argument('--campaign-root', required=True)
    i.add_argument('--source-commit', default='HEAD')
    i.add_argument('--env-dir', required=True)
    i.add_argument('--cuda-home', required=True)
    i.add_argument('--cxx', required=True)
    i.add_argument('--model-path', required=True)
    i.add_argument('--manifest', required=True)
    i.add_argument('--fixtures', required=True)
    i.add_argument('--hf-repo', default='anhdao69/Streaming_episodic')
    i.add_argument('--hf-home', default=os.environ.get('HF_HOME', ''))
    i.add_argument('--numerical-report', required=True, nargs='+',
                   help='Passing numerical report(s) with source_commit/runtime_sha256, one per runtime profile')
    i.add_argument('--no-upload', action='store_true', help='Keep checkpoints local; do not publish to the Hub')
    i.add_argument('--disk-reservation-gib', type=float, default=400)
    s = sub.add_parser('submit')
    s.add_argument('--campaign-root', required=True)
    s.add_argument('--run', required=True)
    s.add_argument('--arm')
    s.add_argument('--profiles', nargs='+', choices=PROFILES, required=True)
    s.add_argument('--qualify', choices=['full', 'startup'], default='startup')
    s.add_argument('--ablations', nargs='*', help='Runs the verified winner submits on its profile')
    s.add_argument('--mem', type=lambda x: dict(kv.split('=') for kv in x.split(',')))
    s.add_argument('--partition', default='highgpu')
    s.add_argument('--cpus', type=per_profile(int), help="'32' or 'H4=32,H2=30'")
    s.add_argument('--omp', type=per_profile(int), help="'4' or 'H4=4,H2=8'")
    s.add_argument('--dry-run', action='store_true')
    s.add_argument('--allow-phase2', action='store_true')
    s.add_argument('--phase2-reason', help='Why the Phase-2 gate is overridden (recorded verbatim)')
    s.add_argument('--allow-existing-lease', action='store_true')
    r = sub.add_parser('resnapshot')
    r.add_argument('--campaign-root', required=True)
    r.add_argument('--source-commit', required=True)
    r.add_argument('--reason', required=True)
    r.add_argument('--numerical-report', required=True, nargs='+')
    t = sub.add_parser('status')
    t.add_argument('--campaign-root', required=True)
    args = p.parse_args()
    dict(init=init, submit=submit, status=status, resnapshot=resnapshot)[args.command](args)


if __name__ == '__main__':
    main()
