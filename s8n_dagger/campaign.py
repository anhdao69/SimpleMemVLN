"""Durable per-host queue for disjoint S8N collection shards."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from s8n_dagger.common import atomic_json, bind_contract, sha256
from s8n_dagger.run import command, environment
from s8n_dagger.validate import validate


def active_collectors(root, proc_root=Path('/proc')):
    prefix = str(Path(root) / 'full') + '/'
    result = []
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode().split('\0')
            if 's8n_dagger.collect' in argv and '--out' in argv:
                out = argv[argv.index('--out') + 1]
                if out.startswith(prefix):
                    result.append(int(entry.name))
        except (OSError, UnicodeDecodeError, IndexError):
            continue
    return sorted(result)


def validate_completed_shard(out, episode_ids):
    contract = json.loads((out / 'contract.json').read_text())
    if contract['episode_ids'] != episode_ids:
        raise ValueError(f'Completed shard episode IDs differ from assignment: {out.name}')
    result = validate(out)
    if not result['complete']:
        raise ValueError(f'Incomplete validated shard {out.name}')
    atomic_json(out / 'validation.json', result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', required=True)
    parser.add_argument('--assignment', required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError('Expected1..8 workers per96GB GPU')
    site = json.loads(Path(args.site).read_text())
    jobs = json.loads(Path(args.assignment).read_text())
    root = Path(site['root'])
    queue = root / 'queue'
    queue.mkdir(exist_ok=True)
    lock = (queue / '.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    surviving = active_collectors(root)
    if surviving:
        raise RuntimeError(f'Prior campaign collectors still alive: {surviving}; refuse overlapping restart')
    keys = [(j['dataset'], j['shard']) for j in jobs]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate assignment')
    shard_inputs, assigned_ids = {}, set()
    for job in jobs:
        dataset, shard = job['dataset'], job['shard']
        if dataset not in ('r2r', 'rxr15') or type(shard) is not int or shard < 0:
            raise ValueError('Invalid dataset/shard assignment')
        key = f'{dataset}_{shard:03}'
        content = (root / 'shards' / dataset / f'{shard:03}.json').read_bytes()
        ids = json.loads(content)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f'Invalid shard episode IDs: {key}')
        ids = [str(eid) for eid in ids]
        for eid in ids:
            identity = (dataset, eid)
            if identity in assigned_ids:
                raise ValueError(f'Duplicate dataset episode assignment: {dataset}:{eid}')
            assigned_ids.add(identity)
        shard_inputs[key] = dict(episode_ids=ids, sha256=hashlib.sha256(content).hexdigest())
    bind_contract(queue, dict(site=site, jobs=jobs, shard_inputs=shard_inputs,
                  source={p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}))
    pending, finished, failed, running = [], [], [], {}
    for job in jobs:
        key = f"{job['dataset']}_{job['shard']:03}"
        out = root / 'full' / key
        job = dict(job, key=key, out=str(out), episode_ids=shard_inputs[key]['episode_ids'])
        status_path = out / 'status.json'
        if status_path.exists() and json.loads(status_path.read_text()).get('complete'):
            validate_completed_shard(out, job['episode_ids'])
            finished.append(job)
        else:
            pending.append(job)
    def stop(_signum, _frame):
        raise KeyboardInterrupt('Campaign interrupted')
    signal.signal(signal.SIGTERM, stop)
    next_launch = 0.
    try:
        while pending or running:
            if pending and len(running) < args.workers and time.monotonic() >= next_launch:
                job = pending.pop(0)
                episode_list = root / 'shards' / job['dataset'] / f"{job['shard']:03}.json"
                if sha256(episode_list) != shard_inputs[job['key']]['sha256']:
                    raise ValueError(f'Shard input changed before launch: {job["key"]}')
                log = (queue / (job['key'] + '.log')).open('a')
                argv = command(site, job['dataset'], episode_list, job['out'])
                process = subprocess.Popen(argv, cwd=site['repo'], env=environment(site),
                                           stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                running[process.pid] = (process, job, log)
                next_launch = time.monotonic() + 25 # stagger large CPU/GPU model loads
            for pid, (process, job, log) in list(running.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    if code:
                        failed.append(dict(job, returncode=code))
                    else:
                        result = json.loads((Path(job['out']) / 'status.json').read_text())
                        if not result['complete']:
                            failed.append(dict(job, returncode=code, error='incomplete output'))
                        else:
                            try:
                                validate_completed_shard(Path(job['out']), job['episode_ids'])
                                finished.append(job)
                            except Exception as exc:
                                failed.append(dict(job, returncode=1, error=f'Validation: {exc}'))
                    del running[pid]
            statuses = {}
            for job in jobs:
                key = f"{job['dataset']}_{job['shard']:03}"
                p = root / 'full' / key / 'status.json'
                if p.exists():
                    statuses[key] = json.loads(p.read_text())
            atomic_json(root / 'campaign_status.json', dict(
                phase='running', pid=os.getpid(), updated_unix=time.time(),
                workers=args.workers, jobs_total=len(jobs), jobs_done=len(finished),
                jobs_pending=len(pending), failed_jobs=failed,
                running={str(pid): job['key'] for pid, (_, job, _) in running.items()},
                attempted=sum(s['attempted'] for s in statuses.values()),
                accepted=sum(s['accepted'] for s in statuses.values()),
                accepted_steps=sum(s['accepted_steps'] for s in statuses.values()),
                failures=sum(s['failures'] for s in statuses.values()), shards=statuses))
            time.sleep(5)
        # Final manifest uses validated per-shard exports and preserves host paths.
        with (root / 'training.jsonl.tmp').open('w') as destination:
            for job in sorted(finished, key=lambda j: j['key']):
                destination.write((Path(job['out']) / 'training.jsonl').read_text())
            destination.flush()
            os.fsync(destination.fileno())
        (root / 'training.jsonl.tmp').replace(root / 'training.jsonl')
        final_shards = {j['key']: json.loads((Path(j['out']) / 'status.json').read_text()) for j in finished}
        status = dict(phase='failed' if failed else 'complete', pid=os.getpid(),
                      workers=args.workers, jobs_total=len(jobs), jobs_done=len(finished),
                      jobs_pending=0, running={}, failed_jobs=failed, shards=final_shards,
                      attempted=sum(s['attempted'] for s in final_shards.values()),
                      accepted=sum(s['accepted'] for s in final_shards.values()),
                      accepted_steps=sum(s['accepted_steps'] for s in final_shards.values()),
                      failures=sum(s['failures'] for s in final_shards.values()),
                      updated_unix=time.time(), training_manifest=str(root / 'training.jsonl'))
        atomic_json(root / 'campaign_status.json', status)
        if failed:
            raise RuntimeError(f'{len(failed)} shards failed; inspect queue logs')
    finally:
        error = sys.exc_info()[1]
        try:
            for process, _, log in running.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                log.close()
        finally:
            error = error or sys.exc_info()[1]
            if error is not None:
                status_path = root / 'campaign_status.json'
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                status.update(phase='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                              pid=os.getpid(), updated_unix=time.time(), running={},
                              jobs_total=len(jobs), jobs_done=len(finished),
                              jobs_pending=len(pending) + len(running), failed_jobs=failed,
                              controller_error=f'{type(error).__name__}: {error}')
                atomic_json(status_path, status)


if __name__ == '__main__':
    main()
