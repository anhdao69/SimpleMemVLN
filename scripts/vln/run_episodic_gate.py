"""Neutral owned-command supervisor: job-level host RAM, OOM counters, GPU memory.

It starts one command in a new process group, samples the enclosing Slurm job
cgroup (v1 or v2) about every second, and samples per-GPU memory.  The guard
sends SIGTERM (then SIGKILL) only to this command's own process group.  It
never changes cgroup limits, drops caches, cancels an allocation or touches
pre-existing processes.  Exit status 0 only for exit 0 with no guard trigger
and no new OOM or OOM-kill event.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from qwen_vl.train.host_memory import cgroup_sample, guard_limit_gib, job_cgroup  # noqa: E402


def gpu_sample():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
        rows = [line.split(', ') for line in out.stdout.strip().splitlines()]
        return [dict(index=int(r[0]), used_mib=int(r[1]), total_mib=int(r[2]), util=int(r[3])) for r in rows]
    except Exception as exc:     # sampling must never kill the command
        return [dict(error=repr(exc)[:120])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--host-guard-gib', type=float,
                        help='Working-set guard; default min(0.9*limit, limit-48GiB)')
    parser.add_argument('--min-free-disk-gib', type=float, default=0)
    parser.add_argument('--disk-path', default='.')
    parser.add_argument('--interval', type=float, default=1.0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a command is required after --')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    location = job_cgroup()
    first = cgroup_sample(location)
    guard_gib = args.host_guard_gib or guard_limit_gib(first['job_limit_gib'])
    free = shutil.disk_usage(args.disk_path).free / 2**30
    if free < args.min_free_disk_gib:
        report = dict(status='FAIL', reason=f'free disk {free:.1f} GiB < {args.min_free_disk_gib}')
        (out / 'resources.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        raise SystemExit(1)
    peaks = dict(job_usage_gib=0., working_set_gib=0., anon_gib=0.)
    gpu_peak = {}
    started, guard, last_gpu = time.time(), False, 0.
    new_oom = new_oom_kill = 0
    with (out / 'command.log').open('w') as log, (out / 'host_memory.jsonl').open('w') as samples:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (out / 'owned_process.json').write_text(json.dumps(dict(
            pid=process.pid, pgid=os.getpgid(process.pid), command=command,
            slurm_job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename,
            started=started)) + '\n')
        while process.poll() is None:
            sample = cgroup_sample(location)
            new_oom = max(new_oom, sample['oom'] - first['oom'])
            new_oom_kill = max(new_oom_kill, sample['oom_kill'] - first['oom_kill'])
            for key in peaks:
                peaks[key] = max(peaks[key], sample[key])
            record = dict(unix_time=time.time(), **sample)
            if time.time() - last_gpu > 5:
                last_gpu = time.time()
                gpus = gpu_sample()
                record['gpus'] = gpus
                for g in gpus:
                    if 'index' in g:
                        gpu_peak[g['index']] = max(gpu_peak.get(g['index'], 0), g['used_mib'])
            samples.write(json.dumps(record) + '\n')
            samples.flush()
            guard = sample['working_set_gib'] > guard_gib
            if guard or new_oom or new_oom_kill:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(args.interval)
        status = process.wait()
    last = cgroup_sample(location)
    new_oom = max(new_oom, last['oom'] - first['oom'])
    new_oom_kill = max(new_oom_kill, last['oom_kill'] - first['oom_kill'])
    report = dict(
        status='PASS' if status == 0 and not guard and new_oom == 0 and new_oom_kill == 0 else 'FAIL',
        exit_code=status, host_guard_triggered=guard, host_guard_gib=guard_gib,
        job_limit_gib=first['job_limit_gib'], sampled_peak_job_usage_gib=peaks['job_usage_gib'],
        sampled_peak_working_set_gib=peaks['working_set_gib'], sampled_peak_anon_gib=peaks['anon_gib'],
        cgroup_recorded_peak_gib=last['job_peak_gib'], new_oom_events=new_oom, new_oom_kill_events=new_oom_kill,
        new_limit_events=last['limit_events'] - first['limit_events'],
        start_usage_gib=first['job_usage_gib'], nvidia_smi_peak_used_mib=gpu_peak,
        seconds=time.time() - started, command=command, cgroup=str(location[1]),
        cgroup_version=location[0], free_disk_gib_at_start=free)
    (out / 'resources.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
