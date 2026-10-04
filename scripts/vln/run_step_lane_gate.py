"""Run an owned validation command and record job-level host RAM.

The optional RAM guard terminates only this command's new process group. It never
cancels a Slurm allocation or signals pre-existing training processes.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def memory_sample(job):
    """Conservative working set: retain dirty/writeback cache in the guard.

    memory.current includes clean inactive filesystem cache, which Linux can
    reclaim under pressure. Record total usage too; never change cgroup limits
    or drop caches shared with other work.
    """
    used = int((job / 'memory.current').read_text())
    stats = dict(line.split() for line in (job / 'memory.stat').read_text().splitlines())
    clean_inactive = max(0, int(stats.get('inactive_file', 0))
                         - int(stats.get('file_dirty', 0))
                         - int(stats.get('file_writeback', 0)))
    return dict(job_host_memory_gib=used / 2**30,
                guard_working_set_gib=max(0, used - clean_inactive) / 2**30,
                clean_inactive_file_gib=clean_inactive / 2**30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--host-guard-gib',type=float,default=440)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:
        parser.error('a command is required after --')
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    relative=Path('/proc/self/cgroup').read_text().splitlines()[0].split(':',2)[2]
    current=Path('/sys/fs/cgroup')/relative.lstrip('/')
    job=next((p for p in (current,*current.parents) if p.name.startswith('job_')),current)
    maximum=0.
    working_maximum=0.
    started=time.time()
    guard=False
    with (out/'command.log').open('w') as log,(out/'host_memory.jsonl').open('w') as memory:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            sample=memory_sample(job)
            used=sample['job_host_memory_gib']
            maximum=max(maximum,used)
            working_maximum=max(working_maximum,sample['guard_working_set_gib'])
            memory.write(json.dumps(dict(unix_time=time.time(),**sample))+'\n')
            memory.flush()
            if sample['guard_working_set_gib']>args.host_guard_gib:
                guard=True
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                break
            time.sleep(2)
        status=process.wait()
    report=dict(status='PASS' if status==0 and not guard else 'FAIL',exit_code=status,
                host_guard_triggered=guard,sampled_peak_job_host_gib=maximum,
                sampled_peak_guard_working_set_gib=working_maximum,
                seconds=time.time()-started,command=command,job_cgroup=str(job))
    (out/'resources.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    raise SystemExit(0 if report['status']=='PASS' else 1)


if __name__=='__main__':
    main()
