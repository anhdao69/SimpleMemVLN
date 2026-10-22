#!/usr/bin/env python3
"""Keep the VRAM our next RL training needs while only our evaluations (or nothing of ours) run.

The trainer reserves its own memory (runtime.hold_gpu_gib) only while it runs; between runs (evaluations,
post-processing, queue gaps) other users' jobs took the freed memory and the next training waited or hit
CUDA OOM. ``run`` (unit rl-gpu-keeper) holds 1 GiB blocks so that, per GPU,

    keeper + our other processes + pending requests + 2 GiB per our evaluation model server = target,

never takes more than ``target`` (targets.json, our training footprint), leaves pending requests free, and 1 GiB
more so other users' running jobs can still grow. It holds nothing while one of our trainers runs on that GPU
(the trainer's update peaks exceed its reservation), and releases a GPU as soon as our trainer appears
on it, or 90 s after our torchrun launcher started.

``request`` (called by eval_export.sh before the evaluator starts): ask the keeper for GIB on a GPU for
TTL seconds and wait (at most WAIT s) until that much is free. Never fails the caller.

    gpu_keeper.py run
    gpu_keeper.py request --gpu 0 --gib 20 --name rl-eval-x-0 [--ttl 1200] [--wait 600]
    targets.json: {"0": 74, "1": 70, "paused": false}   (GiB; edited live)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KEEP = Path('/data/anh/rl_baseline/gpu_keeper')
GIB = 2 ** 30
MARGIN = 1 * GIB             # always left free for other users' running jobs
SERVER_HEADROOM = 2 * GIB    # growth room per running evaluation model server of ours
CONTEXT = int(0.6 * GIB)     # the keeper's own CUDA context per GPU
LAUNCH_GRACE = 90            # s a launched trainer may take to appear on the GPU


def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout


def gpu_apps():
    index = {}
    for line in sh('nvidia-smi', '--query-gpu=uuid,index', '--format=csv,noheader').splitlines():
        uuid, i = [x.strip() for x in line.split(',')]
        index[uuid] = int(i)
    apps = []
    for line in sh('nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                   '--format=csv,noheader,nounits').splitlines():
        pid, uuid, mem = [x.strip() for x in line.split(',')]
        if uuid in index:
            apps.append((int(pid), index[uuid], int(mem) * 2 ** 20))
    return apps


def free_bytes(gpu):
    out = sh('nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits')
    return int(out.strip()) * 2 ** 20


def mine(pid):
    try:
        return os.stat(f'/proc/{pid}').st_uid == os.getuid()
    except OSError:
        return False


def cmdline(pid):
    try:
        return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except OSError:
        return ''


def pending_requests(now):
    pending = {}
    for path in (KEEP / 'requests').glob('*.json'):
        try:
            r = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if r['expires'] < now:
            path.unlink(missing_ok=True)
        else:
            pending[int(r['gpu'])] = pending.get(int(r['gpu']), 0) + int(r['gib'] * GIB)
    return pending


def log(message):
    with (KEEP / 'keeper.log').open('a') as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")


def run():
    import torch
    (KEEP / 'requests').mkdir(parents=True, exist_ok=True)
    gpus = list(range(torch.cuda.device_count()))
    held = {g: [] for g in gpus}
    launched_at, last = None, None
    log(f'keeper start pid {os.getpid()} gpus {gpus}')
    while True:
        now = time.time()
        targets = json.loads((KEEP / 'targets.json').read_text())
        apps = gpu_apps()
        launcher = subprocess.run(['pgrep', '-u', str(os.getuid()), '-f', 'qwen_vl.rl.train_grpo'],
                                  capture_output=True).returncode == 0
        launched_at = (launched_at or now) if launcher else None
        pending = pending_requests(now)
        state = {}
        for g in gpus:
            ours = [(pid, mem) for pid, gpu, mem in apps if gpu == g and pid != os.getpid() and mine(pid)]
            commands = {pid: cmdline(pid) for pid, _ in ours}
            training = any('qwen_vl.rl.train_grpo' in c for c in commands.values())
            servers = sum('qwen_vl.stream.server' in c for c in commands.values())
            if targets.get('paused') or training or (launcher and now - launched_at > LAUNCH_GRACE):
                want = 0
            else:
                want = (int(targets[str(g)] * GIB) - sum(m for _, m in ours) - pending.get(g, 0)
                        - servers * SERVER_HEADROOM - CONTEXT)
            want = max(0, want)
            changed = False
            while held[g] and len(held[g]) * GIB > want:
                held[g].pop()
                changed = True
            if changed:
                torch.cuda.empty_cache()
            # Pending requests stay free (not only out of the target): a request granted from memory
            # another of our jobs just freed was otherwise re-held before the evaluator loaded (OOM, 2026-10-22).
            free = free_bytes(g) if len(held[g]) * GIB + GIB <= want else 0
            while len(held[g]) * GIB + GIB <= want and free - GIB >= MARGIN + pending.get(g, 0):
                try:
                    held[g].append(torch.empty(GIB, dtype=torch.uint8, device=g))
                except RuntimeError:
                    break
                free -= GIB
            state[g] = dict(held_gib=len(held[g]), want_gib=round(want / GIB, 1), ours_gib=round(
                sum(m for _, m in ours) / GIB, 1), training=training, servers=servers,
                pending_gib=round(pending.get(g, 0) / GIB, 1))
        summary = {str(g): (s['held_gib'], s['training'], s['servers'], s['pending_gib']) for g, s in state.items()}
        if summary != last:
            log(json.dumps(dict(launcher=launcher, **{str(g): s for g, s in state.items()})))
            last = summary
        (KEEP / 'status.json').write_text(json.dumps(dict(time=now, launcher=launcher, gpus=state)) + '\n')
        time.sleep(2)


def request(args):
    (KEEP / 'requests').mkdir(parents=True, exist_ok=True)
    path = KEEP / 'requests' / f'{args.name}-g{args.gpu}.json'
    path.write_text(json.dumps(dict(gpu=args.gpu, gib=args.gib, expires=time.time() + args.ttl)))
    start = time.time()
    while time.time() - start < args.wait:
        try:
            alive = time.time() - json.loads((KEEP / 'status.json').read_text())['time'] < 30
        except (OSError, ValueError, KeyError):
            alive = False
        if not alive or free_bytes(args.gpu) >= args.gib * GIB:
            break
        time.sleep(3)
    print(f'gpu_keeper request {args.name} gpu {args.gpu}: {args.gib} GiB, free '
          f'{free_bytes(args.gpu) / GIB:.1f} GiB after {time.time() - start:.0f} s', file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('run')
    r = sub.add_parser('request')
    r.add_argument('--gpu', type=int, required=True)
    r.add_argument('--gib', type=float, required=True)
    r.add_argument('--name', required=True)
    r.add_argument('--ttl', type=float, default=1200)
    r.add_argument('--wait', type=float, default=600)
    args = p.parse_args()
    if args.command == 'run':
        run()
    else:
        try:
            request(args)
        except Exception as exc:          # never block an evaluation on the keeper
            print(f'gpu_keeper request failed: {exc}', file=sys.stderr)


if __name__ == '__main__':
    main()
