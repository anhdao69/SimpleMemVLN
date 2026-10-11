"""Job-level host-memory accounting for cgroup v1 and v2 (read-only).

Working set (heuristic, not reclaimability proof):

    clean_inactive = max(0, inactive_file - dirty - writeback)
    working_set    = max(0, usage - clean_inactive)

OOM evidence: v2 ``memory.events`` (oom, oom_kill); v1 ``memory.oom_control``
(oom_kill) and ``memory.failcnt`` (limit hits, recorded, not called OOM).
Nothing here changes limits, drops caches or signals anything.
"""
import os
from pathlib import Path

GIB = 2 ** 30


def _job_dir(base, relative):
    current = Path(base) / relative.lstrip('/')
    for path in (current, *current.parents):
        if path.name.startswith('job_'):
            return path
    return current


def job_cgroup():
    """(version, directory) of the enclosing Slurm job memory cgroup."""
    lines = Path('/proc/self/cgroup').read_text().splitlines()
    for line in lines:
        _, controllers, relative = line.split(':', 2)
        if 'memory' in controllers.split(','):
            return 1, _job_dir('/sys/fs/cgroup/memory', relative)
    for line in lines:
        if line.startswith('0::'):
            return 2, _job_dir('/sys/fs/cgroup', line.split(':', 2)[2])
    raise RuntimeError('No memory cgroup found')


def _stat(path):
    return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}


def cgroup_sample(location=None):
    version, job = location or job_cgroup()
    if version == 1:
        usage = int((job / 'memory.usage_in_bytes').read_text())
        limit = int((job / 'memory.limit_in_bytes').read_text())
        peak = int((job / 'memory.max_usage_in_bytes').read_text())
        stats = _stat(job / 'memory.stat')
        inactive = stats.get('total_inactive_file', 0)
        dirty = stats.get('total_dirty', 0) + stats.get('total_writeback', 0)
        anon = stats.get('total_rss', 0)
        unevictable = stats.get('total_unevictable', 0)
        swap = stats.get('total_swap', 0)
        oom = _stat_text(job / 'memory.oom_control')
        events = dict(oom_kill=oom.get('oom_kill', 0), under_oom=oom.get('under_oom', 0),
                      failcnt=int((job / 'memory.failcnt').read_text()))
    else:
        usage = int((job / 'memory.current').read_text())
        raw = (job / 'memory.max').read_text().strip()
        limit = 2 ** 63 if raw == 'max' else int(raw)
        peak_file = job / 'memory.peak'
        peak = int(peak_file.read_text()) if peak_file.exists() else usage
        stats = _stat(job / 'memory.stat')
        inactive = stats.get('inactive_file', 0)
        dirty = stats.get('file_dirty', 0) + stats.get('file_writeback', 0)
        anon = stats.get('anon', 0)
        unevictable = stats.get('unevictable', 0)
        swap = int((job / 'memory.swap.current').read_text()) if (job / 'memory.swap.current').exists() else 0
        events = _stat(job / 'memory.events')
    clean_inactive = max(0, inactive - dirty)
    return dict(cgroup_version=version, job_usage_gib=usage / GIB, job_limit_gib=limit / GIB,
                job_peak_gib=peak / GIB, working_set_gib=max(0, usage - clean_inactive) / GIB,
                clean_inactive_file_gib=clean_inactive / GIB, anon_gib=anon / GIB,
                unevictable_gib=unevictable / GIB, dirty_writeback_gib=dirty / GIB,
                swap_gib=swap / GIB, oom_kill=int(events.get('oom_kill', 0)),
                oom=int(events.get('oom', events.get('under_oom', 0))),
                limit_events=int(events.get('failcnt', events.get('max', 0))))


def _stat_text(path):
    return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}


def process_rss_gib(pid='self'):
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return int(line.split()[1]) * 1024 / GIB
    return 0.0


def guard_limit_gib(limit_gib):
    """Owned-command guard: min(0.9 * limit, limit - 48 GiB) for large allocations."""
    return min(0.9 * limit_gib, limit_gib - 48) if limit_gib > 96 else 0.9 * limit_gib
