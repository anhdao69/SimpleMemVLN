"""Memory-aware placement of training episodes across ranks (systems only; the update is unchanged).

Each rank owns the episodes it rolled out, but the GPUs are shared with other users' jobs, so
their free memory differs and changes over time. Before the update, every rank reports its free
device memory; the longest episodes are placed on ranks where their activations fit, their frames
are moved with NCCL point-to-point transfers, and each rank decides per episode whether to keep
checkpointed layer inputs in host memory. Every trained episode keeps its own advantage and loss
weight, and DeepSpeed sums gradients over ranks, so the applied gradient is identical.
"""
import time

import numpy as np
import torch
import torch.distributed as dist

GIB = 2 ** 30
# Measured on Qwen3.5-4B Window8 (fwd+bwd with decoder checkpointing): ~0.5 MiB per token, of which
# ~0.16 MiB are checkpointed layer inputs that host offload removes. Slightly conservative here.
ACTIVATION_BYTES_PER_TOKEN = 0.56 * 2 ** 20
OFFLOADED_BYTES_PER_TOKEN = 0.40 * 2 ** 20
STEP_RESERVE_BYTES = 10 * GIB   # fp32 grad partition (8.4 GiB) allocated during the first backward + logits
ITEM_MARGIN_BYTES = 2 * GIB


# GPU reservation (runtime.hold_gpu_gib): the GPUs are shared, and memory this process hands back
# between steps can be taken by other jobs, after which the update runs out of memory. While a target
# is set, the process keeps its cached blocks instead of returning them to the driver.
HOLD = {'target': 0, 'helper_reserve': 0}


def release_cache():
    """torch.cuda.empty_cache(), unless this process holds a GPU reservation."""
    if not HOLD['target']:
        torch.cuda.empty_cache()


def hold_memory(device, chunk=GIB):
    """Grow this process's reserved GPU memory towards HOLD['target'] without waiting.

    Leaves HOLD['helper_reserve'] free on the device for this rank's rollout helper processes.
    Returns the reserved bytes afterwards.
    """
    blocks = []
    try:
        while (torch.cuda.memory_reserved(device) < HOLD['target']
               and torch.cuda.mem_get_info(device)[0] > HOLD['helper_reserve'] + chunk):
            blocks.append(torch.empty(chunk, dtype=torch.uint8, device=device))
    except torch.OutOfMemoryError:
        pass
    del blocks  # freed into this process's cache, which release_cache() no longer empties
    return torch.cuda.memory_reserved(device)


def start_hold(device, target_bytes, helper_reserve_bytes, log, wait_seconds, poll_seconds=30):
    """Set the reservation and grow it, waiting up to wait_seconds for other jobs to free memory."""
    HOLD['target'], HOLD['helper_reserve'] = target_bytes, helper_reserve_bytes
    start = time.time()
    while hold_memory(device) < target_bytes and time.time() - start < wait_seconds:
        time.sleep(poll_seconds)
    reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)  # the reservation blocks are not part of the step peaks
    log(f'GPU reservation {reserved / GIB:.1f} of {target_bytes / GIB:.0f} GiB '
        f'(waited {time.time() - start:.0f} s; topped up before every update)')
    return reserved


def episode_tokens(episode):
    """Upper estimate of the training sequence length (prefix + 300 image + text tokens per step)."""
    return 320 * len(episode['_responses']) + 400


def plan(free_bytes, lengths):
    """free_bytes[r], lengths[r][i] -> {(r, i): target rank}; deterministic on every rank."""
    ranks = range(len(free_bytes))
    order = sorted(((t, r, i) for r in ranks for i, t in enumerate(lengths[r])), key=lambda x: (-x[0], x[1], x[2]))
    load = [0] * len(free_bytes)
    assignment = {}
    for tokens, owner, index in order:
        need = tokens * ACTIVATION_BYTES_PER_TOKEN + STEP_RESERVE_BYTES
        fits = [q for q in ranks if free_bytes[q] >= need]
        candidates = fits or [max(ranks, key=lambda q: (free_bytes[q], -q))]
        target = min(candidates, key=lambda q: (load[q], q != owner, q))
        assignment[(owner, index)] = target
        load[target] += tokens
    return assignment


def redistribute(items, device, margin_bytes=0):
    """items: this rank's [(episode, weight)]. Returns this rank's items after placement + stats.

    margin_bytes is withheld from every rank's reported free memory, so that a GPU whose free memory
    other users' jobs keep changing only takes episodes it can fit with room to spare.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    release_cache()
    free = max(device_free(device) - margin_bytes, 0)
    reports = [None] * world
    dist.all_gather_object(reports, dict(free=free, lengths=[episode_tokens(e) for e, _ in items]))
    assignment = plan([r['free'] for r in reports], [r['lengths'] for r in reports])
    moved = sorted(key for key, target in assignment.items() if target != key[0])
    # Small metadata for moved episodes travels with all_gather_object; frames with NCCL p2p.
    outgoing = {i: {k: v for k, v in items[i][0].items() if k != '_frames'} | dict(_weight=items[i][1],
               _frame_count=len(items[i][0]['_frames'])) for (owner, i) in moved if owner == rank}
    metadata = [None] * world
    dist.all_gather_object(metadata, outgoing)
    ops, sends, receives = [], [], []
    for owner, i in moved:
        target = assignment[(owner, i)]
        if owner == rank:
            frames = torch.from_numpy(np.stack(items[i][0]['_frames'])).to(device)
            sends.append(frames)
            ops.append(dist.P2POp(dist.isend, frames, target))
        elif target == rank:
            count = metadata[owner][i]['_frame_count']
            buffer = torch.empty((count, 480, 640, 3), dtype=torch.uint8, device=device)
            receives.append((owner, i, buffer))
            ops.append(dist.P2POp(dist.irecv, buffer, owner))
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()
    torch.cuda.synchronize(device)
    kept = [items[i] for i in range(len(items)) if assignment[(rank, i)] == rank]
    for owner, i, buffer in receives:
        episode = dict(metadata[owner][i])
        weight = episode.pop('_weight')
        episode.pop('_frame_count')
        episode['_frames'] = list(buffer.cpu().numpy())
        kept.append((episode, weight))
    del sends, receives
    release_cache()
    return kept, dict(free_gib=[r['free'] / GIB for r in reports], moved=len(moved))


def wait_for_free(device, min_bytes, log, poll_seconds=60):
    """Before the update, wait until every rank's GPU has min_bytes free instead of running into OOM.

    The GPUs are shared: other users' jobs can take memory that the update (ZeRO-2 gradient partition,
    activations, optimizer temporaries) needs. Waiting costs time; an OOM costs the steps since the
    last checkpoint plus a restart. Returns the seconds waited.
    """
    waited, last_log = 0, -1
    while True:
        release_cache()
        if HOLD['target']:
            hold_memory(device)  # take memory other jobs have freed since the last poll
        frees = [None] * dist.get_world_size()
        dist.all_gather_object(frees, device_free(device))
        if min(frees) >= min_bytes:
            return waited
        if waited // 600 != last_log:
            last_log = waited // 600
            log(f'waiting for {min_bytes / GIB:.0f} GiB free on every GPU before the update '
                f'(free {[round(f / GIB, 1) for f in frees]} GiB, waited {waited} s)')
        time.sleep(poll_seconds)
        waited += poll_seconds


HOST_SAFETY_BYTES = 16 * GIB


def device_free(device):
    """Free device memory including blocks this process's caching allocator holds but does not use."""
    return (torch.cuda.mem_get_info(device)[0] + torch.cuda.memory_reserved(device)
            - torch.cuda.memory_allocated(device))


def host_available():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    return 0


def should_offload(episode, first_item, device):
    """Offload checkpointed layer inputs only if the GPU is short and the host can take them."""
    tokens = episode_tokens(episode)
    need = tokens * ACTIVATION_BYTES_PER_TOKEN + ITEM_MARGIN_BYTES
    if first_item:
        need += STEP_RESERVE_BYTES
    if need <= device_free(device):
        return False
    host_cost = tokens * (ACTIVATION_BYTES_PER_TOKEN - OFFLOADED_BYTES_PER_TOKEN)
    return host_available() - host_cost > HOST_SAFETY_BYTES
