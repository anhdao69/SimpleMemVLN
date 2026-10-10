"""Memory-aware placement of training episodes across ranks (systems only; the update is unchanged).

Each rank owns the episodes it rolled out, but the GPUs are shared with other users' jobs, so
their free memory differs and changes over time. Before the update, every rank reports its free
device memory; the longest episodes are placed on ranks where their activations fit, their frames
are moved with NCCL point-to-point transfers, and each rank decides per episode whether to keep
checkpointed layer inputs in host memory. Every trained episode keeps its own advantage and loss
weight, and DeepSpeed sums gradients over ranks, so the applied gradient is identical.
"""
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


def redistribute(items, device):
    """items: this rank's [(episode, weight)]. Returns this rank's items after placement + stats."""
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info(device)[0]
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
    torch.cuda.empty_cache()
    return kept, dict(free_gib=[r['free'] / GIB for r in reports], moved=len(moved))


def should_offload(episode, first_item, device):
    free = torch.cuda.mem_get_info(device)[0]
    need = episode_tokens(episode) * ACTIVATION_BYTES_PER_TOKEN + ITEM_MARGIN_BYTES
    if first_item:
        need += STEP_RESERVE_BYTES
    return need > free
