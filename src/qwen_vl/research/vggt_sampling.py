"""Fixed instruction-independent frame samples; original serialized positions."""
import hashlib
import json
import torch


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def physical_trajectory_key(episode):
    # Ignore instruction/episode ID. Preserve ordered physical view identities.
    paths = [s['rgb_path'].split('/trajectory_data/')[-1] for s in episode['steps']]
    return stable_hash([episode['dataset'], episode['scene_id'], paths])


def sample_frames(key, length, seed=429):
    if length < 1:
        raise ValueError('Cannot sample an empty trajectory')
    count = min(16, length)
    return tuple(j * length // count + int(stable_hash([key, j, seed]), 16) %
                 ((j + 1) * length // count - j * length // count) for j in range(count))


def sampled_visual_positions(input_ids, image_id, spans, frames, tokens_per_frame=300):
    rows = []
    for frame in frames:
        start, end = spans[frame]
        indices = input_ids[0, start:end].eq(image_id).nonzero().flatten() + start
        if len(indices) != tokens_per_frame:
            raise ValueError('Sampled frame has incorrect actual image-token count')
        rows.append(indices)
    return torch.stack(rows) if rows else input_ids.new_empty((0, tokens_per_frame))
