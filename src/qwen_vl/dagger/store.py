"""Atomic frame folders and append-only shard journals for DAgger collection.

Layout under the collection root::

    frames/<folder>/step_0000_MOVE_FORWARD.png ...   one folder per saved episode
    journal.shard003.jsonl                           one line per attempted episode

The journal is the single source of truth.  A saved episode's frames are written to
``frames/.tmp-<folder>`` and renamed into place before its journal line (which carries the
manifest row) is appended; on resume, a frame folder of this shard without a journal line is a
crash leftover and is deleted.  Frames are lossless PNG of exactly the RGB the policy received.
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from qwen_vl.contracts import ACTIONS, validate_episode

FRAME_NAME = 'step_{:04d}_{}.png'
SHAPE = (480, 640, 3)


class CollectionStore:
    def __init__(self, root, shard):
        if type(shard) is not int or shard < 0:
            raise ValueError('Shard index must be a nonnegative integer')
        self.root = Path(root).resolve()        # manifest rgb_paths must not depend on the cwd
        self.frames = self.root / 'frames'
        self.frames.mkdir(parents=True, exist_ok=True)
        self.journal = self.root / f'journal.shard{shard:03d}.jsonl'

    def folder(self, episode_id):
        return self.frames / str(episode_id)

    def clean(self, episode_ids, completed):
        """Remove partial and unjournaled folders of this shard's episodes; returns removed paths."""
        removed = []
        for episode_id in sorted(set(map(str, episode_ids))):
            final = self.folder(episode_id)
            partial = self.frames / f'.tmp-{episode_id}'
            if partial.exists():
                shutil.rmtree(partial)
                removed.append(str(partial))
            if final.exists() and episode_id not in completed:
                shutil.rmtree(final)
                removed.append(str(final))
        return removed

    def save_frames(self, episode_id, frames, actions):
        """Write ``frames[t]`` as ``step_{t}_{ACTIONS[actions[t]]}.png``; returns the final paths."""
        if len(frames) != len(actions) or not frames:
            raise ValueError('Every saved frame needs exactly one executed action')
        if actions[-1] != ACTIONS.index('STOP') or ACTIONS.index('STOP') in actions[:-1]:
            raise ValueError('A saved trajectory has exactly one final STOP')
        final = self.folder(episode_id)
        partial = self.frames / f'.tmp-{episode_id}'
        if final.exists() or partial.exists():
            raise FileExistsError(f'Frame folder for episode {episode_id} already exists')
        partial.mkdir()
        names = []
        for t, (rgb, action) in enumerate(zip(frames, actions)):
            rgb = np.asarray(rgb)
            if rgb.shape != SHAPE or rgb.dtype != np.uint8:
                raise ValueError(f'Frame {t}: expected uint8 {SHAPE}, got {rgb.dtype} {rgb.shape}')
            name = FRAME_NAME.format(t, ACTIONS[action])
            Image.fromarray(rgb).save(partial / name, format='PNG')
            names.append(name)
        partial.rename(final)
        return [str(final / name) for name in names]

    def append(self, record):
        line = json.dumps(record, allow_nan=False) + '\n'
        with self.journal.open('a') as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def manifest_row(tag, split, source, episode_id, paths, actions, steps, summary):
    """Training-manifest row (``scripts/vln/prepare_r2r.py`` schema) for one saved episode.

    ``source``: dict(scene_id, instruction_id, instruction, dataset_version) taken from the same
    R2R json the oracle manifest was built from, so instruction text and IDs match exactly.
    """
    episode_id = str(episode_id)
    if not (len(paths) == len(actions) == len(steps)):
        raise ValueError('Paths, actions and step provenance must align')
    row = dict(
        schema_version=1,
        episode_uid=f"r2rce_dagger:{tag}:{split}:{source['scene_id']}:{episode_id}:{source['instruction_id']}",
        dataset='r2rce_dagger', dataset_version=source['dataset_version'], split=split,
        scene_id=source['scene_id'], episode_id=episode_id, instruction_id=source['instruction_id'],
        instruction=source['instruction'],
        observation_action_alignment='observation_before_action',
        alignment_evidence='dagger collector: frame t is the observation the executed action t was chosen from',
        dagger=dict(tag=tag, **summary),
        steps=[dict(step_id=t, rgb_path=path, action_name=ACTIONS[action], is_valid=True, dagger=meta)
               for t, (path, action, meta) in enumerate(zip(paths, actions, steps))],
    )
    validate_episode(row)
    return row
