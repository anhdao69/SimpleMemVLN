"""Path-independent fingerprint of a training manifest, for cross-cluster matching.

Byte SHA-256 differs between clusters because every step stores an absolute
``rgb_path``.  This hashes, in manifest order, each episode's uid, instruction,
action sequence and image file *names* (no directories), so two clusters can
confirm they train on identical episodes in identical order:

    python scripts/vln/manifest_fingerprint.py <manifest.jsonl>
"""
import hashlib
import json
import sys
from pathlib import PurePosixPath


def fingerprint(path):
    digest, episodes, actions = hashlib.sha256(), 0, 0
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            steps = [(s['action_name'], PurePosixPath(s['rgb_path']).name, s.get('is_valid', True))
                     for s in row['steps']]
            digest.update(json.dumps([row['episode_uid'], row['instruction'], steps]).encode() + b'\n')
            episodes += 1
            actions += len(steps)
    return dict(fingerprint_v1=digest.hexdigest(), episodes=episodes, steps=actions)


if __name__ == '__main__':
    print(json.dumps(fingerprint(sys.argv[1])))
