"""Scene-stratified evaluation subset (proportional allocation, fixed seed) of an R2R-VLNCE split.

python scripts/rl/make_eval_subset.py DATASET_GZ SIZE SEED OUT.json
Used for pilot model selection on val_seen so that val_unseen is never used for tuning.
"""
import gzip
import json
import random
import sys
from collections import defaultdict


def main():
    path, size, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    with gzip.open(path, 'rt') as f:
        episodes = json.load(f)['episodes']
    by_scene = defaultdict(list)
    for ep in episodes:
        by_scene[ep['scene_id']].append(str(ep['episode_id']))
    rng = random.Random(seed)
    total = len(episodes)
    quota = {s: size * len(v) / total for s, v in by_scene.items()}
    take = {s: int(q) for s, q in quota.items()}
    for s in sorted(quota, key=lambda s: quota[s] - take[s], reverse=True)[:size - sum(take.values())]:
        take[s] += 1
    chosen = []
    for scene in sorted(by_scene):
        ids = sorted(by_scene[scene], key=int)
        chosen += rng.sample(ids, take[scene])
    assert len(chosen) == size == len(set(chosen))
    with open(out, 'w') as f:
        json.dump(sorted(chosen, key=int), f)
    print(len(chosen), 'episodes from', sum(1 for s in take if take[s]), 'scenes')


if __name__ == '__main__':
    main()
