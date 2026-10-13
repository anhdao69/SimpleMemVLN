"""Collect small benchmark evidence; never copy model/optimizer tensors."""
import argparse
import json
import math
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root = Path(args.root)
    records = []
    for path in sorted((root / 'experiments').glob('*/result.json')):
        result = json.loads(path.read_text())
        ranks = [[json.loads(line) for line in f.read_text().splitlines()]
                 for f in sorted((path.parent / 'train').glob('profile_rank*.jsonl'))]
        result['check']['peak_allocated_gib'] = max(
            (r['peak_allocated_gib'] for rank in ranks for r in rank), default=None)
        # Intervals start at update 2. Dataset projections explicitly discard
        # both startup updates and include loader/teacher time between updates.
        intervals = result['intervals']
        if len(intervals) >= 7:
            warm = intervals[1:]
            result['warm_updates_from_3'] = dict(
                count=len(warm), mean_seconds=statistics.mean(warm),
                min_seconds=min(warm), max_seconds=max(warm))
        result['rank_profiles'] = ranks
        result.pop('source_status', None)
        records.append(result)
    indexed = {r['name']: r for r in records}
    estimates = {}
    for name, episodes in [('r2rce-representative', 10819), ('rxrce-representative', 19996), ('joint', 30815)]:
        r = indexed.get('online-h4-best-' + name)
        if r and r['passed']:
            seconds = r['warm_updates_from_3']['mean_seconds']
            updates = math.ceil(episodes / 8)
            estimates[name] = dict(episodes=episodes, updates_per_epoch=updates,
                warm_seconds_per_update=seconds, projected_epoch_hours=seconds*updates/3600,
                caveat='Short diagnostic projection; excludes startup/checkpoint time. Repeated representatives have warm image caches; joint uses the real manifest.')
    Path(args.out).write_text(json.dumps(dict(experiments=records, estimates=estimates), indent=2) + '\n')
    print(json.dumps(estimates, indent=2))


if __name__ == '__main__':
    main()
