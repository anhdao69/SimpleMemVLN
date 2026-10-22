"""Split the baseline's val_unseen episode list into shards, and merge finished shard journals.

The evaluator scores each episode independently (fresh model session and Habitat reset per
episode), so running the identical evaluator on disjoint episode lists in parallel and merging
the per-episode records reproduces the single-process denominator and metrics.

split: python scripts/rl/eval_shards.py split BASELINE_DIR N OUT_PREFIX   -> OUT_PREFIX{i}.json
merge: python scripts/rl/eval_shards.py merge BASELINE_DIR OUT_DIR SHARD_DIR...
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from qwen_vl.eval.metrics import summarize  # noqa: E402


def split(baseline, n, prefix):
    ids = json.loads((Path(baseline) / 'evaluation_contract.json').read_text())['episode_ids']
    size = -(-len(ids) // n)
    for i in range(n):
        Path(f'{prefix}{i}.json').write_text(json.dumps(ids[i * size:(i + 1) * size]))
    print(n, 'shards of', size)


def merge(baseline, out, shards):
    order = json.loads((Path(baseline) / 'evaluation_contract.json').read_text())['episode_ids']
    contracts, rows = [], {}
    for shard in map(Path, shards):
        summary = json.loads((shard / 'summary.json').read_text())
        if not summary.get('complete'):
            raise SystemExit(f'{shard} is incomplete')
        contract = json.loads((shard / 'evaluation_contract.json').read_text())
        contracts.append(contract)
        for line in (shard / 'episodes.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['episode_id'] in rows:
                raise SystemExit(f'duplicate episode {row["episode_id"]}')
            rows[row['episode_id']] = row
    strip = lambda c: {k: v for k, v in c.items() if k != 'episode_ids'}
    if any(strip(c) != strip(contracts[0]) for c in contracts):
        raise SystemExit('Shard contracts differ beyond their episode lists')
    if set(rows) != set(order):
        raise SystemExit(f'Shards cover {len(rows)} episodes, baseline has {len(order)}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = [rows[i] for i in order]
    (out / 'episodes.jsonl').write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records))
    (out / 'evaluation_contract.json').write_text(json.dumps(
        dict(strip(contracts[0]), episode_ids=order, merged_from=[str(s) for s in shards]), indent=2) + '\n')
    (out / 'summary.json').write_text(json.dumps(
        dict(summarize(records), scheduled_episodes=len(order), complete=True), indent=2) + '\n')
    print('merged', len(records), 'episodes into', out)


if __name__ == '__main__':
    if sys.argv[1] == 'split':
        split(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        merge(sys.argv[2], sys.argv[3], sys.argv[4:])
