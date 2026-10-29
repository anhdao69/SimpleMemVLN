"""Paired comparison of two complete evaluator journals on the same episodes.

python scripts/rl/compare_evals.py --base BASE_DIR --new NEW_DIR [--name-base ..] [--name-new ..] --out OUT.json
BASE_DIR/NEW_DIR contain the evaluator's episodes.jsonl, summary.json, evaluation_contract.json.
Refuses incomplete journals or mismatched protocols (dataset, config, split, seed, cap, episodes).
"""
import argparse
import json
import math
import random
from pathlib import Path

PROTOCOL_KEYS = ('dataset_sha256', 'habitat_config_sha256', 'split', 'episode_ids', 'seed', 'protocol',
                 'max_steps', 'failure_policy', 'transport', 'simulator_version', 'scenes_dir',
                 'base_assets_sha256')
EVAL_SOURCES = ('eval/habitat_r2r.py', 'eval/metrics.py', 'stream/session.py', 'stream/server.py',
                'stream/cache.py', 'stream/positions.py', 'stream/window_attention.py', 'stream/transport.py',
                'models/nav_model.py', 'data/episode_serializer.py', 'data/data_qwen.py',
                'train/vln_runtime.py', 'contracts.py')


def load(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    contract = json.loads((directory / 'evaluation_contract.json').read_text())
    if not summary.get('complete'):
        raise SystemExit(f'{directory}: evaluation is not complete; refusing to compare')
    rows = {r['episode_id']: r for r in map(json.loads, (directory / 'episodes.jsonl').read_text().splitlines())}
    if len(rows) != summary['scheduled_episodes'] or set(rows) != set(contract['episode_ids']):
        raise SystemExit(f'{directory}: journal does not cover the scheduled episodes')
    return summary, contract, rows


def episode_metrics(row):
    m = row['metrics']
    failed = row['failed']
    return dict(success=0.0 if failed else float(m.get('success', 0.0)),
                spl=0.0 if failed else float(m.get('spl', 0.0)),
                ne=m.get('distance_to_goal'), osr=m.get('oracle_success'),
                tl=m.get('trajectory_length'), steps=len(row.get('actions', [])),
                forced=bool(row.get('forced_stop')), failed=failed,
                min_d=row.get('minimum_distance_to_goal'))


def category(e):
    if e['failed']:
        return 'invalid_or_failed'
    if e['success']:
        return 'success'
    if e['forced']:
        return 'forced_stop_at_cap'
    if e['osr']:
        return 'passed_goal_then_stopped_outside'  # reached <3 m at some step, stopped >=3 m
    if e['ne'] is not None and e['ne'] < 5.0:
        return 'near_miss_3_to_5m'
    return 'never_within_3m'


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap(diffs, iters=10000, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', required=True)
    p.add_argument('--new', required=True)
    p.add_argument('--name-base', default='base')
    p.add_argument('--name-new', default='new')
    p.add_argument('--out', required=True)
    p.add_argument('--allow-source-diff', action='store_true',
                   help='compare although evaluation sources differ (listed in the output as source_diff); only '
                        'after a separate check that the source change leaves every episode unchanged')
    args = p.parse_args()
    (sb, cb, rb), (sn, cn, rn) = load(args.base), load(args.new)
    protocol_diff = [k for k in PROTOCOL_KEYS if cb.get(k) != cn.get(k)]
    source_diff = [k for k in EVAL_SOURCES if cb['source_sha256'].get(k) != cn['source_sha256'].get(k)]
    if protocol_diff or (source_diff and not args.allow_source_diff):
        raise SystemExit(f'Protocol differs: {protocol_diff}; evaluation sources differ: {source_diff}')
    ids = cb['episode_ids']
    eb = {i: episode_metrics(rb[i]) for i in ids}
    en = {i: episode_metrics(rn[i]) for i in ids}
    table = {}
    for key, name in (('sr', 'SR'), ('spl', 'SPL'), ('navigation_error', 'NE'), ('oracle_success', 'OSR'),
                      ('trajectory_length', 'TL'), ('mean_episode_steps', 'steps'), ('forced_stops', 'forced_stops'),
                      ('invalid_responses', 'invalid'), ('failures', 'failures')):
        a, b = sb[key], sn[key]
        table[name] = dict(base=a, new=b, abs_delta=None if a is None or b is None else b - a,
                           rel_delta=None if not a or b is None else (b - a) / a)
    both = sum(bool(eb[i]['success'] and en[i]['success']) for i in ids)
    only_base = sum(bool(eb[i]['success'] and not en[i]['success']) for i in ids)
    only_new = sum(bool(en[i]['success'] and not eb[i]['success']) for i in ids)
    neither = len(ids) - both - only_base - only_new
    ci = {name: bootstrap([en[i][k] - eb[i][k] for i in ids]) for name, k in (('SR', 'success'), ('SPL', 'spl'))}
    cats = sorted({category(e) for e in list(eb.values()) + list(en.values())})
    taxonomy = {c: dict(base=sum(category(eb[i]) == c for i in ids), new=sum(category(en[i]) == c for i in ids))
                for c in cats}
    transitions = {}
    for i in ids:
        key = f'{category(eb[i])} -> {category(en[i])}'
        transitions[key] = transitions.get(key, 0) + 1
    result = dict(base=args.name_base, new=args.name_new, episodes=len(ids), metrics=table,
                  paired_success=dict(both=both, only_base=only_base, only_new=only_new, neither=neither,
                                      mcnemar_exact_p=mcnemar_exact(only_base, only_new)),
                  bootstrap_95ci=ci, failure_taxonomy=taxonomy,
                  transitions=dict(sorted(transitions.items(), key=lambda kv: -kv[1])),
                  checkpoints=dict(base=cb['checkpoint'], new=cn['checkpoint'],
                                   base_weights=cb['weights_sha256'], new_weights=cn['weights_sha256']))
    if source_diff:
        result['source_diff'] = source_diff
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('metrics', 'paired_success', 'bootstrap_95ci', 'failure_taxonomy')},
                     indent=1))


if __name__ == '__main__':
    main()
