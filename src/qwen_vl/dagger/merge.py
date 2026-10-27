"""Merge DAgger shard journals into a validated manifest, optionally mixed into an oracle manifest.

    python -m qwen_vl.dagger.merge --run-dir RUN --out dagger.jsonl
    python -m qwen_vl.dagger.merge --run-dir RUN --out mix.jsonl --oracle-manifest joint.jsonl \
        --replace-dataset r2rce --dagger-fraction 0.5 --seed 429

The mixture keeps the oracle manifest's size (the S8N_E2 update count) and every row of the other
datasets (RxR); DAgger rows (all R2R) replace up to ``fraction`` of the ``--replace-dataset`` oracle
rows, so the dataset balance of the S8N_E2 control is unchanged and only R2R oracle trajectories
are exchanged for R2R recovery trajectories.  Draws are deterministic: rows are ranked by
SHA-256(seed | uid).  All shard contracts must agree and every shard must be complete unless
``--allow-partial``.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from qwen_vl.contracts import validate_episode

SHARED_CONTRACT_KEYS = ('collector', 'rules', 'tag', 'split', 'num_shards', 'checkpoint', 'weights_sha256',
                        'navigation_sha256', 'dataset_sha256', 'habitat_config_sha256',
                        'oracle_manifest_sha256', 'source_sha256', 'base_assets_sha256', 'simulator_version')


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f'{path}:{number}: corrupt JSON line') from exc
    return rows


def shard_index(path):
    match = re.fullmatch(r'(?:journal|contract)\.shard(\d{3})\.jsonl?', Path(path).name)
    if not match:
        raise ValueError(f'Unexpected shard file name {path}')
    return int(match[1])


def check_contracts(run_dir, allow_partial):
    contracts = {shard_index(p): json.loads(p.read_text()) for p in sorted(Path(run_dir).glob('contract.shard*.json'))}
    if not contracts:
        raise ValueError(f'No shard contracts under {run_dir}')
    reference = next(iter(contracts.values()))
    for index, contract in contracts.items():
        differing = [k for k in SHARED_CONTRACT_KEYS if contract.get(k) != reference.get(k)]
        if differing:
            raise ValueError(f'Shard {index} contract differs in {differing}')
        if contract['shard_index'] != index:
            raise ValueError(f'Contract file of shard {index} declares shard {contract["shard_index"]}')
    missing = sorted(set(range(reference['num_shards'])) - contracts.keys())
    if missing and not allow_partial:
        raise ValueError(f'Shards without a contract: {missing}')
    return contracts


def journal_records(run_dir, contracts, allow_partial):
    records, seen = [], {}
    for index, contract in sorted(contracts.items()):
        path = Path(run_dir) / f'journal.shard{index:03d}.jsonl'
        rows = read_jsonl(path) if path.exists() else []
        ids = [str(r['episode_id']) for r in rows]
        expected = set(map(str, contract['episode_ids']))
        if set(ids) - expected:
            raise ValueError(f'Shard {index} journaled episodes outside its contract')
        if set(ids) != expected and not allow_partial:
            raise ValueError(f'Shard {index} is incomplete: {len(set(ids))}/{len(expected)} episodes')
        for key, record in zip(ids, rows):
            if key in seen:
                raise ValueError(f'Episode {key} journaled by shard {seen[key]} and shard {index}')
            seen[key] = index
            records.append(record)
    return records


def rank(seed, rows):
    return sorted(rows, key=lambda r: hashlib.sha256(f'{seed}|{r["episode_uid"]}'.encode()).hexdigest())


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    return {str(q): values[min(len(values) - 1, int(q * len(values)))] for q in (0.0, 0.25, 0.5, 0.75, 1.0)}


def audit(records, rows):
    saved = [r for r in records if r.get('saved')]
    return dict(attempted=len(records), saved=len(saved),
                reasons=dict(Counter(r['reason'] for r in records)),
                takeover_episodes=sum(1 for r in records if r.get('takeovers', 0) > 0),
                saved_takeover_episodes=sum(1 for r in saved if r.get('takeovers', 0) > 0),
                saved_without_takeover=sum(1 for r in saved if r.get('takeovers', 0) == 0),
                action_counts=dict(Counter(s['action_name'] for row in rows for s in row['steps'])),
                step_sources=dict(Counter(s['dagger']['source'] for row in rows for s in row['steps'])),
                steps=quantiles([len(row['steps']) for row in rows]),
                saved_path_efficiency=quantiles([r['path_efficiency'] for r in saved]),
                total_steps=sum(len(row['steps']) for row in rows))


def mix(rows, oracle, replace, fraction, seed):
    slots = [r for r in oracle if r.get('dataset') == replace]
    if not slots:
        raise ValueError(f'The oracle manifest has no {replace!r} rows to replace')
    kept = [r for r in oracle if r.get('dataset') != replace]
    dagger = rank(seed, rows)[:round(len(slots) * fraction)]
    filled = rank(seed, slots)[:len(slots) - len(dagger)]
    mixture = kept + filled + dagger
    if len(mixture) != len(oracle) or len({r['episode_uid'] for r in mixture}) != len(mixture):
        raise ValueError('Mixture size or episode-UID uniqueness broken')
    versions = {r['dataset_version'] for r in slots}
    report = dict(total=len(mixture), dagger_rows=len(dagger), replaced_oracle_rows=len(slots) - len(filled),
                  datasets=dict(Counter(r['dataset'] for r in mixture)),
                  oracle_datasets=dict(Counter(r['dataset'] for r in oracle)), seed=seed,
                  replace_dataset=replace, fraction=fraction,
                  r2r_source_matches_oracle=all(r['dataset_version'] in versions for r in dagger))
    return mixture, report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--oracle-manifest')
    p.add_argument('--replace-dataset', default='r2rce')
    p.add_argument('--dagger-fraction', type=float, default=0.5,
                   help='Largest share of the replaced dataset rows taken by DAgger rows')
    p.add_argument('--seed', type=int, default=429)
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--skip-image-check', action='store_true')
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    contracts = check_contracts(args.run_dir, args.allow_partial)
    records = journal_records(args.run_dir, contracts, args.allow_partial)
    rows = [r['row'] for r in records if r.get('saved')]
    if not rows:
        raise ValueError('No saved DAgger episodes')
    for row in rows:
        validate_episode(row, check_images=not args.skip_image_check)
    report = dict(dagger=audit(records, rows), shards=len(contracts),
                  rules=next(iter(contracts.values()))['rules'], partial=args.allow_partial)
    if args.oracle_manifest:
        if not 0 < args.dagger_fraction <= 1:
            raise ValueError('--dagger-fraction must be in (0, 1]')
        oracle = read_jsonl(args.oracle_manifest)
        rows, report['mixture'] = mix(rows, oracle, args.replace_dataset, args.dagger_fraction, args.seed)
        report['mixture'].update(oracle_manifest=str(Path(args.oracle_manifest).resolve()),
                                 oracle_manifest_sha256=hashlib.sha256(
                                     Path(args.oracle_manifest).read_bytes()).hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    report['manifest'] = str(out.resolve())
    report['manifest_sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix('.audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
