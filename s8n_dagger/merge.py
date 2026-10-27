"""Validate portable S8N v2 shards and write recovery/optional mixture candidates.

Example (after both hosts' outputs and original RGB are accessible locally)::

    python -m s8n_dagger.merge --shards /local/full/* /mounted/remote/full/* \
      --path-map /remote/campaign=/mounted/remote \
      --expected-count r2r=10819 --expected-count rxr15=19996 --out /new/merged

Add --oracle-manifest /data/oracle.jsonl --recovery-fraction .5 --seed 429
for a candidate that replaces matching original rows, keeping dataset counts.
The fraction is a requested cap, not an experimentally established optimum.
Dataset versions must match unless --source-versions versions.json explicitly
maps each relevant dataset to its exact oracle/recovery version pair.
Generated recovery UIDs are preserved. Inputs are never rewritten; no copying,
remote access, or training is performed. Only supplied shards are certified.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

import numpy as np
from PIL import Image

from s8n_dagger.common import ACTION_NAMES, accepted_trajectory, atomic_json, sha256, training_row


PROTOCOL = 's8n_janus_recovery_v2'
WEIGHTS = 'c89de946f3cb05aa91a5cc498fd63991c901852ba0c05e6546cbaddf23446a2b'
DATASETS = {'r2r': 'r2r', 'r2rce': 'r2r', 'rxr15': 'rxr15', 'rxrce': 'rxr15'}
CANONICAL = {1: [28164, 61846, 248046], 2: [15834, 18440, 248046],
             3: [15834, 20549, 248046], 0: [48964, 248046]}
REQUIRED = {'protocol', 'dataset', 'split', 'dataset_sha256', 'gt_sha256', 'episode_ids',
            'weights_sha256', 'navigation_sha256', 'model_source', 'collector_source',
            'base_assets', 'habitat_version', 'habitat_config', 'max_steps', 'seed',
            'beta', 'waypoint_radius', 'final_radius', 'recovery_ratio', 'recovery_errors',
            'save_recovery_efficiency', 'save_efficiency', 'save_distance', 'expert_only',
            'keep_rejected_rgb'}
SITE_CONFIG_PATHS = {('habitat', 'dataset', 'data_path'), ('habitat', 'dataset', 'scenes_dir'),
                     ('habitat', 'task', 'measurements', 'ndtw', 'gt_path')}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bad_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def _json(path):
    return json.loads(Path(path).read_text(), parse_constant=_bad_constant)


def _jsonl(path):
    return [json.loads(line, parse_constant=_bad_constant)
            for line in Path(path).read_text().splitlines() if line.strip()]


def _maps(path_maps):
    result = []
    for old, new in path_maps:
        _require(Path(old).is_absolute() and Path(new).is_absolute(), 'Path maps require absolute OLD=NEW paths')
        result.append((Path(old), Path(new).resolve()))
    _require(len({str(old) for old, _ in result}) == len(result), 'Duplicate path-map prefix')
    return sorted(result, key=lambda pair: len(pair[0].parts), reverse=True)


def _relocate(value, mappings):
    path = Path(value)
    if path.is_absolute():
        for old, new in mappings:
            if path.is_relative_to(old):
                return str(new / path.relative_to(old))
    return value


def _relocate_row(row, mappings, relative_root=None):
    row = deepcopy(row)
    row['scene_id'] = _relocate(row['scene_id'], mappings)
    for step in row['steps']:
        value = _relocate(step['rgb_path'], mappings)
        if relative_root is not None and not Path(value).is_absolute():
            value = str((Path(relative_root) / value).resolve())
        step['rgb_path'] = value
    return row


def _config(value, path=()):
    if path in SITE_CONFIG_PATHS:
        return '<site-path>'
    if isinstance(value, dict):
        return {key: _config(child, path + (key,)) for key, child in value.items()}
    if isinstance(value, list):
        return [_config(child, path + (str(index),)) for index, child in enumerate(value)]
    return value


def portable_provenance(contract):
    """Retain all unknown policy fields; ignore only explicit site path fields."""
    _require(REQUIRED <= contract.keys(), f'Missing contract provenance: {sorted(REQUIRED - contract.keys())}')
    _require(contract['protocol'] == PROTOCOL and contract['split'] == 'train', 'Expected protocol v2 training collection')
    _require(contract['dataset'] in ('r2r', 'rxr15'), 'Unsupported collection dataset')
    _require(contract['weights_sha256'] == WEIGHTS, 'Expected original S8N weights')
    _require(contract['expert_only'] is False, 'Expert-only diagnostics are not recovery collection')
    for key in ('model_source', 'collector_source', 'base_assets'):
        _require(isinstance(contract[key], dict) and contract[key], f'Missing {key} content hashes')
        for name, content_hash in contract[key].items():
            _require(isinstance(name, str) and name and isinstance(content_hash, str)
                     and len(content_hash) == 64 and all(c in '0123456789abcdef' for c in content_hash),
                     f'Invalid {key} content hash: {name}')
    for key in ('dataset_sha256', 'gt_sha256', 'weights_sha256', 'navigation_sha256'):
        value = contract[key]
        _require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), f'Invalid {key}')
    global_fields = {key: deepcopy(value) for key, value in contract.items()
                     if key not in {'dataset', 'dataset_sha256', 'gt_sha256', 'episode_ids', 'habitat_config'}}
    habitat = contract['habitat_config']['habitat']
    global_fields['shared_habitat_config'] = {key: _config(habitat.get(key), ('habitat', key))
                                              for key in ('simulator', 'environment', 'seed')}
    per_dataset = {key: deepcopy(contract[key]) for key in ('dataset_sha256', 'gt_sha256')}
    per_dataset['habitat_config'] = _config(contract['habitat_config'])
    return global_fields, per_dataset


def _image(path, expected_hash=None):
    try:
        with Image.open(path) as image:
            _require(image.mode == 'RGB' and image.size == (640, 480), f'Invalid RGB contract: {path}')
            pixels = np.asarray(image)
            _require(pixels.dtype == np.uint8, f'Invalid RGB dtype: {path}')
            if expected_hash is not None:
                _require(hashlib.sha256(pixels.tobytes()).hexdigest() == expected_hash, f'Raw RGB hash mismatch: {path}')
    except (OSError, ValueError) as exc:
        raise ValueError(f'Invalid image {path}: {exc}') from exc


def _training(row):
    from qwen_vl.contracts import validate_episode
    _require(isinstance(row, dict) and row.get('schema_version') == 1, 'Expected schema-v1 training row')
    _require(row.get('split') == 'train' and row.get('dataset') in DATASETS, 'Only supported training datasets may be merged')
    try:
        validate_episode(row)
        for step in row['steps']:
            _image(step['rgb_path'])
    except (AssertionError, KeyError, OSError, TypeError) as exc:
        raise ValueError(f'Invalid training export: {exc}') from exc


def _raw_frame(directory, step):
    path = Path(step['rgb_path'])
    _require(not path.is_absolute() and '..' not in path.parts, 'Raw frame path must remain inside its episode')
    resolved = (directory / path).resolve()
    _require(resolved.is_relative_to(directory.resolve()), 'Raw frame escapes its episode')
    _image(resolved, step['rgb_sha256'])


def _validate_record(record, directory, contract):
    _require(record.get('schema_version') == 1 and record.get('split') == 'train', 'Invalid raw record schema/split')
    _require(record.get('dataset') == contract['dataset'] and record.get('dataset_version') == contract['dataset_sha256'], 'Raw record dataset provenance mismatch')
    _require(record.get('failed') is False and not record.get('fatal'), 'Failed attempt cannot enter a merged collection')
    _require(record.get('termination') in ('oracle_stop', 'max_steps', 'environment_limit'), 'Incomplete raw attempt')
    _require(isinstance(record.get('accepted'), bool), 'Missing acceptance decision')
    for key in ('distance_to_goal', 'efficiency', 'initial_distance', 'path_length'):
        _require(isinstance(record.get(key), (int, float)) and math.isfinite(record[key]) and record[key] >= 0, f'Invalid raw {key}')
    expected = accepted_trajectory(record['termination'], record['distance_to_goal'], record['efficiency'], record['model_success'])
    _require(record['accepted'] == expected, 'Acceptance differs from recorded collection policy')
    steps = record['steps']
    _require(0 < len(steps) <= contract['max_steps'], 'Incomplete or over-cap raw trajectory')
    _require(not record['accepted'] or record.get('rgb_retained') is True, 'Accepted trajectory missing RGB')
    travelled = 0.0
    for index, step in enumerate(steps):
        for key in ('distance_before', 'distance_after'):
            _require(isinstance(step.get(key), (int, float)) and math.isfinite(step[key]) and step[key] >= 0, f'Invalid raw {key}')
        expected_before = record['initial_distance'] if index == 0 else steps[index - 1]['distance_after']
        _require(step['distance_before'] == expected_before, 'Goal-distance history alignment mismatch')
        _require(type(step['step_id']) is int and step['step_id'] == index, 'Raw step order mismatch')
        for key in ('executed_action', 'oracle_action'):
            _require(type(step[key]) is int and step[key] in ACTION_NAMES, f'Invalid raw {key}')
        action = step['executed_action']
        tokens = step['committed_response_token_ids']
        _require(tokens and all(type(token) is int for token in tokens) and tokens[-1] == 248046, 'Invalid committed response/EOS')
        _require(step['committed_separator_ids'] == [198], 'Invalid committed separator')
        source = step['action_source']
        _require(source in ('model', 'recovery', 'stop_override', 'oracle_stop'), 'Invalid action source')
        if source == 'model':
            _require(step['proposed_action'] == action and step['proposed_response_token_ids'] == tokens, 'Learner response/executed history mismatch')
        else:
            _require(tokens == CANONICAL[action] and step['oracle_action'] == action, 'Canonical expert response differs from executed action')
        if action == 0:
            _require(index == len(steps) - 1 and record['termination'] == 'oracle_stop' and source == 'oracle_stop', 'Fabricated or premature STOP')
        for key in ('pose_before', 'pose_after'):
            pose = step[key]
            _require(len(pose['position']) == 3 and len(pose['rotation_xyzw']) == 4,
                     'Invalid recorded pose shape')
            _require(all(math.isfinite(value) for value in pose['position'] + pose['rotation_xyzw']), 'Nonfinite recorded pose')
        travelled += float(np.linalg.norm(np.subtract(step['pose_after']['position'], step['pose_before']['position'])))
        if index:
            _require(step['pose_before'] == steps[index - 1]['pose_after'], 'Pre-action pose alignment mismatch')
        if record['rgb_retained']:
            _raw_frame(directory, step)
    _require(math.isclose(travelled, record['path_length'], rel_tol=1e-10, abs_tol=1e-8), 'Recorded path length differs from action poses')
    expected_efficiency = record['initial_distance'] / max(record['initial_distance'], travelled) if record['initial_distance'] > 0 else 1.0
    _require(math.isclose(expected_efficiency, record['efficiency'], rel_tol=1e-10, abs_tol=1e-8), 'Recorded efficiency differs from action poses')
    _require(steps[-1]['distance_after'] == record['distance_to_goal'], 'Final distance differs from action history')
    _require((steps[-1]['executed_action'] == 0) == (record['termination'] == 'oracle_stop'), 'Termination/action mismatch')


def _validate_shard(root, contract, mappings):
    ids = contract['episode_ids']
    _require(ids and all(isinstance(eid, str) and eid for eid in ids) and len(ids) == len(set(ids)), 'Invalid/duplicate shard episode IDs')
    directories = sorted((root / 'episodes').iterdir())
    _require({entry.name for entry in directories} == set(ids), 'Incomplete or unexpected shard episode coverage')
    pending = list((root / 'pending').iterdir()) if (root / 'pending').exists() else []
    archived = [entry for entry in pending if '.interrupted-' in entry.name]
    _require(len(archived) == len(pending), 'Incomplete pending episode staging')
    rows, attempts, uid_seen, count_steps = [], [], set(), 0
    for directory in directories:
        _require(directory.is_dir(), 'Invalid episode directory')
        record = _json(directory / 'record.json')
        _require(record['episode_id'] == directory.name and record['episode_id'] in ids, 'Episode ID/directory mismatch')
        _require(record['episode_uid'] not in uid_seen and record['episode_uid'], 'Duplicate raw episode UID')
        uid_seen.add(record['episode_uid'])
        _validate_record(record, directory, contract)
        attempts.append((contract['dataset'], record['episode_id']))
        count_steps += len(record['steps'])
        export_path = directory / 'training_episode.json'
        if record['accepted']:
            export = _relocate_row(_json(export_path), mappings)
            local_record = deepcopy(record)
            local_record['scene_id'] = _relocate(local_record['scene_id'], mappings)
            expected = training_row(local_record, directory)
            _require(export == expected, 'Training export differs from captured trajectory or relocated paths')
            _training(export)
            rows.append(export)
        else:
            _require(not export_path.exists(), 'Rejected attempt has a training export')
    exported = [_relocate_row(row, mappings) for row in _jsonl(root / 'training.jsonl')]
    _require(len({row['episode_uid'] for row in exported}) == len(exported), 'Duplicate shard manifest export')
    _require({row['episode_uid']: row for row in exported} == {row['episode_uid']: row for row in rows}, 'Shard manifest differs from validated exports')
    validation = _json(root / 'validation.json')
    expected_validation = dict(passed=True, complete=True, episodes=len(ids), expected=len(ids), accepted=len(rows), steps=count_steps)
    _require(all(validation.get(key) == value for key, value in expected_validation.items()), 'Incomplete or mismatched saved validation')
    status = _json(root / 'status.json')
    expected_status = dict(complete=True, attempted=len(ids), total=len(ids), accepted=len(rows), failures=0,
                           total_steps=count_steps, accepted_steps=sum(len(row['steps']) for row in rows))
    _require(all(status.get(key) == value for key, value in expected_status.items()), 'Incomplete or failed shard status')
    return rows, attempts, len(archived)


def aggregate_shards(shards, *, path_maps=(), expected_counts=None):
    """Return validated relocated rows and audit; completeness covers supplied shards."""
    roots = [Path(root).resolve() for root in shards]
    _require(roots and len(roots) == len(set(roots)), 'Nonempty unique shard directories required')
    mappings = _maps(path_maps)
    baseline, datasets, seen, uids, rows, audit = None, {}, set(), set(), [], []
    archived_total = 0
    for root in sorted(roots):
        try:
            contract = _json(root / 'contract.json')
            global_fields, per_dataset = portable_provenance(contract)
            if baseline is None:
                baseline = global_fields
            _require(global_fields == baseline, f'Portable policy provenance mismatch: {root}')
            dataset = contract['dataset']
            if dataset in datasets:
                _require(datasets[dataset]['provenance'] == per_dataset, f'Dataset/config provenance mismatch: {root}')
            else:
                datasets[dataset] = dict(provenance=per_dataset, attempted=0, accepted=0, accepted_steps=0)
            shard_rows, attempts, archived = _validate_shard(root, contract, mappings)
            for key in attempts:
                _require(key not in seen, f'Duplicate dataset+episode attempt: {key}')
                seen.add(key)
            for row in shard_rows:
                _require(row['episode_uid'] not in uids, 'Duplicate global training episode UID')
                uids.add(row['episode_uid'])
            rows.extend(shard_rows)
            datasets[dataset]['attempted'] += len(attempts)
            datasets[dataset]['accepted'] += len(shard_rows)
            datasets[dataset]['accepted_steps'] += sum(len(row['steps']) for row in shard_rows)
            archived_total += archived
            audit.append(dict(root=str(root), contract_sha256=sha256(root / 'contract.json'),
                              validation_sha256=sha256(root / 'validation.json'), attempted=len(attempts), accepted=len(shard_rows)))
        except (AssertionError, KeyError, OSError, TypeError, IndexError) as exc:
            raise ValueError(f'Invalid shard {root}: {exc}') from exc
    for dataset, count in (expected_counts or {}).items():
        _require(dataset in datasets and datasets[dataset]['attempted'] == count, f'Expected full {dataset} count {count}')
    rows.sort(key=lambda row: (DATASETS[row['dataset']], row['episode_id'], row['instruction_id']))
    report = dict(protocol=PROTOCOL, complete_supplied_shards=True, shard_count=len(roots),
                  attempted=len(seen), accepted=len(rows), datasets=datasets, provenance=baseline,
                  shards=audit, archived_pending=archived_total,
                  path_maps=[dict(old=str(old), new=str(new)) for old, new in mappings])
    return rows, report


def _key(row):
    _require(row.get('dataset') in DATASETS and row.get('episode_id') is not None, 'Invalid source identity')
    return DATASETS[row['dataset']], str(row['episode_id'])


def _scene(value):
    parts = PurePosixPath(value).parts
    if 'mp3d' in parts:
        return '/'.join(parts[parts.index('mp3d'):])
    return str(PurePosixPath(value))


def build_mixture(oracle_rows, recovery_rows, *, recovery_fraction=.5, seed=429, source_versions=None):
    """Deterministically replace at most floor(fraction * N) rows per dataset."""
    _require(type(recovery_fraction) in (int, float) and math.isfinite(recovery_fraction) and 0 <= recovery_fraction <= 1, 'Recovery fraction must be finite and in [0,1]')
    _require(type(seed) is int, 'Seed must be an integer')
    oracle_rows, recovery_rows = list(oracle_rows), list(recovery_rows)
    source_versions = {} if source_versions is None else deepcopy(source_versions)
    _require(isinstance(source_versions, dict), 'Source versions must be a dataset mapping')
    for dataset, pair in source_versions.items():
        _require(dataset in ('r2r', 'rxr15') and isinstance(pair, dict)
                 and set(pair) == {'oracle', 'recovery'}
                 and all(isinstance(version, str) and version.strip() for version in pair.values()),
                 'Source versions require dataset->{oracle: version, recovery: version}')
    original, candidates, original_uids = {}, {}, set()
    for row in oracle_rows:
        _training(row)
        key = _key(row)
        if key[0] in source_versions:
            _require(row['dataset_version'] == source_versions[key[0]]['oracle'], f'Original source version mismatch: {key}')
        _require(row['episode_uid'] not in original_uids, 'Duplicate original episode UID')
        original_uids.add(row['episode_uid'])
        _require(key not in original, f'Duplicate original dataset+episode: {key}')
        original[key] = row
    _require(original, 'Original oracle manifest is empty')
    for row in recovery_rows:
        _training(row)
        key = _key(row)
        _require(key not in candidates, f'Duplicate recovery dataset+episode: {key}')
        _require(key in original, f'Recovery episode missing from original manifest: {key}')
        source = original[key]
        if key[0] in source_versions:
            _require(row['dataset_version'] == source_versions[key[0]]['recovery'], f'Recovery source version mismatch: {key}')
        else:
            _require(row['dataset_version'] == source['dataset_version'], f'Source dataset version mismatch: {key}; explicit source_versions pair required')
        _require(str(row['instruction_id']) == str(source['instruction_id'])
                 and row['instruction'].strip() == source['instruction'].strip()
                 and _scene(row['scene_id']) == _scene(source['scene_id']), f'Recovery source identity mismatch: {key}')
        _require(row.get('supervision') == 'executed_recovery_actions'
                 and row.get('trajectory_variant') == 's8n_janus_recovery_15deg_v2', 'Expected validated recovery-v2 row')
        candidates[key] = row
    counts = Counter(key[0] for key in original)
    selected, per_dataset = set(), {}
    for dataset in sorted(counts):
        eligible = [key for key in candidates if key[0] == dataset]
        eligible.sort(key=lambda key: hashlib.sha256(f'{seed}:{key[0]}:{key[1]}'.encode()).digest())
        cap = math.floor(recovery_fraction * counts[dataset])
        chosen = eligible[:cap]
        selected.update(chosen)
        per_dataset[dataset] = dict(total=counts[dataset], eligible_recovery=len(eligible), cap=cap,
                                    replaced=len(chosen), actual_recovery_fraction=len(chosen) / counts[dataset])
    mixed = [deepcopy(candidates[_key(row)] if _key(row) in selected else row) for row in oracle_rows]
    _require(len({row['episode_uid'] for row in mixed}) == len(mixed), 'Duplicate mixed episode UID')
    _require(Counter(_key(row)[0] for row in mixed) == counts, 'Mixture changed dataset balance')
    return mixed, dict(candidate_only=True, recovery_fraction_requested=recovery_fraction,
                       seed=seed, total=len(mixed), replaced=len(selected), datasets=per_dataset,
                       uid_policy='preserve_generated_recovery_uid', source_versions=source_versions,
                       selected=[dict(dataset=dataset, episode_id=eid) for dataset, eid in sorted(selected)])


def _pairs(values, label, convert=str):
    pairs = []
    for value in values:
        _require('=' in value, f'{label} requires NAME=VALUE')
        key, child = value.split('=', 1)
        _require(key and child, f'{label} requires nonempty NAME=VALUE')
        pairs.append((key, convert(child)))
    return pairs


def _write_jsonl(path, rows):
    with path.open('w') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--shards', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True, help='New output directory; existing outputs are never replaced')
    parser.add_argument('--path-map', action='append', default=[], metavar='OLD=NEW')
    parser.add_argument('--expected-count', action='append', default=[], metavar='DATASET=COUNT')
    parser.add_argument('--require-datasets', nargs='+', choices=('r2r', 'rxr15'), default=['r2r', 'rxr15'])
    parser.add_argument('--oracle-manifest', type=Path)
    parser.add_argument('--source-versions', type=Path, help='JSON file: dataset -> {oracle: exact_version, recovery: exact_version}')
    parser.add_argument('--oracle-root', type=Path, help='Root for relative original RGB paths; defaults to manifest directory')
    parser.add_argument('--recovery-fraction', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=429)
    args = parser.parse_args(argv)
    try:
        path_maps = _pairs(args.path_map, '--path-map')
        expected_pairs = _pairs(args.expected_count, '--expected-count', int)
        _require(len(dict(expected_pairs)) == len(expected_pairs), 'Duplicate expected dataset count')
        _require(all(dataset in ('r2r', 'rxr15') and count >= 0 for dataset, count in expected_pairs), 'Invalid expected dataset/count')
        _require(not args.out.exists(), 'Output directory already exists')
        rows, report = aggregate_shards(args.shards, path_maps=path_maps, expected_counts=dict(expected_pairs))
        _require(set(args.require_datasets) <= report['datasets'].keys(), 'Required dataset missing from supplied shards')
        mixture, mixture_report = None, None
        _require(args.source_versions is None or args.oracle_manifest is not None, '--source-versions requires --oracle-manifest')
        if args.oracle_manifest is not None:
            mappings = _maps(path_maps)
            originals = [_relocate_row(row, mappings, args.oracle_root or args.oracle_manifest.resolve().parent)
                         for row in _jsonl(args.oracle_manifest)]
            mixture, mixture_report = build_mixture(originals, rows, recovery_fraction=args.recovery_fraction, seed=args.seed,
                                                     source_versions=_json(args.source_versions) if args.source_versions else None)
            mixture_report['oracle_manifest_sha256'] = sha256(args.oracle_manifest)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f'.{args.out.name}.pending-', dir=args.out.parent))
        try:
            _write_jsonl(staging / 'recovery.jsonl', rows)
            atomic_json(staging / 'merge_report.json', report)
            if mixture is not None:
                _write_jsonl(staging / 'mixed_candidate.jsonl', mixture)
                atomic_json(staging / 'mixture_report.json', mixture_report)
            staging.rename(args.out)
        except BaseException:
            shutil.rmtree(staging)
            raise
        print(json.dumps(dict(status='validated', output=str(args.out.resolve()), attempted=report['attempted'],
                              accepted=report['accepted'], mixture_candidate=mixture is not None)), flush=True)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
