"""Aggregation rejects cross-host provenance drift and shifted recovery exports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from s8n_dagger.common import atomic_json, training_row
from s8n_dagger.merge import aggregate_shards, build_mixture


WEIGHTS = 'c89de946f3cb05aa91a5cc498fd63991c901852ba0c05e6546cbaddf23446a2b'


def contract(dataset, ids):
    return dict(protocol='s8n_janus_recovery_v2', dataset=dataset, split='train',
                dataset_sha256=('a' if dataset == 'r2r' else 'b') * 64,
                gt_sha256=('c' if dataset == 'r2r' else 'd') * 64,
                episode_ids=ids, max_steps=675, seed=42, beta=0,
                waypoint_radius=1.8, final_radius=.25, recovery_ratio=.8,
                recovery_errors=12, save_recovery_efficiency=.93,
                save_efficiency=.85, save_distance=.5, expert_only=False,
                keep_rejected_rgb=False, weights_sha256=WEIGHTS,
                navigation_sha256='e' * 64, base_assets={'tokenizer.json': 'f' * 64},
                model_source={'stream/session.py': '1' * 64},
                collector_source={'collect.py': '2' * 64}, habitat_version='0.2.4',
                habitat_config={'habitat': {'dataset': {'data_path': '/host/train.gz', 'scenes_dir': '/host/scenes'},
                                             'simulator': {'forward_step_size': .25, 'turn_angle': 15}}})


def shard(tmp_path, name, dataset='r2r', ids=('1',), accepted=True):
    root = tmp_path / name
    (root / 'episodes').mkdir(parents=True)
    (root / 'pending').mkdir()
    cfg = contract(dataset, list(ids))
    atomic_json(root / 'contract.json', cfg)
    exports = []
    for eid in ids:
        directory = root / 'episodes' / eid
        (directory / 'rgb').mkdir(parents=True)
        before = dict(position=[0., 0., 0.], rotation_xyzw=[0., 0., 0., 1.])
        steps = []
        for index, (action, tokens) in enumerate([(2, [15834, 18440, 248046]), (0, [48964, 248046])]):
            pixels = np.full((480, 640, 3), index + 5, np.uint8)
            Image.fromarray(pixels).save(directory / f'rgb/{index:04d}.png')
            after = dict(position=[5. if accepted else 4. / .99, 0., 0.], rotation_xyzw=[0., 0., 0., 1.])
            steps.append(dict(step_id=index, executed_action=action, oracle_action=action,
                              proposed_action=None, action_source='recovery' if action else 'oracle_stop',
                              proposed_response_token_ids=None, committed_response_token_ids=tokens,
                              committed_separator_ids=[198], rgb_path=f'rgb/{index:04d}.png',
                              rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                              pose_before=before, pose_after=after,
                              distance_before=4. if index == 0 else .3,
                              distance_after=.3 if index == 0 else .2))
            before = after
        record = dict(schema_version=1, episode_uid=f'{dataset}:train:s:{eid}:s8n_recovery2',
                      episode_id=eid, dataset=dataset, dataset_version=cfg['dataset_sha256'],
                      split='train', scene_id='mp3d/s/s.glb', instruction_id=eid,
                      instruction=f'Go to point {eid}.', language='en', accepted=accepted,
                      failed=False, fatal=False, termination='oracle_stop', distance_to_goal=.2,
                      initial_distance=4., path_length=5. if accepted else 4. / .99, efficiency=.8 if accepted else .99,
                      model_success=not accepted, rgb_retained=accepted, steps=steps)
        atomic_json(directory / 'record.json', record)
        if accepted:
            row = training_row(record, directory)
            exports.append(row)
            atomic_json(directory / 'training_episode.json', row)
    (root / 'training.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in exports))
    atomic_json(root / 'status.json', dict(complete=True, attempted=len(ids), total=len(ids),
                                         accepted=len(exports), failures=0, total_steps=2 * len(ids),
                                         accepted_steps=2 * len(exports)))
    atomic_json(root / 'validation.json', dict(passed=True, complete=True, episodes=len(ids),
                                             expected=len(ids), accepted=len(exports), steps=2 * len(ids)))
    return root


def edit_json(path, update):
    value = json.loads(path.read_text())
    update(value)
    atomic_json(path, value)


def test_aggregate_both_datasets_ignores_only_host_paths_and_keeps_rejected_attempts(tmp_path):
    local = shard(tmp_path, 'local', ids=('1', '2'))
    remote = shard(tmp_path, 'remote', dataset='rxr15', ids=('1',))
    rejected = shard(tmp_path, 'rejected', ids=('3',), accepted=False)
    edit_json(remote / 'contract.json', lambda value: value['habitat_config']['habitat']['dataset'].update(
        data_path='/different/train.gz', scenes_dir='/different/scenes'))
    (local / 'pending' / '1.interrupted-123').mkdir()
    rows, report = aggregate_shards([remote, rejected, local])
    assert [(row['dataset'], row['episode_id']) for row in rows] == [('r2rce', '1'), ('r2rce', '2'), ('rxrce', '1')]
    assert report['attempted'] == 4 and report['accepted'] == 3
    assert report['datasets']['r2r']['attempted'] == 3
    assert report['datasets']['rxr15']['attempted'] == 1
    assert report['archived_pending'] == 1


@pytest.mark.parametrize('field,value', [
    ('protocol', 's8n_janus_recovery_v1'), ('weights_sha256', '0' * 64),
    ('navigation_sha256', '0' * 64), ('model_source', {'stream/session.py': '0' * 64}),
    ('collector_source', {'collect.py': '0' * 64}), ('base_assets', {'tokenizer.json': '0' * 64}),
    ('recovery_errors', 13), ('gt_sha256', '0' * 64), ('dataset_sha256', '0' * 64),
])
def test_aggregate_rejects_protocol_and_content_provenance_drift(tmp_path, field, value):
    first = shard(tmp_path, 'a')
    second = shard(tmp_path, 'b', ids=('2',))
    edit_json(second / 'contract.json', lambda row: row.update({field: value}))
    with pytest.raises(ValueError):
        aggregate_shards([first, second])


def test_aggregate_rejects_numerical_config_change(tmp_path):
    first = shard(tmp_path, 'a')
    second = shard(tmp_path, 'b', ids=('2',))
    edit_json(second / 'contract.json', lambda row: row['habitat_config']['habitat']['simulator'].update(turn_angle=30))
    with pytest.raises(ValueError, match='provenance|config'):
        aggregate_shards([first, second])


def test_aggregate_rejects_duplicate_dataset_episode_even_when_not_exported(tmp_path):
    first = shard(tmp_path, 'a')
    second = shard(tmp_path, 'b', accepted=False)
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate_shards([first, second])


@pytest.mark.parametrize('corruption', ['incomplete', 'failed', 'pending', 'raw_rgb', 'export_rgb', 'manifest_duplicate', 'token_shift'])
def test_aggregate_rejects_incomplete_or_corrupt_shards(tmp_path, corruption):
    root = shard(tmp_path, 'a')
    if corruption == 'incomplete':
        edit_json(root / 'contract.json', lambda row: row['episode_ids'].append('2'))
    elif corruption == 'failed':
        edit_json(root / 'episodes/1/record.json', lambda row: row.update(failed=True))
    elif corruption == 'pending':
        (root / 'pending/2').mkdir()
    elif corruption == 'raw_rgb':
        Image.fromarray(np.zeros((480, 640, 3), np.uint8)).save(root / 'episodes/1/rgb/0000.png')
    elif corruption == 'export_rgb':
        edit_json(root / 'episodes/1/training_episode.json', lambda row: row['steps'][0].update(rgb_path=row['steps'][1]['rgb_path']))
    elif corruption == 'manifest_duplicate':
        path = root / 'training.jsonl'
        path.write_text(path.read_text() * 2)
    else:
        edit_json(root / 'episodes/1/record.json', lambda row: row['steps'][0].update(committed_response_token_ids=[48964, 248046]))
    with pytest.raises(ValueError):
        aggregate_shards([root])


def test_explicit_path_map_relocates_remote_exports_without_changing_sources(tmp_path):
    root = shard(tmp_path, 'copied_remote')
    training_file = root / 'episodes/1/training_episode.json'
    value = json.loads(training_file.read_text())
    original_prefix = '/remote/campaign/shard'
    for step in value['steps']:
        step['rgb_path'] = step['rgb_path'].replace(str(root), original_prefix)
    atomic_json(training_file, value)
    (root / 'training.jsonl').write_text(json.dumps(value) + '\n')
    before = training_file.read_bytes()
    with pytest.raises(ValueError):
        aggregate_shards([root])
    rows, _ = aggregate_shards([root], path_maps=[(original_prefix, str(root))])
    assert rows[0]['steps'][0]['rgb_path'] == str(root / 'episodes/1/rgb/0000.png')
    assert training_file.read_bytes() == before


def oracle_rows(recovery, count=4):
    rows = []
    template = recovery[0]
    for dataset in ('r2rce', 'rxrce'):
        for eid in range(1, count + 1):
            row = deepcopy(template)
            row.update(dataset=dataset, episode_id=str(eid), instruction_id=str(eid),
                       instruction=f'Go to point {eid}.', episode_uid=f'oracle:{dataset}:{eid}',
                       trajectory_variant='oracle', supervision='oracle_actions',
                       dataset_version=('a' if dataset == 'r2rce' else 'b') * 64)
            rows.append(row)
    return rows


def test_mixture_is_deterministic_balanced_and_preserves_generated_recovery_uid(tmp_path):
    r2r = shard(tmp_path, 'r2r', ids=('1', '2', '3'))
    rxr = shard(tmp_path, 'rxr', dataset='rxr15', ids=('1', '2', '3'))
    recovery, _ = aggregate_shards([r2r, rxr])
    oracle = oracle_rows(recovery)
    first, report = build_mixture(oracle, recovery, recovery_fraction=.5, seed=429)
    reordered, _ = build_mixture(oracle, list(reversed(recovery)), recovery_fraction=.5, seed=429)
    assert first == reordered and len(first) == 8
    for dataset in ('r2rce', 'rxrce'):
        subset = [row for row in first if row['dataset'] == dataset]
        assert len(subset) == 4
        replacements = [row for row in subset if row.get('supervision') == 'executed_recovery_actions']
        assert len(replacements) == 2
        assert all(row['episode_uid'].endswith('s8n_recovery2') for row in replacements)
    assert report['replaced'] == 4 and report['recovery_fraction_requested'] == .5
    assert [row['episode_id'] for row in first] == [row['episode_id'] for row in oracle]


@pytest.mark.parametrize('field,value', [('instruction', 'Wrong instruction'), ('instruction_id', '99'), ('scene_id', 'mp3d/other/other.glb'), ('episode_id', 'unknown'), ('split', 'val_unseen')])
def test_mixture_rejects_original_source_identity_mismatch(tmp_path, field, value):
    recovery, _ = aggregate_shards([shard(tmp_path, 'a')])
    oracle = oracle_rows(recovery)
    recovery[0][field] = value
    with pytest.raises(ValueError):
        build_mixture(oracle, recovery)


def test_mixture_rejects_duplicates_bad_fraction_and_missing_original_images(tmp_path):
    recovery, _ = aggregate_shards([shard(tmp_path, 'a')])
    oracle = oracle_rows(recovery)
    for fraction in (-.1, 1.1, float('nan')):
        with pytest.raises(ValueError):
            build_mixture(oracle, recovery, recovery_fraction=fraction)
    with pytest.raises(ValueError, match='Duplicate'):
        build_mixture(oracle + oracle[:1], recovery)
    oracle[0]['steps'][0]['rgb_path'] = '/missing/image.png'
    with pytest.raises(ValueError):
        build_mixture(oracle, recovery)


@pytest.mark.parametrize('field,value', [('path_length', 100.), ('efficiency', .7), ('initial_distance', 7.), ('distance_to_goal', .1)])
def test_raw_metrics_must_match_poses_and_distance_history(tmp_path, field, value):
    root = shard(tmp_path, 'a')
    edit_json(root / 'episodes/1/record.json', lambda row: row.update({field: value}))
    with pytest.raises(ValueError):
        aggregate_shards([root])


def test_cli_writes_only_validated_candidates_and_refuses_overwrite(tmp_path, capsys):
    from s8n_dagger.merge import main

    first = shard(tmp_path, 'a')
    second = shard(tmp_path, 'b', dataset='rxr15')
    recovery, _ = aggregate_shards([first, second])
    originals = oracle_rows(recovery, count=2)
    manifest = tmp_path / 'oracle.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in originals))
    out = tmp_path / 'merged'
    argv = ['--shards', str(first), str(second), '--expected-count', 'r2r=1',
            '--expected-count', 'rxr15=1', '--oracle-manifest', str(manifest), '--out', str(out)]
    main(argv)
    assert json.loads(capsys.readouterr().out)['status'] == 'validated'
    assert len((out / 'recovery.jsonl').read_text().splitlines()) == 2
    assert len((out / 'mixed_candidate.jsonl').read_text().splitlines()) == 4
    assert json.loads((out / 'mixture_report.json').read_text())['replaced'] == 2
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_cli_missing_dataset_or_failed_mixture_creates_no_output(tmp_path):
    from s8n_dagger.merge import main

    root = shard(tmp_path, 'a')
    out = tmp_path / 'merged'
    with pytest.raises(SystemExit):
        main(['--shards', str(root), '--out', str(out)])
    assert not out.exists()
    manifest = tmp_path / 'bad_oracle.jsonl'
    manifest.write_text('{}\n')
    with pytest.raises(SystemExit):
        main(['--shards', str(root), '--require-datasets', 'r2r', '--out', str(out),
              '--oracle-manifest', str(manifest)])
    assert not out.exists()


def test_numerical_simulator_contract_must_match_across_datasets(tmp_path):
    first = shard(tmp_path, 'a')
    second = shard(tmp_path, 'b', dataset='rxr15')
    edit_json(second / 'contract.json', lambda row: row['habitat_config']['habitat']['simulator'].update(turn_angle=30))
    with pytest.raises(ValueError, match='provenance|config'):
        aggregate_shards([first, second])


def test_portable_provenance_rejects_missing_content_hashes_even_if_all_shards_agree(tmp_path):
    root = shard(tmp_path, 'a')
    edit_json(root / 'contract.json', lambda row: row.update(model_source={'stream/session.py': 'unknown'}))
    with pytest.raises(ValueError):
        aggregate_shards([root])


def test_mixture_rejects_duplicate_original_uid_even_if_replacement_would_hide_it(tmp_path):
    recovery, _ = aggregate_shards([shard(tmp_path, 'a')])
    originals = oracle_rows(recovery)
    originals[1]['episode_uid'] = originals[0]['episode_uid']
    with pytest.raises(ValueError, match='Duplicate'):
        build_mixture(originals, recovery)


def test_mixture_rejects_different_source_versions_without_explicit_pair(tmp_path):
    recovery, _ = aggregate_shards([shard(tmp_path, 'a')])
    originals = oracle_rows(recovery)
    for row in originals:
        if row['dataset'] == 'r2rce':
            row['dataset_version'] = 'original-r2r-v1'
    with pytest.raises(ValueError, match='version'):
        build_mixture(originals, recovery)


def test_explicit_source_versions_are_checked_and_recorded(tmp_path):
    recovery, _ = aggregate_shards([shard(tmp_path, 'a')])
    originals = oracle_rows(recovery)
    for row in originals:
        if row['dataset'] == 'r2rce':
            row['dataset_version'] = 'original-r2r-v1'
    pair = {'r2r': {'oracle': 'original-r2r-v1', 'recovery': 'a' * 64}}
    mixed, report = build_mixture(originals, recovery, source_versions=pair)
    assert report['source_versions'] == pair
    assert any(row['episode_uid'].endswith('s8n_recovery2') for row in mixed)
    for side in ('oracle', 'recovery'):
        wrong = deepcopy(pair)
        wrong['r2r'][side] = 'wrong-version'
        with pytest.raises(ValueError, match='version'):
            build_mixture(originals, recovery, source_versions=wrong)
    originals[1]['dataset_version'] = 'wrong-nonreplaced-version'
    with pytest.raises(ValueError, match='version'):
        build_mixture(originals, recovery, source_versions=pair)


def test_cli_audits_explicit_source_version_file(tmp_path, capsys):
    from s8n_dagger.merge import main

    root = shard(tmp_path, 'a')
    recovery, _ = aggregate_shards([root])
    originals = oracle_rows(recovery)
    for row in originals:
        if row['dataset'] == 'r2rce':
            row['dataset_version'] = 'named-r2r-version'
    manifest = tmp_path / 'oracle.jsonl'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in originals))
    mapping = {'r2r': {'oracle': 'named-r2r-version', 'recovery': 'a' * 64}}
    versions = tmp_path / 'versions.json'
    atomic_json(versions, mapping)
    out = tmp_path / 'merged'
    main(['--shards', str(root), '--require-datasets', 'r2r', '--out', str(out),
          '--oracle-manifest', str(manifest), '--source-versions', str(versions)])
    assert json.loads((out / 'mixture_report.json').read_text())['source_versions'] == mapping
