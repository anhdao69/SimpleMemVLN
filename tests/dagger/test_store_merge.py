"""Frame folders, journals, manifest rows and the merge/mixture tool."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from qwen_vl.contracts import ACTIONS, validate_episode
from qwen_vl.dagger import merge
from qwen_vl.dagger.store import CollectionStore, manifest_row

STOP, FORWARD = ACTIONS.index('STOP'), ACTIONS.index('MOVE_FORWARD')
SOURCE = dict(scene_id='mp3d/abc/abc.glb', instruction_id='7', instruction='Walk to the sofa.',
              dataset_version='sha-r2r')


def frames(n, seed):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(n)]


def saved_record(store, eid, n=3, seed=0, tag='t'):
    actions = [FORWARD] * (n - 1) + [STOP]
    images = frames(n, seed)
    paths = store.save_frames(eid, images, actions)
    steps = [dict(source='model', expert='MOVE_FORWARD', executed=ACTIONS[a], distance_to_goal=1.0, waypoint=1)
             for a in actions]
    row = manifest_row(tag, 'train', SOURCE, eid, paths, actions, steps, dict(takeovers=1))
    return dict(episode_id=eid, saved=True, reason='takeover', takeovers=1, path_efficiency=0.8, row=row), images


class StoreTests(unittest.TestCase):
    def test_frames_are_lossless_and_rows_validate(self):
        with tempfile.TemporaryDirectory() as root:
            store = CollectionStore(root, 2)
            record, images = saved_record(store, '42')
            row = record['row']
            validate_episode(row)
            self.assertEqual(row['episode_uid'], 'r2rce_dagger:t:train:mp3d/abc/abc.glb:42:7')
            self.assertEqual([s['action_name'] for s in row['steps']], ['MOVE_FORWARD', 'MOVE_FORWARD', 'STOP'])
            for step, image in zip(row['steps'], images):
                self.assertTrue(Path(step['rgb_path']).name.startswith(f"step_{step['step_id']:04d}_"))
                with Image.open(step['rgb_path']) as stored:
                    self.assertEqual(stored.mode, 'RGB')
                    np.testing.assert_array_equal(np.asarray(stored), image)
            store.append(record)
            self.assertEqual(Path(root, 'journal.shard002.jsonl').read_text().count('\n'), 1)
            with self.assertRaises(FileExistsError):
                store.save_frames('42', images, [FORWARD, FORWARD, STOP])

    def test_save_frames_rejects_bad_trajectories(self):
        with tempfile.TemporaryDirectory() as root:
            store = CollectionStore(root, 0)
            with self.assertRaises(ValueError):
                store.save_frames('1', frames(2, 0), [FORWARD, FORWARD])           # no final STOP
            with self.assertRaises(ValueError):
                store.save_frames('2', frames(2, 0), [STOP, STOP])                 # early STOP
            with self.assertRaises(ValueError):
                store.save_frames('3', frames(2, 0), [STOP])                       # misaligned
            with self.assertRaises(ValueError):
                store.save_frames('4', [np.zeros((480, 640), np.uint8)], [STOP])   # not RGB

    def test_clean_removes_partial_and_unjournaled_folders_only(self):
        with tempfile.TemporaryDirectory() as root:
            store = CollectionStore(root, 0)
            saved_record(store, '1')
            saved_record(store, '2')
            (store.frames / '.tmp-3').mkdir()
            removed = store.clean(['1', '2', '3'], completed={'1'})
            self.assertTrue((store.frames / '1').exists())
            self.assertFalse((store.frames / '2').exists())
            self.assertFalse((store.frames / '.tmp-3').exists())
            self.assertEqual(len(removed), 2)


class MergeTests(unittest.TestCase):
    def run_merge(self, *argv):
        with mock.patch('sys.argv', ['merge', *argv]), mock.patch('builtins.print'):
            merge.main()

    def make_run(self, root, contracts=True):
        run = Path(root, 'run')
        stores = [CollectionStore(run, 0), CollectionStore(run, 1)]
        shard_ids = {0: ['10', '12', '13'], 1: ['11']}
        for i, eid in enumerate(['10', '11', '12']):
            record, _ = saved_record(stores[i % 2], eid, seed=i)
            stores[i % 2].append(record)
        stores[0].append(dict(episode_id='13', saved=False, reason='near_oracle', takeovers=0))
        if contracts:
            for index, ids in shard_ids.items():
                self.write_contract(run, index, ids)
        return run

    @staticmethod
    def write_contract(run, index, ids, **changes):
        contract = {key: 'same' for key in merge.SHARED_CONTRACT_KEYS}
        contract.update(num_shards=2, rules=dict(waypoint_radius=1.8), shard_index=index, episode_ids=ids)
        contract.update(changes)
        Path(run, f'contract.shard{index:03d}.json').write_text(json.dumps(contract))

    def test_dagger_manifest_and_audit(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root)
            out = Path(root, 'dagger.jsonl')
            self.run_merge('--run-dir', str(run), '--out', str(out))
            rows = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            report = json.loads(out.with_suffix('.audit.json').read_text())
            audit = report['dagger']
            self.assertEqual((audit['attempted'], audit['saved']), (4, 3))
            self.assertEqual(audit['reasons'], dict(takeover=3, near_oracle=1))
            self.assertEqual((report['shards'], report['rules']), (2, dict(waypoint_radius=1.8)))
            with self.assertRaises(ValueError):
                self.run_merge('--run-dir', str(run), '--out', str(out))         # never overwrites

    def oracle(self, root, r2r=6, rxr=4):
        rows = []
        for i in range(r2r + rxr):
            row = json.loads(json.dumps(saved_record(CollectionStore(Path(root, 'o'), i), str(100 + i),
                                                     seed=50 + i)[0]['row']))
            dataset = 'r2rce' if i < r2r else 'rxrce'
            row.update(episode_uid=f'{dataset}:train:scene:{i}:{i}', dataset=dataset,
                       dataset_version='sha-r2r' if dataset == 'r2rce' else 'sha-rxr')
            rows.append(row)
        path = Path(root, 'oracle.jsonl')
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return path

    def test_mixture_replaces_only_r2r_rows_deterministically(self):
        with tempfile.TemporaryDirectory() as root:
            run, oracle = self.make_run(root), self.oracle(root)
            outputs = []
            for name in ('a.jsonl', 'b.jsonl'):
                out = Path(root, name)
                self.run_merge('--run-dir', str(run), '--out', str(out), '--oracle-manifest', str(oracle),
                               '--dagger-fraction', '0.2')
                outputs.append(out.read_text())
                report = json.loads(out.with_suffix('.audit.json').read_text())['mixture']
                self.assertEqual((report['total'], report['dagger_rows'], report['replaced_oracle_rows']), (10, 1, 1))
                self.assertEqual(report['datasets'], dict(rxrce=4, r2rce=5, r2rce_dagger=1))
                self.assertTrue(report['r2r_source_matches_oracle'])
            self.assertEqual(outputs[0], outputs[1])
            out = Path(root, 'c.jsonl')
            self.run_merge('--run-dir', str(run), '--out', str(out), '--oracle-manifest', str(oracle),
                           '--dagger-fraction', '0.9')                       # 5 wanted, only 3 DAgger rows exist
            report = json.loads(out.with_suffix('.audit.json').read_text())['mixture']
            self.assertEqual(report['datasets'], dict(rxrce=4, r2rce=3, r2rce_dagger=3))

    def test_duplicate_episode_across_shards_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root)
            CollectionStore(run, 1).append(dict(episode_id='10', saved=False, reason='failure'))
            self.write_contract(run, 1, ['11', '10'])
            with self.assertRaisesRegex(ValueError, 'journaled by'):
                self.run_merge('--run-dir', str(run), '--out', str(Path(root, 'x.jsonl')))

    def test_incomplete_or_inconsistent_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root)
            self.write_contract(run, 1, ['11', '14'])                         # 14 not collected yet
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                self.run_merge('--run-dir', str(run), '--out', str(Path(root, 'x.jsonl')))
            self.run_merge('--run-dir', str(run), '--out', str(Path(root, 'y.jsonl')), '--allow-partial')
            self.write_contract(run, 1, ['11'], weights_sha256='other')
            with self.assertRaisesRegex(ValueError, 'differs'):
                self.run_merge('--run-dir', str(run), '--out', str(Path(root, 'z.jsonl')))
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root, contracts=False)
            with self.assertRaisesRegex(ValueError, 'No shard contracts'):
                self.run_merge('--run-dir', str(run), '--out', str(Path(root, 'x.jsonl')))


if __name__ == '__main__':
    unittest.main()
