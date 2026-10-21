"""Immutable subset and complete labeled-frame prerequisites."""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from scripts.diag.prepare_expert import write_subset, check_frames

class ExpertTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/'official.json.gz'
        self.data = {'instruction_vocab': {'word_list': ['a']}, 'episodes': [
            {'episode_id': i, 'scene_id': 'scene', 'instruction': {'instruction_text': str(i)},
             'start_position': [i, 0, 0], 'start_rotation': [0, 0, 0, 1]} for i in (1, 2, 3)]}
        with gzip.open(self.source, 'wt') as f: json.dump(self.data, f)
        self.before = self.source.read_bytes()
    def test_subset_exact_order_metadata_and_immutable_source(self):
        target = self.root/'subset.json.gz'
        provenance = write_subset(self.source, ['3', '1'], target)
        with gzip.open(target, 'rt') as f: result = json.load(f)
        self.assertEqual(result, dict(self.data, episodes=[self.data['episodes'][2], self.data['episodes'][0]]))
        self.assertEqual(self.source.read_bytes(), self.before)
        self.assertNotEqual(provenance['parent_sha256'], provenance['subset_sha256'])
        second = self.root/'second.json.gz'
        write_subset(self.source, ['3', '1'], second)
        self.assertEqual(second.read_bytes(), target.read_bytes())
    def test_reject_missing_duplicate_ids_and_same_destination(self):
        for ids in (['8'], ['1', '1'], []):
            with self.assertRaises(ValueError): write_subset(self.source, ids, self.root/'bad.gz')
        with self.assertRaises(ValueError): write_subset(self.source, ['1'], self.source)
        self.assertEqual(self.source.read_bytes(), self.before)
    def make_frames(self):
        folder = self.root/'frames'/'1'; folder.mkdir(parents=True)
        for name in ('step_0000_MOVE_FORWARD.png', 'step_0001_STOP.png'):
            Image.new('RGB', (640, 480)).save(folder/name)
        return folder
    def test_complete_labels_include_stop(self):
        self.make_frames()
        report = check_frames(self.root/'frames', ['1'], {'1': {'actions': [1, 0]}})
        self.assertEqual(report['total_frames'], 2)
        self.assertEqual(report['episodes'][0]['frame_count'], 2)
    def test_missing_stop_wrong_action_or_extra_frame_rejected(self):
        folder = self.make_frames()
        stop = folder/'step_0001_STOP.png'
        stop.unlink()
        with self.assertRaises(ValueError): check_frames(folder.parent, ['1'], {'1': {'actions': [1, 0]}})
        Image.new('RGB', (640, 480)).save(folder/'step_0001_TURN_LEFT.png')
        with self.assertRaises(ValueError): check_frames(folder.parent, ['1'], {'1': {'actions': [1, 0]}})
    def test_unchanged_converter_uses_subset_hash_and_validates_stop(self):
        self.make_frames()
        subset = self.root/'subset.json.gz'
        evidence = write_subset(self.source, ['1'], subset)
        repo = Path(__file__).resolve().parents[2]
        manifest = self.root/'episodes.jsonl'
        result = subprocess.run([sys.executable, str(repo/'scripts/vln/prepare_r2r.py'),
            '--data-root', str(self.root), '--source', str(subset), '--split', 'val_unseen',
            '--frames-root', str(self.root/'frames'), '--out', str(manifest)],
            env=dict(os.environ, PYTHONPATH=str(repo/'src')), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        row = json.loads(manifest.read_text())
        self.assertEqual(row['episode_id'], '1')
        self.assertEqual(row['dataset_version'], evidence['subset_sha256'])
        self.assertEqual([s['action_name'] for s in row['steps']], ['MOVE_FORWARD', 'STOP'])
        self.assertEqual(row['observation_action_alignment'], 'observation_before_action')
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_wrong_dimensions_rejected(self):
        folder = self.make_frames()
        Image.new('RGB', (320, 240)).save(folder/'step_0001_STOP.png')
        with self.assertRaises(ValueError): check_frames(folder.parent, ['1'], {'1': {'actions': [1, 0]}})

if __name__ == '__main__': unittest.main()
