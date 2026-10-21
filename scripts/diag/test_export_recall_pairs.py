import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.diag.export_recall_pairs import export_recall_pairs


class RecallPairExportTests(unittest.TestCase):
    def test_exports_only_required_layers_and_verifies_capture_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'captures'
            panel = Path(tmp) / 'panel.json'
            output = Path(tmp) / 'native_pairs.csv'
            panel.write_text(json.dumps({'episodes': [
                {'episode_id': 'e1', 'instruction_id': 'i1', 'scene_id': 's1', 'gt_action_count': 4},
            ]}))
            for model, layers in {'M5_E1': [22, 24, 26], 'G3_M5': [22, 24, 26],
                                  'TRIPLE': [8, 16, 22, 24, 26], 'G3': [22, 24, 26],
                                  'G1': [22, 24, 26]}.items():
                episode = root / model / 'episode_000'
                episode.mkdir(parents=True)
                files = {}
                for layer in layers:
                    name = f'recall_layer_{layer}.csv'
                    data = b'anchor,source,lag,lag_group,lag_bucket,region,epsilon,source_epsilon,forgetting\n2,1,1,A,1-3,Global,0.2,0.1,0.1\n'
                    (episode / name).write_bytes(data)
                    files[name] = hashlib.sha256(data).hexdigest()
                (episode / 'COMPLETE.json').write_text(json.dumps({
                    'episode_id': 'e1', 'episode_uid': 'u1', 'scene_id': 's1',
                    'frames': 4, 'layers': layers, 'files_sha256': files,
                }))
            manifest = export_recall_pairs(root, panel, output)
            with output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 17)
            self.assertEqual(manifest['models']['TRIPLE']['layers'], [8, 16, 22, 24, 26])
            self.assertEqual(manifest['rows'], 17)
            self.assertEqual(rows[0]['model'], 'M5_E1')
            self.assertTrue(all('instruction_id' in r for r in rows))

    def test_rejects_capture_file_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'captures'; panel = Path(tmp) / 'panel.json'; output = Path(tmp) / 'out.csv'
            panel.write_text(json.dumps({'episodes': [{'episode_id': 'e1', 'scene_id': 's1'}]}))
            ep = root / 'M5_E1' / 'episode_000'; ep.mkdir(parents=True)
            (ep / 'recall_layer_22.csv').write_text('anchor,source,lag,lag_group,lag_bucket,region,epsilon,source_epsilon,forgetting\n2,1,1,A,1-3,Global,0.2,0.1,0.1\n')
            (ep / 'COMPLETE.json').write_text(json.dumps({'episode_id': 'e1', 'scene_id': 's1',
                'layers': [22], 'files_sha256': {'recall_layer_22.csv': 'wrong'}}))
            with self.assertRaisesRegex(ValueError, 'capture hash mismatch'):
                export_recall_pairs(root, panel, output, models=('M5_E1',), required_layers={'M5_E1': [22]})
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
