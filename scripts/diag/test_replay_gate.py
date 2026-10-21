"""CPU contracts for independent frozen-source replay comparisons."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

path = Path(__file__).with_name('replay_gate.py')
spec = importlib.util.spec_from_file_location('replay_gate', path)
gate = importlib.util.module_from_spec(spec) if path.exists() else None
if gate is not None:
    spec.loader.exec_module(gate)

class ReplayGateTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(gate, 'replay_gate implementation is missing')

    def test_rms_boundary_and_combined_relative_tolerance(self):
        ref = np.zeros((1, 100)); candidate = ref.copy(); candidate[0, 0] = 1.5
        ref[0, 0] = 25.; candidate[0, 0] += 25.
        result = gate.compare_logits(candidate, ref)
        self.assertTrue(result['passed'])  # 1.5 == .75 + .03 * 25; RMS == .15
        self.assertEqual(result['max_abs'], 1.5)
        candidate[0, 0] += .0001
        self.assertFalse(gate.compare_logits(candidate, ref)['passed'])

    def test_elementwise_failure_even_with_small_rms(self):
        ref = np.zeros((1, 100)); candidate = ref.copy(); candidate[0, 0] = .76
        result = gate.compare_logits(candidate, ref)
        self.assertFalse(result['passed'])
        self.assertEqual(result['outside_tolerance'], 1)

    def test_argmax_flips_and_rms_failure(self):
        result = gate.compare_logits(np.array([[.2, 0.]]), np.array([[0., .2]]))
        self.assertEqual(result['argmax_flips'], 1)
        self.assertFalse(result['passed'])

    def test_invalid_logits_rejected(self):
        for a, b in [(np.zeros((2, 2)), np.zeros((1, 2))),
                     (np.array([[float('nan')]]), np.zeros((1, 1))),
                     (np.zeros((0, 1)), np.zeros((0, 1)))]:
            with self.assertRaises(ValueError): gate.compare_logits(a, b)

    def test_plan_overlap_empty_and_different_cardinality(self):
        result = gate.compare_plans({'3': [[], [1, 2]]}, {'3': [[], [2, 3]]})
        self.assertEqual(result['per_layer']['3']['mean_selected_overlap'], .5)
        self.assertEqual(result['per_layer']['3']['empty_steps'], 1)
        self.assertFalse(result['exact'])
        with self.assertRaises(ValueError): gate.compare_plans({'3': [[]]}, {})
        with self.assertRaises(ValueError): gate.compare_plans({'3': [[1]]}, {'3': [[1, 2]]})

    def test_mapping_preserves_identity_labels_and_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'rgb-000.png').write_bytes(b'original image')
            ep = dict(episode_uid='keep-me', scene_id='/old/scene.glb',
                      steps=[dict(rgb_path='/old/rgb-000.png', action_name='STOP', step_id=0)])
            original = copy.deepcopy(ep)
            mapped, evidence = gate.relocate_fixture(ep, tmp)
            self.assertEqual(ep, original)
            self.assertEqual(mapped['episode_uid'], ep['episode_uid'])
            self.assertEqual(mapped['scene_id'], ep['scene_id'])
            self.assertEqual(mapped['steps'][0]['action_name'], 'STOP')
            self.assertEqual(evidence[0]['original'], '/old/rgb-000.png')
            self.assertEqual(len(evidence[0]['sha256']), 64)
            ep['steps'][0]['rgb_path'] = '/old/missing.png'
            with self.assertRaises(FileNotFoundError): gate.relocate_fixture(ep, tmp)

if __name__ == '__main__': unittest.main()
