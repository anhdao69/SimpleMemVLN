import unittest
import numpy as np
from scripts.diag.probe_drift import bitwise_invariant, drift_metrics
from scripts.diag import probe_drift
from unittest.mock import patch
from types import SimpleNamespace


class ProbeDriftTests(unittest.TestCase):
    def test_invariance_requires_logits_and_plans(self):
        a = np.array([[0., 1.]], dtype=np.float32)
        plans = {'3': [[1, 2], []]}
        self.assertTrue(bitwise_invariant(a, a.copy(), plans, plans))
        b = a.copy(); b[0, 0] = -0.
        self.assertFalse(bitwise_invariant(a, b, plans, plans))
        self.assertFalse(bitwise_invariant(a, a, plans, {'3': [[2, 1], []]}))

    def test_dummy_scope_restores_prior_state_after_exception(self):
        original_head=object();original_spec={'run_id':'existing'}
        model=SimpleNamespace(g1_spec=original_spec,research_aux=original_head)
        def attach(m,research,seed):
            m.g1_spec={'seed':seed};m.research_aux=object()
        with patch.object(probe_drift,'attach_dummy_heads',side_effect=attach):
            with self.assertRaisesRegex(RuntimeError,'forward failed'):
                with probe_drift.temporary_dummy_heads(model,{},429):
                    raise RuntimeError('forward failed')
        self.assertIs(model.g1_spec,original_spec)
        self.assertIs(model.research_aux,original_head)

    def test_dummy_scope_restores_serving_state_after_partial_attachment_failure(self):
        model=SimpleNamespace(g1_spec=None)
        def attach(m,research,seed):
            m.g1_spec={'seed':seed}
            raise RuntimeError('attachment failed')
        with patch.object(probe_drift,'attach_dummy_heads',side_effect=attach):
            with self.assertRaises(RuntimeError):
                with probe_drift.temporary_dummy_heads(model,{},430):pass
        self.assertIsNone(model.g1_spec)
        self.assertFalse(hasattr(model,'research_aux'))

    def test_drift_reports_effect_without_parity_cutoff(self):
        a = np.array([[1., 0.], [0., 2.]], dtype=np.float32)
        b = np.array([[0., 1.], [0., 2.]], dtype=np.float32)
        result = drift_metrics(a, b, {'3': [[], [1, 2]]}, {'3': [[], [2, 3]]})
        self.assertEqual(result['argmax_flips'], 1)
        self.assertAlmostEqual(result['rms'], np.sqrt(.5))
        self.assertEqual(result['per_layer']['3']['mean_selected_overlap'], .5)
        self.assertEqual(result['per_layer']['3']['empty_steps'], 1)
        self.assertNotIn('passed', result)


if __name__ == '__main__':
    unittest.main()
