"""D-b driver evidence and prefix-gate bookkeeping, no GPU."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).with_name('run_bank_ablation.py')
spec=importlib.util.spec_from_file_location('run_bank_ablation',path)
r=importlib.util.module_from_spec(spec) if path.exists() else None
if r is not None:spec.loader.exec_module(r)

class DriverTests(unittest.TestCase):
    def test_existing_adapter_and_global_kl_guards_reject_nonfinite_logits(self):
        import numpy as np
        import torch
        from scripts.diag.instrument_gate import bitwise_logits
        from scripts.diag.bank_ablation import logit_kl
        good=np.array([[1.,2.]],dtype=np.float32)
        for value in (np.nan,np.inf,-np.inf):
            bad=np.array([[1.,value]],dtype=np.float32)
            with self.subTest(value=value):
                with self.assertRaises(ValueError):bitwise_logits(torch.from_numpy(good),torch.from_numpy(bad))
                with self.assertRaises(ValueError):logit_kl(good,bad)
                with self.assertRaises(ValueError):logit_kl(bad,good)

    def setUp(self):self.assertIsNotNone(r,'D-b driver missing')
    def test_baseline_requires_full_same_model_panel_and_weights(self):
        p=dict(model='M5_E1',weights_sha256='w',manifest_sha256='m',panel_sha256='p')
        c=dict(episodes=240,weights_sha256='w')
        r.validate_baseline_contract(p,c,'M5_E1','w','m','p')
        for changed in [dict(c,episodes=239),dict(c,weights_sha256='other')]:
            with self.assertRaises(ValueError):r.validate_baseline_contract(p,changed,'M5_E1','w','m','p')
        with self.assertRaises(ValueError):r.validate_baseline_contract(p,c,'G3_M5','w','m','p')
    def test_prefix_gate_selection_uses_only_preselected_anchors(self):
        self.assertEqual(r.prefix_gate_anchors({'anchors':[9,17,35,60]},79),[9,17,35,60])
        self.assertEqual(r.prefix_gate_anchors({'anchors':[]},20),[19])
        with self.assertRaises(ValueError):r.prefix_gate_anchors({'anchors':[80]},79)
    def test_gate_phase_cannot_be_claimed_complete_with_one_episode(self):
        rows=[dict(episode_id='1007',adapter_bitwise=True,plans_equal=True,
                   prefix_sensitivity=[dict(relative_frobenius_error=.02)])]
        self.assertFalse(r.adapter_gate_passed(rows,['1007','1013']))
        rows.append(dict(rows[0],episode_id='1013'))
        self.assertTrue(r.adapter_gate_passed(rows,['1007','1013']))
        rows[1]=dict(rows[1],adapter_bitwise=False)
        self.assertFalse(r.adapter_gate_passed(rows,['1007','1013']))
    def test_length_truncation_sensitivity_is_reported_but_not_a_parity_gate(self):
        rows=[dict(episode_id=episode,adapter_bitwise=True,plans_equal=True,
                   prefix_sensitivity=dict(relative_frobenius_error=.034,
                                           interpretation='sequence_length_effect'))
              for episode in ('1007','1013')]
        self.assertTrue(r.adapter_gate_passed(rows,['1007','1013']))

if __name__=='__main__':unittest.main()
