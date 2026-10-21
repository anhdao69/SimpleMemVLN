import unittest
import tempfile
from pathlib import Path
from scripts.diag.measure_panel import validate_gate_evidence, checkpoint_weight, recall_denominators


class MeasurementGateTests(unittest.TestCase):
    def test_nonfinite_logits_fail_without_changing_valid_tensor(self):
        import torch
        from scripts.diag.measure_panel import require_finite_logits
        finite=torch.tensor([[1.,-2.,0.]],dtype=torch.bfloat16)
        original=finite.clone();version=finite._version
        self.assertIs(require_finite_logits(finite),finite)
        self.assertEqual(finite.dtype,torch.bfloat16)
        self.assertTrue(torch.equal(finite,original))
        self.assertEqual(finite._version,version)
        for value in (float('nan'),float('inf'),-float('inf')):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'Nonfinite'):
                require_finite_logits(torch.tensor([[0.,value]],dtype=torch.bfloat16))

    def good(self):
        return dict(status='PASS',model='M5_E1',weights_sha256='abc',panel_episodes_completed=2,
            selected_episode_uids=['a','b'],input_capture_layers=[2,6,8,10,14,16,18,22,24,26,30],
            attention_capture_layers=[3,7,11,15,19,23,27,31],
            rows=[dict(episode_uid=uid,input_kind='panel',g1_pass=True,g3_pass=True,attention_pass=True) for uid in ['a','b']])
    def test_weight_identity_follows_native_loader_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'pytorch_model.bin').touch()
            self.assertEqual(checkpoint_weight(root).name,'pytorch_model.bin')
            (root/'model.safetensors').touch()
            self.assertEqual(checkpoint_weight(root).name,'model.safetensors')

    def test_recall_raw_denominators_respect_unequal_quadrants(self):
        rows=[dict(anchor=2,source=0,region=r) for r in ['Q1','Q2','Q3','Q4','Global']]
        rows += [dict(anchor=2,source=2,region=r) for r in ['Q1','Q2','Q3','Q4','Global']]
        counts=recall_denominators(rows)
        self.assertEqual(counts['unique_anchor_source_pairs'],2)
        self.assertEqual(counts['region_rows'],10)
        self.assertEqual(counts['regions']['Q1']['token_evaluations'],140)
        self.assertEqual(counts['regions']['Q3']['token_evaluations'],160)
        self.assertEqual(counts['regions']['Global']['token_evaluations'],600)

    def test_gate_binds_manifest_bytes_and_declared_selection(self):
        gate=self.good();gate['inputs']=[dict(path='/manifest.jsonl',sha256='manifest')]
        validate_gate_evidence(gate,'M5_E1','abc',{'a','b'},manifest_hash='manifest')
        with self.assertRaises(ValueError):
            validate_gate_evidence(gate,'M5_E1','abc',{'a','b'},manifest_hash='changed')
        gate['selected_episode_uids']=['a','c']
        with self.assertRaises(ValueError):
            validate_gate_evidence(gate,'M5_E1','abc',{'a','b'},manifest_hash='manifest')

    def test_requires_all_hooks_two_distinct_panel_episodes_and_matching_model(self):
        g=self.good();validate_gate_evidence(g,'M5_E1','abc',{'a','b','c'})
        for field,value in [('status','PARTIAL'),('weights_sha256','wrong'),('input_capture_layers',[8,16,22,24,26])]:
            bad=dict(g);bad[field]=value
            with self.assertRaises(ValueError):validate_gate_evidence(bad,'M5_E1','abc',{'a','b','c'})
        bad=self.good();bad['rows'][1]['attention_pass']=False
        with self.assertRaises(ValueError):validate_gate_evidence(bad,'M5_E1','abc',{'a','b','c'})
        with self.assertRaises(ValueError):validate_gate_evidence(g,'M5_E1','abc',{'a','c'})


if __name__=='__main__':unittest.main()
