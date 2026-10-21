"""CPU-only coverage and bitwise contracts for the instrument gate runner."""
import importlib.util
from pathlib import Path
import tempfile
import json
import unittest
import torch

path=Path(__file__).with_name('instrument_gate.py')
spec=importlib.util.spec_from_file_location('instrument_gate',path)
gate=importlib.util.module_from_spec(spec) if path.exists() else None
if gate is not None: spec.loader.exec_module(gate)

class InstrumentTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(gate,'instrument runner is missing')

    def test_one_fixture_cannot_complete_two_panel_gate(self):
        report=gate.coverage([dict(episode_uid='x',input_kind='fixture',g1_pass=True,g3_pass=True,dc_pass=True)])
        self.assertEqual(report['status'],'PARTIAL')
        self.assertTrue(report['passed_for_checked_episodes'])
        self.assertEqual(report['panel_episodes_completed'],0)

    def test_two_distinct_panel_episodes_complete(self):
        rows=[dict(episode_uid=x,input_kind='panel',g1_pass=True,g3_pass=True,dc_pass=True) for x in ['x','y']]
        self.assertEqual(gate.coverage(rows)['status'],'PASS')
        rows[1]['g3_pass']=False
        self.assertEqual(gate.coverage(rows)['status'],'FAIL')
        rows[1]['episode_uid']='x'
        with self.assertRaises(ValueError): gate.coverage(rows)

    def test_attention_failure_prevents_completed_gate(self):
        rows=[dict(episode_uid=x,input_kind='panel',g1_pass=True,g3_pass=True,dc_pass=False) for x in ['x','y']]
        self.assertEqual(gate.coverage(rows)['status'],'FAIL')

    def test_capture_union_covers_scorers_only_for_episodic_models(self):
        self.assertEqual(gate.capture_layers(True),(2,6,8,10,14,16,18,22,24,26,30))
        self.assertEqual(gate.capture_layers(False),(8,16,22,24,26))

    def test_raw_attention_steps_fixed_before_values(self):
        self.assertEqual(gate.raw_attention_steps(79),(0,78))
        self.assertEqual(gate.raw_attention_steps(1),(0,))

    def test_g1_is_bitwise_not_tolerance(self):
        x=torch.ones(2,3)
        self.assertTrue(gate.bitwise_logits(x,x.clone())['passed'])
        y=x.clone(); y[0,0]+=1e-6
        self.assertFalse(gate.bitwise_logits(x,y)['passed'])
        with self.assertRaises(ValueError): gate.bitwise_logits(x,x[:1])
        with self.assertRaises(ValueError): gate.bitwise_logits(x,torch.full_like(x,float('nan')))

    def test_manifest_selection_keeps_episode_identity_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'episodes.jsonl'
            rows=[dict(episode_uid=x,scene_id='original',steps=[]) for x in ['a','b','c']]
            p.write_text('\n'.join(json.dumps(x) for x in rows))
            result=gate.read_episodes([],str(p),['c','a'])
            self.assertEqual([x['episode_uid'] for x in result],['c','a'])
            self.assertEqual(result[0]['scene_id'],'original')
            with self.assertRaises(ValueError): gate.read_episodes([],str(p),['missing'])
            with self.assertRaises(ValueError): gate.read_episodes([],str(p),['a','a'])

if __name__=='__main__': unittest.main()
