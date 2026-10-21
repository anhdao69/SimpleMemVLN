import unittest
from types import SimpleNamespace
from scripts.diag.post_measurement_queue import ordered_models, validate_completion, owned_groups


class PostQueueTests(unittest.TestCase):
    def test_model_subsets_cannot_depend_on_input_order(self):
        names=['G1','TRIPLE','M5_E1','G3','G3_M5']
        models=ordered_models([{'model':name} for name in names])
        self.assertEqual([m['model'] for m in models[:3]],['M5_E1','G3_M5','TRIPLE'])
        self.assertEqual([m['model'] for m in models[:2]],['M5_E1','G3_M5'])
        for bad in (models[:-1],models+[models[0]],models[:-1]+[models[0]]):
            with self.assertRaises(ValueError):ordered_models(bad)

    def test_stage_completion_rejects_missing_or_wrong_counts(self):
        validate_completion({'status':'complete','models':['a','b']},status='complete',models=['a','b'])
        for record in (None,{},dict(episodes=239),dict(episodes=240,adapter_gate_passed=False)):
            with self.assertRaises(ValueError):validate_completion(record,episodes=240,adapter_gate_passed=True)
        validate_completion(dict(episodes=240,adapter_gate_passed=True),episodes=240,adapter_gate_passed=True)

    def test_owned_groups_include_exited_leader_with_possible_descendants(self):
        children={'done':SimpleNamespace(pid=12345,poll=lambda:1),
                  'running':SimpleNamespace(pid=23456,poll=lambda:None)}
        self.assertEqual(owned_groups(children),[12345,23456])

if __name__=='__main__':unittest.main()
