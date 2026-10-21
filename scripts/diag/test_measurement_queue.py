import unittest
from scripts.diag.measurement_queue import prerequisite, validate_all_gates
from scripts.diag.measure_panel import M5_CAPTURE, WINDOW_CAPTURE, ATTENTION


class MeasurementQueueTests(unittest.TestCase):
    def test_all_individual_gates_must_pass_before_first_measurement(self):
        names=['M5_E1','G3_M5','TRIPLE','G3','G1']
        models=[dict(model=name,weights_sha256=name) for name in names]
        results={name:dict(status='PASS',model=name,weights_sha256=name,
            panel_episodes_completed=2,inputs=[dict(sha256='manifest')],selected_episode_uids=['a','b'],
            input_capture_layers=list(M5_CAPTURE if name in names[:3] else WINDOW_CAPTURE),
            attention_capture_layers=list(ATTENTION),rows=[dict(episode_uid=uid,input_kind='panel',
                g1_pass=True,g3_pass=True,attention_pass=True) for uid in ['a','b']]) for name in names}
        validate_all_gates(models,results,{'a','b'},'manifest')
        results['G1']['status']='FAIL'
        with self.assertRaises(ValueError):validate_all_gates(models,results,{'a','b'},'manifest')
        with self.assertRaises(ValueError):validate_all_gates(models[:-1],results,{'a','b'},'manifest')

    def test_gate_failure_or_missing_coverage_cannot_launch(self):
        self.assertFalse(prerequisite(None))
        with self.assertRaises(RuntimeError):prerequisite({'status':'failed'})
        with self.assertRaises(ValueError):prerequisite({'status':'complete','models_completed':['M5_E1']})
        self.assertTrue(prerequisite(dict(status='complete',models_completed=['M5_E1','G3_M5','TRIPLE','G3','G1'])))


if __name__=='__main__':unittest.main()
