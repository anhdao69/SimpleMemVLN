"""CPU-only deterministic panel selection and prerequisite gatekeeping."""
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).with_name('gate_queue.py')
spec=importlib.util.spec_from_file_location('gate_queue',path)
g=importlib.util.module_from_spec(spec) if path.exists() else None
if g is not None: spec.loader.exec_module(g)

class GateQueueTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(g,'queue missing')

    def test_selection_is_lexicographic_not_numeric_and_fixed(self):
        panel={'episodes':[dict(episode_id=x) for x in ['2','11','100','3']]}
        self.assertEqual(g.select_ids(panel),['100','11'])
        self.assertEqual(panel['episodes'][0]['episode_id'],'2')
        with self.assertRaises(ValueError):g.select_ids({'episodes':[dict(episode_id='x')]*2})

    def test_wait_until_both_teacher_and_preparation_complete(self):
        ready={'status':'complete','returncode':0}
        prep={'status':'complete','episode_count':240}
        complete={'episodes':240}
        self.assertTrue(g.prerequisites_ready(ready,prep,complete))
        self.assertFalse(g.prerequisites_ready({'status':'running'},prep,complete))
        self.assertFalse(g.prerequisites_ready(ready,prep,None))
        with self.assertRaises(RuntimeError):g.prerequisites_ready({'status':'failed'},prep,None)
        with self.assertRaises(RuntimeError):g.prerequisites_ready(ready,{'status':'error'},complete)
        with self.assertRaises(ValueError):g.prerequisites_ready(ready,prep,{'episodes':239})

    def test_manifest_uid_mapping_matches_declared_identity(self):
        panel={'episodes':[dict(episode_id='1007',instruction_id='1007',scene_id='scene',gt_action_count=2),
                           dict(episode_id='1013',instruction_id='1013',scene_id='other',gt_action_count=1)]}
        manifest=[dict(episode_id='1013',instruction_id='1013',scene_id='other',steps=[{}],episode_uid='u2'),
                  dict(episode_id='1007',instruction_id='1007',scene_id='scene',steps=[{},{}],episode_uid='u1')]
        self.assertEqual(g.selected_uids(panel,manifest,['1007','1013']),['u1','u2'])
        manifest[0]['scene_id']='wrong'
        with self.assertRaises(ValueError):g.selected_uids(panel,manifest,['1007','1013'])

if __name__=='__main__':unittest.main()
