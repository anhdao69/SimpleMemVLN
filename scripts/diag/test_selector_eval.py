import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from scripts.diag import selector_eval as e, selector_server as s

class SelectorTests(unittest.TestCase):
    def test_random_override_is_explicit_copy_and_keeps_original(self):
        from qwen_vl.research.episodic_contract import parse_episodic_spec
        config={'model':{'output_mode':'qwen_text'},
                'memory':{'mode':'episodic_window_v1','recent_groups_including_current':4,
                          'episodic_budget_tokens':1200,'episodic_score':'committed_mean_l2_v1',
                          'episodic_selector_seed':429,'evict_before_new_step':True,'kv_guard_tokens':4784},
                'observations':{'max_prefix_tokens':2048,'max_step_group_tokens':384},
                'runtime':{'text_attention':'simplememvln_step_attention'}}
        old=copy.deepcopy(config)
        effective,original,new=s.effective_config(config,'random')
        self.assertEqual(config,old)
        self.assertEqual(new.score,'random_hash_v1');self.assertEqual(new.selector_seed,429)
        self.assertEqual(new.recent_groups,4);self.assertEqual(new.budget_tokens,1200)
        self.assertEqual(original.score,'committed_mean_l2_v1')
        self.assertEqual(s.effective_config(config,'score')[0],config)

    def test_panel_and_determinism_subset_strict(self):
        panel={'episodes':[{'episode_id':str(i)} for i in range(600)],'metadata':{'determinism_episode_ids':[str(i) for i in range(60)]}}
        self.assertEqual(len(e.panel_ids(panel,'random')),600)
        self.assertEqual(len(e.panel_ids(panel,'score')),60)
        panel['metadata']['determinism_episode_ids'][0]='missing'
        with self.assertRaises(ValueError):e.panel_ids(panel,'score')

    def test_baseline_uid_instruction_validation(self):
        ep=SimpleNamespace(episode_id='1',scene_id='/original/scene.glb',instruction=SimpleNamespace(instruction_text=' hello '))
        row={'episode_id':'1','episode_uid':'val_unseen:/original/scene.glb:1','scene_id':ep.scene_id,'instruction':'hello'}
        e.validate_baseline([ep],{'1':row})
        row['episode_uid']='val_unseen:/relocated/scene.glb:1'
        with self.assertRaises(ValueError):e.validate_baseline([ep],{'1':row})

    def test_source_hash_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'src/qwen_vl';p.mkdir(parents=True);(p/'x.py').write_text('frozen')
            c={'source_sha256':{'x.py':e.sha256(p/'x.py')}}
            e.verify_source(root,c)
            (p/'x.py').write_text('changed')
            with self.assertRaises(ValueError):e.verify_source(root,c)

if __name__=='__main__':unittest.main()
