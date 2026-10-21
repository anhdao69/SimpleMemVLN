import unittest
import numpy as np
from scripts.diag import summarize_captures as s

class SummaryTests(unittest.TestCase):
    def test_episode_unit_prevents_long_episode_domination(self):
        c=s.Collector();meta={'diagnostic':'D-a','model':'x','metric':'epsilon'}
        for value in (0,0,0):c.add(meta,{'episode_id':'a','scene_id':'same'},value,counts={'pairs':1})
        c.add(meta,{'episode_id':'b','scene_id':'same'},10,counts={'pairs':1})
        result=c.summaries()[0]
        self.assertEqual(result['estimate'],5.)
        self.assertEqual(result['n_episodes'],2)
        self.assertEqual(result['counts']['pairs'],4)

    def test_combined_bank_normalizes_by_joint_causal_count(self):
        row={'query_roles':['anchor','visual'],'query_ids':[9,8],
             'categories':{'bank_lag_8_15':{'visible_tokens':[2,0],'mass':[[.2,.4],[0,0]]},
                           'bank_lag_16_32':{'visible_tokens':[8,0],'mass':[[.4,.6],[0,0]]}}}
        out=s.attention_rows(row,strict=False)
        result=next(r for r in out if r['role']=='anchor' and r['category']=='bank_lag_8_32')
        self.assertAlmostEqual(result['mass_per_token'],.08)
        self.assertEqual(result['n_visible_tokens'],10)
        absent=next(r for r in out if r['role']=='visual' and r['category']=='bank_lag_8_32')
        self.assertIsNone(absent['mass_per_token'])
        self.assertEqual(absent['mass'],0)

    def test_attention_roles_remain_separate(self):
        row={'query_roles':['anchor','visual'],'query_ids':[9,8],
             'categories':{'prefix':{'visible_tokens':[1,1],'mass':[[.1],[.9]]}}}
        out=s.attention_rows(row,strict=False)
        values={r['role']:r['mass'] for r in out if r['category']=='prefix'}
        self.assertAlmostEqual(values['anchor'],.1);self.assertAlmostEqual(values['visual'],.9)
        self.assertAlmostEqual(values['combined'],.5)

    def test_paired_contrasts_reject_scene_change(self):
        c=s.Collector();base={'diagnostic':'D-a','metric':'forgetting'}
        c.add(dict(base,model='left'),{'episode_id':'a','scene_id':'x'},1)
        c.add(dict(base,model='right'),{'episode_id':'a','scene_id':'y'},2)
        with self.assertRaises(ValueError):c.contrasts('left','right')

    def test_undefined_is_not_zero(self):
        c=s.Collector();meta={'diagnostic':'D-f','metric':'overlap'}
        c.add(meta,{'episode_id':'a','scene_id':'s'},None,counts={'empty_steps':4})
        out=c.summaries()[0];self.assertIsNone(out['estimate']);self.assertEqual(out['n_undefined'],1)
        self.assertEqual(out['counts']['empty_steps'],4)

if __name__=='__main__':unittest.main()
