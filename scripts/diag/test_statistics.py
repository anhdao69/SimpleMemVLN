import unittest
from scripts.diag import statistics as s

class StatisticsTests(unittest.TestCase):
    def test_paired_contrast_and_scene_counts(self):
        rows=[dict(episode_id=str(i),scene='a' if i<3 else 'b',left=100*i,right=100*i+2) for i in range(5)]
        out=s.paired_bootstrap(rows)
        self.assertEqual(out['estimate'],2)
        self.assertEqual(out['ci95'],[2.,2.])
        self.assertEqual(out['n_episodes'],5)
        self.assertEqual(out['resamples'],2000)
        self.assertEqual(out['scene_counts'],{'a':3,'b':2})
        self.assertEqual(out,s.paired_bootstrap(list(reversed(rows))))

    def test_missing_is_excluded_pairwise_and_reported(self):
        rows=[dict(episode_id='1',scene='a',left=1,right=3),dict(episode_id='2',scene='a',left=None,right=3)]
        out=s.paired_bootstrap(rows)
        self.assertEqual(out['n_episodes'],1)
        self.assertEqual(out['n_undefined'],1)
        self.assertEqual(out['estimate'],2)
        self.assertIsNone(s.paired_bootstrap(rows[1:])['ci95'])
        with self.assertRaises(ValueError):s.paired_bootstrap(rows+rows)
        with self.assertRaises(ValueError):s.paired_bootstrap([dict(episode_id='x',scene='a',left=float('inf'),right=3)])

    def test_bootstrap_keeps_scene_composition(self):
        rows=[dict(episode_id='0',scene='a',left=0,right=0),dict(episode_id='1',scene='b',left=0,right=10)]
        self.assertEqual(s.paired_bootstrap(rows)['ci95'],[5.,5.])

    def test_exact_mcnemar(self):
        out=s.mcnemar([1,1,1,0,1,None],[0,0,0,1,1,1])
        self.assertEqual(out['left_only'],3)
        self.assertEqual(out['right_only'],1)
        self.assertEqual(out['p_two_sided'],.625)
        self.assertEqual(out['n_undefined'],1)
        self.assertEqual(s.mcnemar([1,0],[1,0])['p_two_sided'],1)
        with self.assertRaises(ValueError):s.mcnemar([.5],[0])

if __name__=='__main__':unittest.main()
