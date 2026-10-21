import unittest
import numpy as np
from scripts.diag import selection as s

class Plan:
    # Small hand-calculated capture: R=1, bank cap 2, four tokens per group.
    recent_groups=1;budget_tokens=2;groups=3;layer_idx=3
    group_offsets=np.array([0,0,2,4]);token_ids=np.array([10,11,11,20])
    token_sources=np.array([0,0,0,1]);cutoff_margins=np.array([np.nan,1.,1.])
    source_digest='fixture'
    def group_ids(self,t):return self.token_ids[self.group_offsets[t]:self.group_offsets[t+1]]
    def verify(self):return self

class SelectionTests(unittest.TestCase):
    def test_candidate_admission_and_native_summary(self):
        ids=np.array([[10,11,12,13],[20,21,22,23],[30,31,32,33]])
        scores=np.array([[3,4,2,1],[5,1,1,1],[9,8,7,6]],dtype=np.float32)
        out=s.selection_metrics(Plan(),ids,scores,grid_shape=(2,2),border_columns=(0,))
        self.assertEqual(out['steps'][1]['candidate_count'],4)
        self.assertEqual(out['steps'][2]['candidate_count'],6)
        self.assertAlmostEqual(out['steps'][2]['candidate_cv'],np.std([3,4,5,1,1,1])/2.5)
        self.assertAlmostEqual(out['steps'][2]['normalized_cutoff_margin'],1/np.std([3,4,5,1,1,1]))
        self.assertAlmostEqual(out['steps'][2]['selected_gini'],1/18)
        self.assertEqual(out['steps'][2]['turnover'],1)
        self.assertEqual(out['summary']['evicted_total'],1)
        self.assertEqual(out['summary']['mean_age_groups'],1.5)
        self.assertEqual(out['steps'][2]['max_source_share'],.5)
        self.assertEqual(out['steps'][2]['spatial_histogram'],[[1,1],[0,0]])
        self.assertEqual(out['steps'][2]['border_share'],.5)
        self.assertIsNone(out['steps'][0]['candidate_cv'])

    def test_degenerate_metrics(self):
        self.assertIsNone(s.gini([]))
        self.assertIsNone(s.gini([0,0]))
        self.assertEqual(s.gini([1,1]),0)
        self.assertEqual(s.jaccard([],[]),1)
        self.assertEqual(s.jaccard([1,2],[2,3]),1/3)
        self.assertIsNone(s.spearman([1,1],[2,3])['value'])
        self.assertAlmostEqual(s.spearman([1,1,3],[3,3,1])['value'],-1)
        self.assertEqual(s.spearman([1,None,3],[3,2,1])['n_undefined'],1)
        with self.assertRaises(ValueError):s.gini([-1,2])
        with self.assertRaises(ValueError):s.spearman([float('inf')],[1])

    def test_zero_spread_and_cross_model_alignment(self):
        ids=np.array([[10,11,12,13],[20,21,22,23],[30,31,32,33]])
        scores=np.zeros((3,4),dtype=np.float32)
        out=s.selection_metrics(Plan(),ids,scores,grid_shape=(2,2))
        self.assertIsNone(out['steps'][2]['candidate_cv'])
        self.assertIsNone(out['steps'][2]['normalized_cutoff_margin'])
        self.assertEqual(out['steps'][2]['undefined_reasons']['normalized_cutoff_margin'],'zero_candidate_std')
        self.assertEqual(out['episode_means']['selected_gini']['n_defined'],0)
        scores=np.arange(12).reshape(3,4)
        cross=s.compare_models(Plan(),Plan(),ids,scores,-scores)
        self.assertTrue(all(r['bank_jaccard']==1 for r in cross))
        self.assertTrue(all(abs(r['write_score_spearman']['value']+1)<1e-12 for r in cross))
        self.assertEqual(cross[0]['left_bank_size'],0)
        with self.assertRaises(ValueError):s.compare_models(Plan(),Plan(),ids,scores,scores[:2])

    def test_invalid_selected_id_and_nonfinite_scores_rejected(self):
        ids=np.arange(12).reshape(3,4);scores=np.ones((3,4))
        with self.assertRaises(ValueError):s.selection_metrics(Plan(),ids,scores,grid_shape=(2,2))
        scores[0,0]=np.nan
        with self.assertRaises(ValueError):s.selection_metrics(Plan(),ids,scores,grid_shape=(2,2))

if __name__=='__main__':unittest.main()
