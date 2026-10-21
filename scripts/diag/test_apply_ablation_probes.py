import unittest
import numpy as np
from scripts.diag import apply_ablation_probes as a

class ApplyTests(unittest.TestCase):
    def test_exact_scene_fold_no_training_leak(self):
        folds=[dict(fold=0,test_scenes=['full/scene'],train_scenes=['other']),dict(fold=1,test_scenes=['other'],train_scenes=['full/scene'])]
        self.assertEqual(a.heldout_fold(folds,'full/scene')['fold'],0)
        for scene in ('scene','missing'):
            with self.assertRaises(ValueError):a.heldout_fold(folds,scene)
        with self.assertRaises(ValueError):a.heldout_fold(folds+[folds[0]],'full/scene')
        folds[0]['train_scenes'].append('full/scene')
        with self.assertRaises(ValueError):a.heldout_fold(folds,'full/scene')
    def test_wrong_projection_or_teacher_manifest_rejected(self):
        p=dict(baseline_complete_sha256='c',projection_manifest_sha256='p',weights_sha256='w',manifest_sha256='m')
        b=dict(state_projection=dict(cache_manifest_sha256='p'),weights_sha256='w')
        a.validate_ablation_identity(p,b,dict(manifest_sha256='m'),'c')
        for key in p:
            with self.assertRaises(ValueError):a.validate_ablation_identity(dict(p,**{key:'wrong'}),b,dict(manifest_sha256='m'),'c')
    def test_original_scalers_apply_without_refit(self):
        from unittest.mock import patch
        fitted=dict(x_mean=np.ones(18),x_scale=np.full(18,2.),weights=np.ones((18,10)),y_mean=np.zeros(10),target_shape=(5,2))
        targets=np.ones((4,5,2))
        with patch.object(a.probes,'fit_ridge',side_effect=AssertionError('No refit')):
            result=a.metrics(dict(model=fitted),np.array([2.,3.]),9,0,targets)
        self.assertAlmostEqual(result['top1'],.25)
    def test_features_preserve_odd_anchor_and_original_source(self):
        x=a.feature_vector(np.array([2.,3.]),9)
        self.assertEqual(x.shape,(18,));np.testing.assert_array_equal(x[:2],[2,3])
    def test_aggregate_pairs_before_bootstrap_and_keep_missing(self):
        rows=[dict(model='M',episode_id='e',scene_id='s',contrast='B0-baseline',group='C',feature='state',metric='cosine',value=v) for v in (1.,3.)]
        episodes,summary=a.aggregate(rows,[dict(episode_id='e',scene_id='s'),dict(episode_id='z',scene_id='s')])
        self.assertEqual(episodes[0]['n_pairs'],2)
        self.assertEqual(summary[0]['estimate'],2.)
        self.assertEqual(summary[0]['n_undefined'],1)
    def test_did_is_paired_within_episode(self):
        rows=[]
        for model,group,value in [('G3_M5','C',5),('G3_M5','A',2),('M5_E1','C',3),('M5_E1','A',1)]:
            rows.append(dict(model=model,episode_id='e',scene_id='s',contrast='B0-baseline',group=group,feature='state',metric='cosine',value=value,n_pairs=1))
        result=a.difference_in_differences(rows)
        self.assertEqual(result[0]['value'],1)
if __name__=='__main__':unittest.main()
