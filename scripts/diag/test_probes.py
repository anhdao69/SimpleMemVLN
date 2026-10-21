import unittest
import numpy as np
from scripts.diag.probes import scene_folds, fit_ridge, predict, retrieval_metrics, choose_lambda


class ProbeTests(unittest.TestCase):
    def test_scene_folds_disjoint_exhaustive_and_reproducible(self):
        scenes=np.array([f's{x}' for x in range(11) for _ in range(3)])
        folds=scene_folds(scenes,2)
        self.assertEqual(folds,scene_folds(scenes,2))
        self.assertEqual(sorted(sum(folds,[])),sorted(set(scenes)))
        for test in folds:
            train=[s for s in scenes if s not in test]
            inner=scene_folds(train,3)
            self.assertFalse(set(sum(inner,[])) & set(test))

    def test_standardization_uses_training_only_and_handles_constants(self):
        x=np.array([[0.,7.],[1.,7.],[2.,7.],[3.,7.]])
        y=(2*x[:,0]+4)[:,None]
        fitted=fit_ridge(x,y,1e-8)
        self.assertTrue(np.array_equal(fitted['x_mean'],[1.5,7.]))
        self.assertEqual(fitted['x_scale'][1],1.)
        self.assertTrue(np.allclose(predict(fitted,np.array([[100.,7.]])),[[204.]],rtol=1e-7))

    def test_retrieval_all_frames_and_ties(self):
        frames=np.array([[[1.,0.]]*5,[[0.,1.]]*5,[[-1.,0.]]*5])
        self.assertEqual(retrieval_metrics(frames[1],frames,1)['rank'],1.)
        self.assertEqual(retrieval_metrics(frames[1],frames,0)['rank'],2.5)
        tied=np.stack([frames[0],frames[0]])
        result=retrieval_metrics(frames[0],tied,0)
        self.assertEqual(result['top1'],.5)
        self.assertEqual(result['rank'],1.5)

    def test_chunked_cv_preserves_scores(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(24,3));y=np.tile(x[:,None,:],(1,5,1));scenes=np.repeat(['a','b','c','d','e','f'],4)
        small=choose_lambda(x,y,scenes,lambdas=(.1,1.),prediction_chunk_size=2)
        whole=choose_lambda(x,y,scenes,lambdas=(.1,1.),prediction_chunk_size=1000)
        self.assertEqual(small['lambda'],whole['lambda'])
        for key in small['scores']:self.assertAlmostEqual(small['scores'][key],whole['scores'][key],places=12)

    def test_cv_is_scene_disjoint(self):
        rng=np.random.default_rng(1);x=rng.normal(size=(36,3));y=np.tile(x[:,None,:],(1,5,1))
        scenes=np.repeat(['a','b','c','d','e','f'],6)
        result=choose_lambda(x,y,scenes,lambdas=(.01,10.))
        self.assertIn(result['lambda'],(.01,10.))
        for fold in result['folds']:
            self.assertFalse(set(fold['train_scenes'])&set(fold['validation_scenes']))


if __name__=='__main__':unittest.main()
