import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from scripts.diag import fit_probes as f

class FitProbeTests(unittest.TestCase):
    def test_pair_definitions_exclude_zero_and_use_current_anchor(self):
        pairs=f.pair_metadata([dict(episode_id='e',episode_uid='u',scene_id='s',frames=36)],'B')
        self.assertTrue(pairs)
        self.assertTrue(all(p['anchor']%2==0 and 4<=p['lag']<=7 and p['source']==p['anchor']-p['lag'] for p in pairs))
        features=np.arange(36*2,dtype=np.float32).reshape(36,2)
        x=f.pair_features(features,pairs[:2])
        np.testing.assert_array_equal(x[:,:2],features[[p['anchor'] for p in pairs[:2]]])
        self.assertEqual(x.shape[1],18)

    def test_incomplete_cache_blocks_before_any_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'identity.json').write_text('{}')
            (root/'index.jsonl').write_text('')
            (root/'COMPLETE.json').write_text(json.dumps({'episodes':239,'index_sha256':f.sha256(root/'index.jsonl')}))
            panel={'episodes':[{'episode_id':str(i)} for i in range(240)]}
            with self.assertRaisesRegex(ValueError,'Teacher cache incomplete'):
                f.validate_inputs({name:root for name in f.MODELS},panel,root)

    def test_memory_specific_contrasts_are_paired(self):
        rows=[dict(feature='state',episode_id='e',anchor=8,source=1,cosine=.8,top1=.5,mrr=.75),
              dict(feature='control',episode_id='e',anchor=8,source=1,cosine=.3,top1=.25,mrr=.5)]
        out=f.paired_metrics(rows[0],rows[1])
        self.assertAlmostEqual(out['memory_specific_cosine'],.5)
        self.assertEqual(out['memory_specific_top1'],.25)
        wrong=dict(rows[1],source=2)
        with self.assertRaises(ValueError):f.paired_metrics(rows[0],wrong)

    def test_chunked_outer_predictions_have_no_scene_leakage(self):
        rng=np.random.default_rng(4);x=rng.normal(size=(24,3));y=np.tile(x[:,None,:],(1,5,1));scenes=np.repeat([f's{i}' for i in range(6)],4)
        received=[];models=[]
        for model,indices,pred in f.cross_fit_chunks(x,y,scenes,lambdas=(.1,1.),chunk_size=3):
            models.append(model);received.extend(indices.tolist())
            self.assertLessEqual(len(pred),3)
            self.assertTrue(set(scenes[indices])<=set(model['test_scenes']))
            self.assertFalse(set(model['train_scenes'])&set(model['test_scenes']))
            for inner in model['selection']['folds']:
                self.assertFalse(set(inner['train_scenes'])&set(model['test_scenes']))
        self.assertEqual(sorted(received),list(range(24)))

if __name__=='__main__':unittest.main()
