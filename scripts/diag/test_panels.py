"""Selection contracts: counts, stratification, preference and immutable inputs."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).with_name('panels.py')
spec = importlib.util.spec_from_file_location('diag_panels', MODULE)
p = importlib.util.module_from_spec(spec) if MODULE.exists() else None
if p is not None:
    spec.loader.exec_module(p)


def population():
    return [dict(episode_id=str(s*100+i), instruction_id=str(s*100+i),
                 scene=f's{s:02}', scene_id=f'mp3d/s{s:02}/s{s:02}.glb',
                 path_length_m=float(i), gt_action_count=40 if i % 2 else 20,
                 gt_forward_steps=i) for s in range(11) for i in range(90)]


class PanelsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(p, 'panel implementation missing')

    def test_exact_counts_uniqueness_minima_and_subset(self):
        a,b = p.build_panels(population())
        self.assertEqual(len(a['episodes']),240)
        self.assertEqual(len(b['episodes']),600)
        self.assertEqual(len({x['episode_id'] for x in a['episodes']}),240)
        self.assertTrue(all(n >= 8 for n in a['metadata']['scene_counts'].values()))
        ids=b['metadata']['determinism_episode_ids']
        self.assertEqual(len(ids),60)
        self.assertEqual(len(set(ids)),60)
        self.assertTrue(set(ids) <= {x['episode_id'] for x in b['episodes']})

    def test_seed_order_independence_and_no_source_mutation(self):
        rows=population(); old=copy.deepcopy(rows)
        a=p.build_panels(rows)
        self.assertEqual(rows,old)
        self.assertEqual(a,p.build_panels(list(reversed(rows))))
        self.assertNotEqual(a,p.build_panels(rows,seed=12))

    def test_long_preference_preserves_strata(self):
        a,_=p.build_panels(population())
        self.assertTrue(all(x['gt_action_count']>=34 for x in a['episodes']))
        counts=a['metadata']['stratum_counts']
        self.assertTrue(all(6 <= n <= 8 for n in counts.values()))

    def test_constrained_apportionment_ties_and_sparse_capacity(self):
        self.assertEqual(p.apportion({'a':1,'b':100,'c':100},21,minimum=1),{'a':1,'b':10,'c':10})
        self.assertEqual(p.apportion({'b':10,'a':10,'c':10},5),{'a':2,'b':2,'c':1})
        with self.assertRaises(ValueError):p.apportion({'a':2,'b':10},9,minimum=3)

    def test_missing_gt_and_duplicate_ids_fail_closed(self):
        data={'episodes':[{'episode_id':1,'scene_id':'mp3d/s/s.glb','instruction':{}}]}
        with self.assertRaises(ValueError):p.records(data,{})
        gt={'1':{'locations':[[0,0,0],[3,4,0]],'actions':[1,0],'forward_steps':1}}
        row=p.records(data,gt)[0]
        self.assertEqual(row['path_length_m'],5.0)
        self.assertEqual(row['gt_action_count'],2)
        self.assertEqual(row['instruction_id'],'1')
        with self.assertRaises(ValueError):p.records({'episodes':data['episodes']*2},gt)

    def test_file_hash_gate_and_source_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.gz'
            with gzip.open(path,'wt') as f:json.dump({'episodes':[]},f)
            before=path.read_bytes()
            with self.assertRaises(ValueError):p.read_gzip(path,'0'*64)
            self.assertEqual(p.read_gzip(path,p.sha256(path)),{'episodes':[]})
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
