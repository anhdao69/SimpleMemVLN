import unittest
import tempfile
import torch
from scripts.diag.recall import native_error, sampled_pairs, lag_group, gaussian_projection, prepare_projection_cache


class RecallTests(unittest.TestCase):
    def test_state_orientation_and_repeated_key_heads(self):
        state=torch.tensor([[[1.,2.],[3.,4.]], [[2.,4.],[6.,8.]]])
        keys=torch.tensor([[[1.,0.]], [[0.,1.]]])
        values=torch.tensor([[[1.,2.],[2.,4.]], [[3.,4.],[6.,8.]]])
        self.assertTrue(torch.equal(native_error(state,keys,values),torch.zeros(2)))
        self.assertTrue(torch.allclose(native_error(torch.zeros_like(state),keys,values),torch.ones(2)))

    def test_pair_sampling_contains_postwrite_baseline(self):
        pairs=sampled_pairs(35)
        self.assertIn((34,2),pairs)
        self.assertIn((34,34),pairs)
        self.assertNotIn((33,1),pairs)
        self.assertTrue(all(0<=s<=t<35 for t,s in pairs))
        self.assertEqual([lag_group(x) for x in [0,1,3,4,7,8,32]], [None,'A','A','B','B','C','C'])

    def test_head_block_projection_matches_dense_and_is_shared(self):
        states=torch.arange(24,dtype=torch.float32).reshape(3,2,2,2)
        generator=torch.Generator().manual_seed(0)
        matrix=torch.cat([torch.randn(4,7,generator=generator) for _ in range(2)])
        expected=states.flatten(1)@matrix
        actual=gaussian_projection(states,dimensions=7)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-5))
        self.assertTrue(torch.equal(actual,gaussian_projection(states,dimensions=7)))

    def test_disk_blocks_equal_uncached_projection_and_reject_wrong_contract(self):
        states=torch.arange(24,dtype=torch.float32).reshape(3,2,2,2)
        with tempfile.TemporaryDirectory() as root:
            path=root+'/projection'
            prepare_projection_cache(path,heads=2,key_dim=2,value_dim=2,dimensions=7)
            self.assertTrue(torch.equal(gaussian_projection(states,dimensions=7),
                                       gaussian_projection(states,dimensions=7,cache=path)))
            with self.assertRaises(ValueError):gaussian_projection(states,dimensions=6,cache=path)


if __name__=='__main__':unittest.main()
