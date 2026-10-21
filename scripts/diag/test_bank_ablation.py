import unittest
import numpy as np
from scripts.diag import bank_ablation as b
from qwen_vl.stream.episodic import LayerSelectionPlan,plan_digest


def plan(layer=3,groups=14):
    chunks=[np.arange(max(0,t-4)*3,dtype=np.int64) for t in range(groups)]
    offsets=np.cumsum([0]+list(map(len,chunks))).astype(np.int64)
    ids=np.concatenate(chunks);sources=ids//3
    return LayerSelectionPlan(layer,4,1200,offsets,ids,sources,np.full(groups,np.nan),plan_digest(layer,4,1200,offsets,ids))

class BankTests(unittest.TestCase):
    def test_b0_all_layers_preserves_recent_visibility_and_digest(self):
        originals={l:plan(l) for l in (3,7,11,15,19,23,27,31)}
        for p in originals.values():
            q=b.remove_bank(p)
            self.assertEqual(q.recent_groups,4)
            self.assertEqual(q.budget_tokens,0)
            self.assertEqual(len(q.token_ids),0)
            q.verify()
        lower=b.global_bank_off(originals)
        self.assertTrue(all(not len(lower[l].token_ids) for l in originals if l<=23))
        self.assertIs(lower[27],originals[27]);self.assertIs(lower[31],originals[31])
        self.assertTrue(len(originals[3].token_ids)>0)

    def test_all_layer_b0_native_key_indices_are_prefix_plus_recent(self):
        from qwen_vl.stream.window_attention import _episodic_groups, _key_index
        prefix=5;spans=tuple((5+6*t,11+6*t) for t in range(14))
        for layer in (3,7,11,15,19,23,27,31):
            off=b.remove_bank(plan(layer))
            groups=_episodic_groups(prefix,spans,4,off)
            for t,group in enumerate(groups[1:]):
                actual,_=_key_index([group],prefix)
                expected=np.concatenate((np.arange(prefix),np.arange(spans[max(0,t-3)][0],spans[t][1])))
                np.testing.assert_array_equal(actual,expected)

    def test_targeted_removal_only_one_query_and_equal_counts(self):
        plans={l:plan(l) for l in (3,7,11,15,19,23,27,31)}
        result=b.match_target_control(plans,12,0,seed=7)
        self.assertTrue(result['available'])
        self.assertNotEqual(result['source'],result['control_source'])
        a=b.apply_removals(plans,12,result['target_ids'])
        c=b.apply_removals(plans,12,result['control_ids'])
        for l in plans:
            for t in range(14):
                if l>23 or t!=12:
                    np.testing.assert_array_equal(a[l].group_ids(t),plans[l].group_ids(t))
                else:
                    self.assertEqual(len(a[l].group_ids(t)),len(c[l].group_ids(t)))
                    self.assertEqual(len(plans[l].group_ids(t))-len(a[l].group_ids(t)),3)
            a[l].verify();c[l].verify()

    def test_unavailable_match_and_deterministic_selection(self):
        plans={3:plan()}
        self.assertFalse(b.match_target_control(plans,8,0)['available'])
        a=b.choose_pairs(plans,seed=17);self.assertEqual(a,b.choose_pairs(plans,seed=17))
        self.assertEqual(len(a['anchors']),4)
        for pair in a['pairs']:
            self.assertTrue(8<=pair['anchor']-pair['source']<=32)

    def test_kl_is_directed_and_normalized(self):
        p=np.log([[.75,.25]]);q=np.log([[.5,.5]])
        out=b.logit_kl(p,q)
        self.assertAlmostEqual(out['mean'],.75*np.log(1.5)+.25*np.log(.5))
        self.assertEqual(b.logit_kl(p,p)['mean'],0)
        with self.assertRaises(ValueError):b.logit_kl(p,np.array([[np.nan,0]]))


class PrefixTests(unittest.TestCase):
    def test_prefix_plan_keeps_ids_sources_and_new_valid_digest(self):
        original=plan(groups=14)
        shortened=b.prefix_plan(original,10)
        self.assertEqual(shortened.groups,10)
        self.assertNotEqual(shortened.source_digest,original.source_digest)
        shortened.verify();original.verify()
        for t in range(10):np.testing.assert_array_equal(shortened.group_ids(t),original.group_ids(t))
        np.testing.assert_array_equal(shortened.token_sources,original.token_sources[:original.group_offsets[10]])
        self.assertFalse(shortened.token_ids.flags.writeable)
        with self.assertRaises(ValueError):b.prefix_plan(original,15)

    def test_prefix_layout_and_visible_keys_preserve_original_positions(self):
        from qwen_vl.data.episodic_metadata import EpisodicLayout
        from qwen_vl.stream.window_attention import _episodic_groups,_key_index
        layout=EpisodicLayout(5,tuple((5+6*t,11+6*t) for t in range(14)),
                              tuple(9+6*t for t in range(14)),tuple(6+6*t for t in range(14)),visual_tokens=2)
        shortened=b.prefix_layout(layout,9)
        self.assertEqual(shortened.groups,10)
        self.assertEqual(shortened.length,layout.group_spans[9][1])
        self.assertEqual(shortened.observation_ends[-1],layout.observation_ends[9])
        self.assertEqual(shortened.score_segments()[-1],(layout.observation_ends[9],layout.group_spans[9][1]))
        chunks=[np.array([6+6*s for s in range(max(0,t-3))],dtype=np.int64) for t in range(14)]
        offsets=np.cumsum([0]+[len(x) for x in chunks]).astype(np.int64);ids=np.concatenate(chunks)
        original=LayerSelectionPlan(3,4,1200,offsets,ids,(ids-6)//6,np.full(14,np.nan),plan_digest(3,4,1200,offsets,ids))
        p=b.prefix_plan(original,10)
        before=_episodic_groups(layout.prefix,layout.group_spans,4,original)
        after=_episodic_groups(shortened.prefix,shortened.group_spans,4,p)
        for old,new in zip(before[:11],after):
            x,_=_key_index([old],layout.prefix);y,_=_key_index([new],shortened.prefix)
            np.testing.assert_array_equal(x,y)
        with self.assertRaises(ValueError):b.prefix_layout(layout,-1)

if __name__=='__main__':unittest.main()
