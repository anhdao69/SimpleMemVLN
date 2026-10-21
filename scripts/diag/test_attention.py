"""CPU contracts for D-c causal visibility, GQA, categories and transparency."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from qwen_vl.data.episodic_metadata import EpisodicLayout
path=Path(__file__).with_name('attention.py')
spec=importlib.util.spec_from_file_location('diag_attention',path)
a=importlib.util.module_from_spec(spec) if path.exists() else None
if a is not None: spec.loader.exec_module(a)


def layout(groups=10):
    return EpisodicLayout(2,tuple((2+5*t,7+5*t) for t in range(groups)),
                          tuple(6+5*t for t in range(groups)),tuple(3+5*t for t in range(groups)),visual_tokens=2)

class Plan:
    layer_idx=3;recent_groups=4
    def __init__(self,n,ids): self.groups=n;self.ids=np.array(ids,dtype=np.int64)
    def verify(self): return self
    def group_ids(self,t): return self.ids if t==self.groups-1 else np.array([],dtype=np.int64)

class AttentionTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(a,'attention implementation missing')

    def test_exact_visible_index_and_causal_future_exclusion(self):
        l=layout();p=Plan(l.groups,[3,4,8])
        ids=a.visible_key_ids(l,9,4,p)
        expected=list(range(2))+[3,4,8]+list(range(32,52))
        self.assertEqual(ids.tolist(),expected)
        q=torch.zeros(4,2,3);k=torch.zeros(2,len(ids),3);v=torch.ones(2,len(ids),2)
        probs,out=a.probabilities_and_output(q,k,v,torch.tensor([48,50]),ids,1.)
        self.assertTrue(torch.equal(probs[0,:,ids>48],torch.zeros_like(probs[0,:,ids>48])))
        torch.testing.assert_close(probs.sum(-1),torch.ones(2,4))
        torch.testing.assert_close(out,torch.ones(2,4,2))

    def test_gqa_repeats_consecutive_heads_and_value_orientation(self):
        q=torch.zeros(4,1,2);k=torch.zeros(2,2,2)
        v=torch.tensor([[[1.,2.],[3.,4.]],[[10.,20.],[30.,40.]]])
        probs,out=a.probabilities_and_output(q,k,v,torch.tensor([1]),torch.tensor([0,1]),.5)
        expected=torch.tensor([[[2.,3.],[2.,3.],[20.,30.],[20.,30.]]])
        torch.testing.assert_close(out,expected)
        self.assertEqual(tuple(probs.shape),(1,4,2))

    def test_window8_old_exact_tokens_are_not_bank(self):
        l=layout(); ids=a.visible_key_ids(l,9,None,None)
        cats=a.key_categories(l,9,ids,None,None)
        self.assertEqual(cats.count('window8_exact_lag_4_7'),20)
        self.assertFalse(any(c.startswith('bank_') for c in cats))
        self.assertEqual(cats.count('prefix'),2)
        p=Plan(l.groups,[3,4,18]); ids=a.visible_key_ids(l,9,4,p)
        cats=a.key_categories(l,9,ids,4,p)
        self.assertEqual(cats[2:5],['bank_lag_8_15','bank_lag_8_15','bank_lag_4_7'])

    def test_query_selection_reproducible_no_replacement_per_step(self):
        l=EpisodicLayout(2,((2,306),(306,610)),(304,608),(3,307))
        first=a.sample_queries(l,'episode',20261019)
        again=a.sample_queries(l,'episode',20261019)
        self.assertEqual(first,again)
        self.assertEqual(first[0]['ids'][0],303)
        self.assertEqual(len(first[0]['ids']),33)
        self.assertEqual(len(set(first[0]['ids'][1:])),32)
        self.assertTrue(all(3<=x<303 for x in first[0]['ids'][1:]))

    def test_mass_per_token_uses_causally_visible_counts(self):
        probs=torch.tensor([[[.2,.3,.5,0.]],[[.1,.2,.3,.4]]])
        result=a.category_mass(probs,['prefix','bank_lag_8_15','recent_lag_0','recent_lag_0'],
                               torch.tensor([2,3]),torch.tensor([0,1,2,3]))
        self.assertEqual(result['recent_lag_0']['visible_tokens'],[1,2])
        self.assertEqual(result['recent_lag_0']['mass'],[[.5],[.7000000476837158]])
        self.assertEqual(result['bank_lag_4_7']['mass_per_token'],[None,None])

    def test_wrapper_returns_exact_native_tuple_and_does_not_modify_inputs(self):
        l=EpisodicLayout(2,((2,306),),(304,),(3,))
        q=torch.zeros(1,4,306,2);k=torch.zeros(1,2,306,2);v=torch.ones(1,2,306,3)
        before=(q.clone(),k.clone(),v.clone())
        native=(torch.ones(1,306,4,3),None)
        calls=[];raw=[]
        def original(*args,**kwargs): calls.append(1);return native
        capture=a.AttentionCapture(l,'episode',layers=(3,),raw_sink=lambda layer,step,record: raw.append(record))
        capture.original=original
        result=capture.wrapped(SimpleNamespace(layer_idx=3,scaling=.5),q,k,v,None,
                               step_plan=(l.prefix,l.group_spans),scaling=.5)
        self.assertIs(result,native)
        self.assertEqual(calls,[1])
        for old,new in zip(before,(q,k,v)): self.assertTrue(torch.equal(old,new))
        self.assertTrue(capture.gate_summary()['passed'])
        self.assertEqual(len(raw),1)
        self.assertTrue(raw[0]['before_output_gate'])
        self.assertEqual(tuple(raw[0]['query'].shape),(4,33,2))
        with self.assertRaises(RuntimeError):
            capture.wrapped(SimpleNamespace(layer_idx=3,scaling=.5),q,k,v,None,step_plan=(l.prefix,l.group_spans))

    def test_zero_reference_query_is_descriptive_when_block_norm_defined(self):
        native=torch.tensor([[[0.]],[[100.]]])
        estimated=torch.tensor([[[.1]],[[100.]]])
        result=a.output_gate(estimated,native)
        self.assertTrue(result['passed'])
        self.assertIsNone(result['per_query_relative_error'][0])
        self.assertEqual(result['undefined_query_relative_errors'],1)

    def test_output_gate_rejects_nonfinite_and_detects_failure(self):
        native=torch.ones(2,4,3)
        self.assertTrue(a.output_gate(native,native)['passed'])
        self.assertFalse(a.output_gate(native*1.02,native)['passed'])
        with self.assertRaises(ValueError): a.output_gate(native,torch.full_like(native,float('nan')))
        with self.assertRaises(ValueError): a.output_gate(native,native[:1])

if __name__=='__main__': unittest.main()
