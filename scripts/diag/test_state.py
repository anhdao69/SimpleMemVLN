"""Synthetic CPU verification of diagnostic state orientation and segmentation."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
path=Path(__file__).with_name('state.py')
spec=importlib.util.spec_from_file_location('diag_state',path)
s=importlib.util.module_from_spec(spec) if path.exists() else None
if s is not None: spec.loader.exec_module(s)


def layer():
    torch.manual_seed(10)
    m=nn.Module()
    m.conv_kernel_size=3; m.num_k_heads=2; m.num_v_heads=4
    m.head_k_dim=3; m.head_v_dim=2; m.key_dim=6; m.value_dim=8; m.conv_dim=20
    m.in_proj_qkv=nn.Linear(5,20,bias=False)
    m.in_proj_b=nn.Linear(5,4,bias=False); m.in_proj_a=nn.Linear(5,4,bias=False)
    m.conv1d=nn.Conv1d(20,20,3,groups=20,bias=True)
    m.A_log=nn.Parameter(torch.zeros(4)); m.dt_bias=nn.Parameter(torch.zeros(4))
    d=nn.Module(); d.layer_type='linear_attention'; d.linear_attn=m; d.input_layernorm=nn.Identity()
    return d


def sequential(k,v,g,beta,state):
    result=[]; state=state.double().clone()
    k=k.double().repeat_interleave(v.shape[1]//k.shape[1],dim=1)
    for i in range(len(k)):
        state=state*g[i].double().exp()[:,None,None]
        delta=beta[i].double()[:,None]*(v[i].double()-torch.einsum('hkv,hk->hv',state,k[i]))
        state=state+torch.einsum('hk,hv->hkv',k[i],delta)
        result.append(state.clone())
    return torch.stack(result)

class StateTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(s,'state implementation missing')

    def test_key_value_orientation_repeat_and_nonzero_carry(self):
        torch.manual_seed(9)
        k=torch.nn.functional.normalize(torch.randn(137,2,3),dim=-1)
        v=torch.randn(137,4,2); g=-torch.rand(137,4)*.2; beta=torch.rand(137,4)
        initial=torch.randn(4,3,2); initial_before=initial.clone()
        expected=sequential(k,v,g,beta,initial)
        result=s.scan_prepared((k,v,g,beta),[(0,3),(3,72),(72,137)],initial)
        self.assertEqual(tuple(result.shape),(4,3,2))
        torch.testing.assert_close(result.double(),expected[-1],atol=2e-5,rtol=2e-5)
        self.assertTrue(torch.equal(initial,initial_before))

    def test_conv_tail_carries_across_segments(self):
        from qwen_vl.models.committed_write import prepare_gdn_inputs, empty_conv_tail
        d=layer(); hidden=torch.randn(73,5)
        whole=prepare_gdn_inputs(d,hidden,empty_conv_tail(d,'cpu',torch.float32))
        pieces=[]; tail=empty_conv_tail(d,'cpu',torch.float32)
        for start,end in [(0,1),(1,7),(7,73)]:
            *prepared,tail=prepare_gdn_inputs(d,hidden[start:end],tail); pieces.append(prepared)
        for i in range(4): torch.testing.assert_close(torch.cat([p[i] for p in pieces]),whole[i],rtol=2e-6,atol=2e-6)
        torch.testing.assert_close(tail,whole[4])

    def test_snapshots_are_after_observation_before_response(self):
        from qwen_vl.data.episodic_metadata import EpisodicLayout
        from qwen_vl.models.committed_write import prepare_gdn_inputs, empty_conv_tail
        d=layer(); hidden=torch.randn(19,5)
        layout=EpisodicLayout(3,((3,10),(10,19)),(8,16),(4,11),visual_tokens=2)
        result=s.reconstruct(d,hidden,layout,visual_dtype=torch.float32)
        k,v,g,beta,_=prepare_gdn_inputs(d,hidden,empty_conv_tail(d,'cpu',torch.float32))
        expected=sequential(k,v,g,beta,torch.zeros(4,3,2))
        torch.testing.assert_close(result['states'].double(),expected[[7,15]],atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(result['visual_keys'],k[layout.visual_matrix()])
        torch.testing.assert_close(result['visual_values'],v[layout.visual_matrix()])
        torch.testing.assert_close(result['visual_beta'],beta[layout.visual_matrix()])
        self.assertEqual(result['anchor_token_ids'],[7,15])

    def test_window8_layout_preserves_original_positions(self):
        ids=torch.zeros(1,610,dtype=torch.long); ids[0,3:303]=99; ids[0,307:607]=99
        data=dict(input_ids=ids,step_plan=(2,((2,306),(306,610))),
                  response_target_positions=torch.tensor([304,305,608,609]),
                  response_action_index=torch.tensor([0,0,1,1]),num_actions=2)
        result=s.layout_from_encoded(data,99)
        self.assertEqual(result.score_segments(),((0,2),(2,304),(304,306),(306,608),(608,610)))
        self.assertEqual(result.visual_starts,(3,307))

    def test_capture_is_transparent_and_removed(self):
        d=nn.Linear(3,3); layers=nn.ModuleList([d]); hidden=torch.randn(1,7,3)
        before=d(hidden)
        with s.LayerInputCapture(layers,indices=(0,)) as captured:
            after=d(hidden)
            self.assertTrue(torch.equal(before,after))
            self.assertTrue(torch.equal(captured.hidden[0],hidden[0]))
            hidden.add_(1)
            self.assertFalse(torch.equal(captured.hidden[0],hidden[0]))
        self.assertEqual(len(d._forward_pre_hooks),0)

    def test_native_fla_gate_rejects_fallback(self):
        d=layer()
        def fake(*args,**kwargs): return None,None
        d.linear_attn.chunk_gated_delta_rule=fake
        prepared=(torch.zeros(2,2,3),torch.zeros(2,4,2),torch.zeros(2,4),torch.ones(2,4))
        with self.assertRaises(RuntimeError): s.native_fla_state(d,prepared)

    def test_invalid_segments_and_inputs_rejected(self):
        prepared=(torch.zeros(3,2,3),torch.zeros(3,4,2),torch.zeros(3,4),torch.ones(3,4))
        with self.assertRaises(ValueError): s.scan_prepared(prepared,[(0,1),(2,3)])
        with self.assertRaises(ValueError): s.scan_prepared(prepared,[(0,3)],torch.zeros(4,2,3))
        bad=list(prepared); bad[2]=torch.full((3,4),float('nan'))
        with self.assertRaises(ValueError): s.scan_prepared(bad,[(0,3)])

    def test_capture_cleans_hooks_after_exception(self):
        d=nn.Linear(3,3)
        with self.assertRaises(RuntimeError):
            with s.LayerInputCapture([d],indices=(0,)):
                d(torch.zeros(1,2,3)); d(torch.zeros(1,2,3))
        self.assertEqual(len(d._forward_pre_hooks),0)

    def test_relative_error_zero_nonfinite_and_shape(self):
        zero=torch.zeros(2,3,4)
        self.assertEqual(s.relative_state_error(zero,zero),0.)
        with self.assertRaises(ValueError): s.relative_state_error(torch.ones_like(zero),zero)
        with self.assertRaises(ValueError): s.relative_state_error(zero,torch.zeros(2,4,3))
        with self.assertRaises(ValueError): s.relative_state_error(zero,torch.full_like(zero,float('nan')))

if __name__=='__main__': unittest.main()
