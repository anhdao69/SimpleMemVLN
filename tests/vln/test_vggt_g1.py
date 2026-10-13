import pytest
import torch
from qwen_vl.models.vggt_aux_heads import CurrentSpatialProjector
from qwen_vl.research.vggt_g1 import auxiliary_weight, normalized_aux_loss, serving_state_dict, serving_config

def test_projector_exact_size_rng_and_gradients():
    torch.manual_seed(7); state=torch.random.get_rng_state().clone()
    head=CurrentSpatialProjector()
    assert torch.equal(state,torch.random.get_rng_state())
    assert sum(p.numel() for p in head.parameters())==18885632
    hidden=torch.randn(2,300,2560,requires_grad=True)
    target=torch.randn(2,300,2048,requires_grad=True)
    loss=head(hidden,target)
    loss.backward()
    assert torch.isfinite(loss) and hidden.grad.abs().sum()>0
    assert target.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert torch.allclose(loss.detach(),head(hidden[:1],target[:1])+head(hidden[1:],target[1:]),atol=1e-6)

def test_empty_and_zero_prediction_participation():
    head=CurrentSpatialProjector(hidden_size=8,intermediate_size=12,target_size=4)
    empty=head(torch.empty(0,300,8),torch.empty(0,300,4)); empty.backward()
    assert empty==0 and all(p.grad is not None for p in head.parameters())
    for p in head.parameters(): p.data.zero_()
    hidden=torch.randn(1,300,8,requires_grad=True)
    loss=head(hidden,torch.zeros(1,300,4)); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(hidden.grad).all()

def test_global_unequal_support_matches_reference():
    a=torch.tensor(2.,requires_grad=True); b=torch.tensor(7.,requires_grad=True)
    # rank0 has1 frame, rank1 has3; average two distributed rank gradients.
    actual=(normalized_aux_loss(a,4,2,.2)+normalized_aux_loss(b,4,2,.2))/2
    expected=.2*(a+b)/4
    torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-7)
    assert auxiliary_weight(0,771,.2)==0
    assert auxiliary_weight(771,771,.2)==.2
    assert normalized_aux_loss(a,0,2,.2)==0

def test_export_removes_only_auxiliary_namespace():
    state={'backbone.weight':torch.ones(1),'research_aux.current_projector.weight':torch.ones(1),'backbone.step_lane.weight':torch.ones(1)}
    result=serving_state_dict(state)
    assert set(result)=={'backbone.weight','backbone.step_lane.weight'}
    cfg={'research':{'g1':True},'training':{'seed':429},'model':{'output_mode':'qwen_text'}}
    assert 'research' not in serving_config(cfg) and 'research' in cfg


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA RNG isolation')
def test_projector_construction_does_not_reset_cuda_rng():
    torch.cuda.manual_seed_all(983)
    before=torch.cuda.get_rng_state().clone()
    CurrentSpatialProjector()
    assert torch.equal(before,torch.cuda.get_rng_state())
