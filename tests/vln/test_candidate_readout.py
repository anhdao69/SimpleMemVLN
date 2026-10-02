import math
from types import SimpleNamespace
import pytest
import torch
from qwen_vl.models.nav_model import SimpleMemVLNForNavigation


def tiny(mode='lm_rows_trainable', bias=False):
    torch.manual_seed(8)
    b=torch.nn.Module(); b.config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
    b.model=torch.nn.Module(); b.model.visual=torch.nn.Linear(2,2)
    b.model.language_model=torch.nn.Module()
    b.model.language_model.embed_tokens=torch.nn.Embedding(40,8)
    b.lm_head=torch.nn.Linear(8,40,bias=bias)
    b.lm_head.weight=b.model.language_model.embed_tokens.weight
    cfg=dict(model=dict(output_mode='candidate_logits',action_head_mode=mode),training=dict(class_weighting='none'))
    return SimpleMemVLNForNavigation(b,cfg,candidate_token_ids=[32,33,34,35])


@pytest.mark.parametrize('mode',['lm_rows_trainable','lm_rows_frozen','copied_linear'])
@pytest.mark.parametrize('bias',[False,True])
def test_selected_projection_matches_full_and_never_calls_head(mode,bias):
    m=tiny(mode,bias); h=torch.randn(3,8,requires_grad=True)
    expected=m.backbone.lm_head(h)[:,[32,33,34,35]]
    def forbidden(*args):
        raise AssertionError('Full vocabulary projection called')
    m.backbone.lm_head.forward=forbidden
    actual=m.action_logits(h)
    torch.testing.assert_close(actual,expected)
    actual.square().sum().backward()
    assert h.grad.abs().sum()>0
    assert not m.backbone.model.visual.weight.requires_grad
    shared=m.backbone.lm_head.weight
    assert shared is m.backbone.model.language_model.embed_tokens.weight
    if mode=='lm_rows_frozen':
        assert not shared.requires_grad and shared.grad is None
    elif mode=='lm_rows_trainable':
        assert shared.grad[32:36].abs().sum()>0
    else:
        assert m.classifier.weight.grad.abs().sum()>0


def test_candidate_forward_uniform_ce_and_targets():
    m=tiny(); m.backbone.lm_head.weight.data.zero_()
    h=torch.randn(1,6,8,requires_grad=True)
    m.hidden=lambda *a,**k:h
    output=m(input_ids=torch.zeros(1,6,dtype=torch.long),mm_token_type_ids=None,
             pixel_values=None,image_grid_thw=None,step_plan=None,
             action_class_ids=torch.tensor([0,1,2,3]),read_positions=torch.tensor([0,1,3,5]),
             response_target_positions=torch.empty(0,dtype=torch.long),
             response_action_index=torch.empty(0,dtype=torch.long),num_actions=4)
    assert output['logits'].shape==(4,4)
    torch.testing.assert_close(output['loss_sum']/4,torch.tensor(math.log(4)))
    assert output['predictions'].tolist()==[0,0,0,0]


def test_candidate_checkpoint_preserves_logits_and_tying(tmp_path):
    m=tiny(); h=torch.randn(2,8); path=tmp_path/'weights.pt'
    torch.save(m.state_dict(),path)
    n=tiny(); n.load_state_dict(torch.load(path,weights_only=True),strict=True)
    torch.testing.assert_close(m.action_logits(h),n.action_logits(h),rtol=0,atol=0)
    assert n.backbone.lm_head.weight is n.backbone.model.language_model.embed_tokens.weight
