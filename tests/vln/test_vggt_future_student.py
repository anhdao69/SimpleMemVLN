"""Future targets alter gradients, never the causal student's forward values."""
import pytest
import torch
from test_episodic_forward import tiny, _episode
from qwen_vl.models.research_forward import research_text_forward
from qwen_vl.models.vggt_aux_heads import CurrentSpatialProjector


@pytest.mark.parametrize('checkpoint',[False,True])
def test_future_target_and_future_student_observation_cannot_change_current_output(tiny,checkpoint):
    layout,x,pos=_episode(2);tiny.train()
    if checkpoint:tiny.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    rows=torch.arange(layout.visual_starts[0],layout.visual_starts[0]+300)[None]
    head=CurrentSpatialProjector(48,24,4)
    target=torch.randn(1,300,4,requires_grad=True)
    step=(layout.prefix,layout.group_spans)
    baseline,_=research_text_forward(tiny,x,pos,step)
    gradients=[]
    for goal in (target,-target):
        leaf=x.clone().requires_grad_()
        out,loss=research_text_forward(tiny,leaf,pos,step,projector=head,visual_positions=rows,
                                       targets=goal,alignment_layer=1)
        torch.testing.assert_close(out,baseline,atol=0,rtol=0)
        grad,=torch.autograd.grad(loss,leaf)
        assert grad[:,:rows.max()+1].abs().sum()>0
        assert grad[:,rows.max()+1:].count_nonzero()==0
        gradients.append(grad)
    assert not torch.equal(*gradients) and target.grad is None
    changed=x.clone();changed[:,layout.group_spans[1][0]:]+=10
    out,_=research_text_forward(tiny,changed,pos,step)
    torch.testing.assert_close(out[:,:layout.group_spans[0][1]],baseline[:,:layout.group_spans[0][1]],atol=0,rtol=0)
