import torch
from test_episodic_forward import tiny, _episode
from qwen_vl.models.research_forward import research_text_forward
from qwen_vl.models.vggt_aux_heads import CurrentSpatialProjector


def test_native_off_output_and_gradients(tiny):
    layout,x,pos=_episode(2)
    tiny.train(); x.requires_grad_()
    step=(layout.prefix,layout.group_spans)
    native=tiny(inputs_embeds=x,position_ids=pos,use_cache=False,step_plan=step,window_attention_batch_steps=4).last_hidden_state
    actual,aux=research_text_forward(tiny,x,pos,step,window_attention_batch_steps=4)
    assert torch.equal(native,actual) and aux==0
    a,=torch.autograd.grad(native.square().sum(),x)
    b,=torch.autograd.grad(actual.square().sum(),x)
    torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_checkpoint_tap_matches_full_decoder_output_and_has_no_future_gradient(tiny):
    layout,x,pos=_episode(2); tiny.train()
    step=(layout.prefix,layout.group_spans)
    rows=torch.arange(layout.visual_starts[0],layout.visual_starts[0]+300)[None]
    target=torch.randn(1,300,4)
    head=CurrentSpatialProjector(48,24,4)
    observed=[]
    hook=tiny.layers[1].register_forward_hook(lambda m,args,out: observed.append(out))
    tiny(inputs_embeds=x,position_ids=pos,use_cache=False,step_plan=step)
    hook.remove(); expected=head(observed[0][0,rows],target).detach(); observed.clear()
    results=[]
    for checkpoint in (False,True):
        tiny.zero_grad(); head.zero_grad(); leaf=x.clone().requires_grad_()
        if checkpoint: tiny.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        else: tiny.gradient_checkpointing_disable()
        _,aux=research_text_forward(tiny,leaf,pos,step,projector=head,visual_positions=rows,targets=target,alignment_layer=1)
        aux.backward()
        torch.testing.assert_close(aux,expected)
        assert leaf.grad[:,:rows.max()+1].abs().sum()>0
        assert leaf.grad[:,rows.max()+1:].count_nonzero()==0
        results.append((aux.detach(),leaf.grad.clone(),head.fc1.weight.grad.clone()))
    for a,b in zip(*results): torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
