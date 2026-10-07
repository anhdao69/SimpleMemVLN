"""Independent dense-mask derivative oracle and pinned FA2 equivalence."""
import sys
from types import SimpleNamespace
import pytest
import torch
from qwen_vl.stream.window_attention import step_flash_attention
from test_window_memory import _cpu_flash


def _cpu_varlen(q, k, v, cuq, cuk, maxq, maxk, **kwargs):
    return torch.cat([
        _cpu_flash(q[cuq[i]:cuq[i+1]][None], k[cuk[i]:cuk[i+1]][None],
                   v[cuk[i]:cuk[i+1]][None], **kwargs)[0]
        for i in range(len(cuq)-1)
    ])


@pytest.mark.parametrize('batch', [2, 4, 16, 64])
def test_batched_matches_independent_dense_gradient_and_bounded_saved_storage(monkeypatch, batch):
    monkeypatch.setitem(sys.modules, 'flash_attn', SimpleNamespace(
        flash_attn_varlen_func=_cpu_varlen, flash_attn_func=_cpu_flash))
    torch.manual_seed(153)
    sizes = [3, 1, 7, 2, 5, 4, 1, 3, 8, 2, 6, 1, 4, 2, 5, 1, 3]
    prefix, spans, cursor = 5, [], 5
    for size in sizes:
        spans.append((cursor, cursor + size)); cursor += size
    xs = [torch.randn(1, heads, cursor, 8, dtype=torch.float64, requires_grad=True)
          for heads in (4, 2, 2)]
    stored = {}
    def save(t):
        s = t.untyped_storage(); stored[s.data_ptr()] = s.nbytes(); return t
    with torch.autograd.graph.saved_tensors_hooks(save, lambda t:t):
        actual = step_flash_attention(None, *xs, None, scaling=8**-.5,
            step_plan=(prefix, spans), window_attention_batch_steps=batch)[0]
    group = torch.tensor([-1]*prefix + [i for i,n in enumerate(sizes) for _ in range(n)])
    pos = torch.arange(cursor)
    mask = (pos[None,:] <= pos[:,None]) & ((group[None,:] == -1) | (group[None,:] >= group[:,None]-7))
    q,k,v=xs
    reference = torch.nn.functional.scaled_dot_product_attention(
        q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=mask).transpose(1,2)
    torch.testing.assert_close(actual,reference,rtol=1e-10,atol=1e-10)
    weight = torch.randn_like(actual)
    ga=torch.autograd.grad((actual*weight).sum(),xs)
    ge=torch.autograd.grad((reference*weight).sum(),xs)
    for a,e in zip(ga,ge): torch.testing.assert_close(a,e,rtol=1e-9,atol=1e-9)
    assert sum(stored.values()) <= sum(t.numel()*t.element_size() for t in xs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Pinned H100 FA2 required')
@pytest.mark.parametrize('batch',[4,16,32])
def test_real_varlen_matches_serial_forward_backward(batch):
    torch.manual_seed(153)
    prefix,spans,cursor=71,[],71
    for i in range(33):
        size=97+i%7*13; spans.append((cursor,cursor+size)); cursor+=size
    xs=[torch.randn(1,h,cursor,128,device='cuda',dtype=torch.bfloat16,requires_grad=True) for h in (16,4,4)]
    weight=torch.randn(1,cursor,16,128,device='cuda',dtype=torch.bfloat16)/100
    def run(b):
        y=step_flash_attention(None,*xs,None,scaling=128**-.5,step_plan=(prefix,spans),window_attention_batch_steps=b)[0]
        g=torch.autograd.grad((y*weight).sum(),xs)
        return y,g
    old,go=run(1); new,gn=run(batch)
    torch.testing.assert_close(new,old,rtol=.02,atol=.015)
    for a,e in zip(gn,go):
        assert float((a-e).float().norm()/e.float().norm()) < .02
        torch.testing.assert_close(a,e,rtol=.05,atol=.001)
