"""Opt-in integration against the actual pinned frozen Omega, not a mock."""
import os
import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('VLN_TEST_FUTURE_OMEGA') != '1',
                              reason='requires pinned Omega weights and an allocated H100')


def test_real_joint_attention_current_slot_isolation_and_detachment():
    from qwen_vl.research.vggt_future import OnlineFutureOmegaTeacher, future_teacher_identity
    from qwen_vl.research.vggt_omega import pool_teacher_tokens
    root='/mnt/data/vmo-ai-task/anhdh35'
    source=root+'/third_party/vggt-omega'
    weights=root+'/models/vggt-omega/vggt_omega_1b_512.pt'
    spec=dict(teacher_source=source,teacher_weights=weights,teacher_future_offset=1,
              teacher_identity=future_teacher_identity(source,weights,1),teacher_batch_size=2)
    teacher=OnlineFutureOmegaTeacher(spec,'cuda')
    assert not teacher.model.training and not any(p.requires_grad for p in teacher.model.parameters())
    images=torch.rand(2,2,3,448,592,generator=torch.Generator().manual_seed(429))
    target=teacher(images)
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        outputs,start=teacher.model.aggregator(images.cuda())
        expected=pool_teacher_tokens(outputs[23][:,:1],start,(28,37))
    expected=expected.clone()+teacher.position
    torch.testing.assert_close(target,expected,atol=0,rtol=0)
    del outputs,expected
    changed=images.clone();changed[0,1]=1-changed[0,1]
    other=teacher(changed)
    assert (other[0]-target[0]).norm()/target[0].norm()>1e-5
    torch.testing.assert_close(other[1],target[1],atol=0,rtol=0)
    teacher.batch_size=1
    single=teacher(images)
    torch.testing.assert_close(single,target,atol=1e-4,rtol=1e-4)
    assert (single-target).norm()/target.norm()<1e-5
    assert not target.requires_grad and not target.is_inference()
    student=torch.randn_like(target,requires_grad=True)
    (1-torch.nn.functional.cosine_similarity(student,target,dim=-1)).mean().backward()
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert all(p.grad is None for p in teacher.model.parameters())
    assert teacher(images[:0]).shape==(0,300,2048)
    print('REAL_FUTURE_TEACHER_PASS',dict(future_relative_change=float((other[0]-target[0]).norm()/target[0].norm()),
          batch_relative_error=float((single-target).norm()/target.norm()),
          peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30),flush=True)
