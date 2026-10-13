import copy
import pytest
import torch
from transformers import TrainingArguments
from qwen_vl.train.trainer import QwenSFTTrainer


def make_trainer():
    model=torch.nn.Module();model.backbone=torch.nn.Linear(4,4)
    model.research_aux=torch.nn.ModuleDict({'head':torch.nn.Linear(4,3)})
    model.navigation_config={'research':{'head_lr':.03},
        'training':{'cpu_optimizer_impl':'deepspeed_cpu_adam'}}
    args=TrainingArguments(output_dir='/tmp/g1-cpu-adam-test',use_cpu=True,report_to='none',
        learning_rate=.005,weight_decay=.01,adam_beta1=.9,adam_beta2=.95)
    args.mm_projector_lr=.005
    return QwenSFTTrainer(model=model,args=args)


def test_vectorized_cpu_adamw_matches_torch_groups_and_updates():
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    trainer=make_trainer()
    trainer.args.deepspeed={'zero_optimization':{'stage':2,'offload_optimizer':{'device':'cpu'}}}
    optimized=trainer.create_optimizer();assert isinstance(optimized,DeepSpeedCPUAdam)
    # Mirror the actual parameter groups, including independent head LR and
    # no-decay biases, then compare multiple FP32 AdamW transitions.
    reference=torch.optim.AdamW(copy.deepcopy(optimized.param_groups),lr=.005,betas=(.9,.95),eps=1e-8)
    generator=torch.Generator().manual_seed(429)
    for step in range(5):
        for lhs,rhs in zip(optimized.param_groups,reference.param_groups):
            assert lhs['lr']==rhs['lr'] and lhs['weight_decay']==rhs['weight_decay']
            assert lhs['betas']==(.9,.95) and lhs['eps']==1e-8 and lhs['bias_correction'] is True
            for p,q in zip(lhs['params'],rhs['params']):
                p.grad=torch.randn(p.shape,generator=generator);q.grad=p.grad.clone()
        optimized.step();reference.step()
        for lhs,rhs in zip(optimized.param_groups,reference.param_groups):
            for p,q in zip(lhs['params'],rhs['params']):
                torch.testing.assert_close(p,q,atol=2e-7,rtol=2e-6)
                for key in ('exp_avg','exp_avg_sq'):
                    assert optimized.state[p][key].dtype==torch.float32
                    torch.testing.assert_close(optimized.state[p][key],reference.state[q][key],atol=2e-7,rtol=2e-6)
                assert optimized.state[p]['step']==step+1
    import io
    buffer=io.BytesIO();torch.save(optimized.state_dict(),buffer);buffer.seek(0)
    restored=DeepSpeedCPUAdam(copy.deepcopy(optimized.param_groups),adamw_mode=True)
    restored.load_state_dict(torch.load(buffer,weights_only=True))
    for lhs,rhs in zip(optimized.param_groups,restored.param_groups):
        for p,q in zip(lhs['params'],rhs['params']):
            p.grad=torch.randn(p.shape,generator=generator);q.grad=p.grad.clone()
    optimized.step();restored.step()
    for lhs,rhs in zip(optimized.param_groups,restored.param_groups):
        for p,q in zip(lhs['params'],rhs['params']):
            torch.testing.assert_close(p,q,atol=0,rtol=0)
            assert optimized.state[p]['step']==restored.state[q]['step']==6


def test_cpu_adam_refuses_missing_offload_contract():
    trainer=make_trainer();trainer.args.deepspeed={'zero_optimization':{'stage':2}}
    with pytest.raises(ValueError,match='explicit optimizer CPU offload'):trainer.create_optimizer()
