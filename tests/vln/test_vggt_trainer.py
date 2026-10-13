from types import SimpleNamespace
import torch
from transformers import TrainingArguments
from qwen_vl.train.trainer import QwenSFTTrainer


def test_real_trainer_global_frame_normalization_and_nav_unchanged():
    trainer=object.__new__(QwenSFTTrainer)
    trainer.accelerator=SimpleNamespace(num_processes=2,gather=lambda n: torch.stack([n,n*2]))
    batches=[{'num_actions':2,'aux_frame_count':1},{'num_actions':3,'aux_frame_count':2}]
    count=trainer._get_num_items_in_batch(batches,torch.device('cpu'))
    assert count==15 and all(b['update_aux_count']==9 for b in batches)
    class Model(torch.nn.Module):
        navigation_config={'research':{'ramp_updates':771,'lambda_max':.2}}
        def forward(self,**kwargs):
            assert 'update_aux_count' not in kwargs
            return {'loss_sum':torch.tensor(6.,requires_grad=True),'aux_loss_sum':torch.tensor(3.,requires_grad=True)}
    model=Model();trainer.model=model;trainer.state=SimpleNamespace(global_step=771)
    loss,out=trainer.compute_loss(model,batches[0],return_outputs=True,num_items_in_batch=count)
    torch.testing.assert_close(loss,torch.tensor(6*2/15+3*2*.2/9))
    assert out['loss_sum']==6 and out['update_aux_count']==9


def test_aux_optimizer_lr_and_unique_parameter_coverage():
    m=torch.nn.Module();m.backbone=torch.nn.Linear(4,4)
    m.research_aux=torch.nn.ModuleDict({'current_projector':torch.nn.Linear(4,4)})
    m.navigation_config={'research':{'head_lr':1e-4}}
    args=TrainingArguments(output_dir='/tmp/g1-test',use_cpu=True,report_to='none',learning_rate=5e-6,weight_decay=.01)
    args.mm_projector_lr=5e-6
    trainer=QwenSFTTrainer(model=m,args=args)
    opt=trainer.create_optimizer();groups={id(p):g for g in opt.param_groups for p in g['params']}
    assert len(groups)==sum(len(g['params']) for g in opt.param_groups)
    for name,p in m.named_parameters():
        assert groups[id(p)]['lr']==(1e-4 if name.startswith('research_aux.') else 5e-6)
        assert groups[id(p)]['weight_decay']==(.01 if p.ndim>1 else 0)


def test_global_pair_count_includes_empty_gas_and_empty_rank():
    trainer=object.__new__(QwenSFTTrainer)
    trainer.accelerator=SimpleNamespace(num_processes=2,gather=lambda n:torch.stack([n,n*0]))
    batches=[{'num_actions':2,'aux_pair_count':0},{'num_actions':3,'aux_pair_count':8}]
    count=trainer._get_num_items_in_batch(batches,torch.device('cpu'))
    assert count==5 and all(b['update_aux_count']==8 for b in batches)
