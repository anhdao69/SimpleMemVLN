"""Real NCCL/DDP/GAS check against one global frame-mean reference.

Run with torchrun --standalone --nproc_per_node=4. Tiny tensors only;
the production trainer's count collection and loss reduction are exercised.
"""
import contextlib
import json
import os
from types import SimpleNamespace
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from qwen_vl.models.vggt_g3_heads import NativeStateHead
from qwen_vl.train.trainer import QwenSFTTrainer


class Model(torch.nn.Module):
    navigation_config={'research':{'ramp_updates':771,'lambda_max':.1}}
    def __init__(self):
        super().__init__()
        count=int(os.environ.get('G3_REDUCTION_HEADS','1'))
        self.heads=torch.nn.ModuleList([NativeStateHead(heads=2,key_dim=3,value_dim=5,query_hidden=8,decoder_hidden=12,target_dim=8) for _ in range(count)])
    def forward(self,hidden,target,state,**kwargs):
        values=[]
        for head in self.heads:
            if len(hidden):
                queries=head.queries(hidden,torch.full((len(hidden),),8,device=hidden.device)).reshape(-1,5,2,3)
                reads=torch.einsum('prhkv,prhk->prhv',state,queries)
                values.append(head.loss(reads,target))
            else:values.append(head.empty_loss(hidden))
        aux=sum(values)/len(values)
        return {'loss_sum':aux*0,'aux_loss_sum':aux}


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl');world=dist.get_world_size()
    assert world==4
    def gather(value):
        values=[torch.empty_like(value) for _ in range(world)]
        dist.all_gather(values,value);return torch.stack(values)
    report=[]
    for support in ([[0,1],[2,0],[3,4],[0,2]],[[0,0]]*4,[[2,2]]*4):
        model=Model().cuda();wrapped=DDP(model,device_ids=[rank])
        trainer=object.__new__(QwenSFTTrainer);trainer.model=model
        trainer.state=SimpleNamespace(global_step=771)
        trainer.accelerator=SimpleNamespace(num_processes=world,gather=gather)
        generator=torch.Generator().manual_seed(981)
        all_batches=[]
        for counts in support:
            all_batches.append([dict(hidden=torch.randn(n,2,3,generator=generator).cuda(),
                target=torch.randn(n,5,8,generator=generator).cuda(),
                state=torch.randn(n,5,2,3,5,generator=generator).cuda(),num_actions=1,aux_pair_count=n) for n in counts])
        batches=all_batches[rank]
        actions=trainer._get_num_items_in_batch(batches,torch.device('cuda',rank))
        for i,batch in enumerate(batches):
            with wrapped.no_sync() if i==0 else contextlib.nullcontext():
                trainer.compute_loss(wrapped,batch,num_items_in_batch=actions).backward()
        reference=Model().cuda();total=sum(map(sum,support))
        summed=sum(reference(**b)['aux_loss_sum'] for group in all_batches for b in group)
        (summed*(.1/total if total else 0)).backward()
        errors=[]
        for p,q in zip(model.parameters(),reference.parameters()):
            assert p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=2e-5)
            errors.append(float((p.grad-q.grad).abs().max()))
        report.append(dict(support=support,max_gradient_error=max(errors),global_pairs=total))
        dist.barrier();del wrapped,model,reference
    if rank==0: print('DISTRIBUTED_G3_REDUCTION_PASS',json.dumps(report),flush=True)
    dist.destroy_process_group()


if __name__=='__main__':main()
