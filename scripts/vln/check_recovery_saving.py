"""Four-rank ZeRO2 interrupted-versus-uninterrupted recovery equivalence."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import PretrainedConfig, TrainerCallback, set_seed
from qwen_vl.train.argument import TrainingArguments
from qwen_vl.train.recovery import RecoverySaves, latest_checkpoint, validate_complete
from scripts.vln.check_distributed_resume import CheckedTrainer, Serializer
from scripts.vln.check_epoch_saving import build_dataset
from scripts.vln.check_distributed_loss import Toy


class Record(TrainerCallback):
    def __init__(self, trainer, phase, root):
        self.trainer, self.phase, self.root = trainer, phase, root

    def on_step_end(self, args, state, control, **kwargs):
        examples = [None] * dist.get_world_size()
        dist.all_gather_object(examples, self.trainer.pending)
        signatures = [float(e['x'][0,0]) for rank in examples for e in rank]
        record = dict(weight=self.trainer.model.weight.detach().cpu(),
                      signatures=signatures, lr=self.trainer.lr_scheduler.get_last_lr())
        if self.phase == 'resume':
            expected = torch.load(self.root/'baseline'/f'reference-{state.global_step}.pt',weights_only=True)
            torch.testing.assert_close(record['weight'], expected['weight'],rtol=0,atol=0)
            assert record['signatures'] == expected['signatures'], (record['signatures'],expected['signatures'])
            assert record['lr'] == expected['lr']
        if args.process_index == 0:
            torch.save(record,Path(args.output_dir)/f'reference-{state.global_step}.pt')
        self.trainer.pending = []
        if self.phase == 'interrupted' and state.global_step == 3:
            control.should_training_stop = True
        return control


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--phase',choices=['baseline','interrupted','resume'],required=True)
    cli=parser.parse_args()
    root=Path(cli.root)
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    set_seed(429)
    args=TrainingArguments(output_dir=str(root/cli.phase),per_device_train_batch_size=1,
        gradient_accumulation_steps=2,num_train_epochs=2,learning_rate=5e-6,
        warmup_steps=1,lr_scheduler_type='cosine_with_min_lr',
        lr_scheduler_kwargs={'min_lr_rate':0.1},remove_unused_columns=False,
        average_tokens_across_devices=True,report_to='none',bf16=False,
        save_strategy='epoch',save_total_limit=None,dataloader_num_workers=0,
        seed=429,deepspeed='deepspeed.json',
        accelerator_config={'even_batches':True,'split_batches':False,'dispatch_batches':False})
    model=Toy(); model.config=PretrainedConfig()
    trainer=CheckedTrainer(model=model,args=args,train_dataset=build_dataset(),
        data_collator=lambda rows:rows[0],serializer=Serializer(),data_contract={'test':'recovery'},
        recovery_save_steps=1,callbacks=[RecoverySaves(1)])
    trainer.pending=[]
    trainer.add_callback(Record(trainer,cli.phase,root))
    checkpoint=None
    if cli.phase=='resume':
        checkpoint=latest_checkpoint(root/'interrupted')
        assert checkpoint.name=='checkpoint-3'
        validate_complete(checkpoint)
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    trainer.accelerator.wait_for_everyone()
    if args.process_index==0:
        saved=latest_checkpoint(root/cli.phase)
        record=validate_complete(saved)
        assert record['step']==(3 if cli.phase=='interrupted' else 4)
        assert len(list(saved.rglob('*optim_states.pt')))==4
        (root/f'{cli.phase}-PASS.json').write_text(json.dumps({'passed':True,'step':record['step']}))
        print('RECOVERY_EQUIVALENCE_PASS',cli.phase,flush=True)
    dist.destroy_process_group()


if __name__=='__main__':
    main()
