"""Globally aggregated per-update diagnostics, including rare STOP behavior."""
import json
from pathlib import Path
import torch
import torch.distributed as dist
from transformers import TrainerCallback
from qwen_vl.eval.action_metrics import classification_metrics


class ActionMetricsCallback(TrainerCallback):
    def __init__(self,out):
        self.out=Path(out)
        self.counts=None
        self.losses=None

    def add(self,truth,prediction,weighted,unweighted):
        counts=torch.bincount(truth.detach()*4+prediction.detach(),minlength=16)
        losses=torch.stack([weighted.detach().double(),unweighted.detach().double()])
        self.counts=counts if self.counts is None else self.counts+counts
        self.losses=losses if self.losses is None else self.losses+losses

    def on_step_end(self,args,state,control,**kwargs):
        if self.counts is None:
            return
        if dist.is_initialized():
            dist.all_reduce(self.counts)
            dist.all_reduce(self.losses)
        if args.process_index==0:
            report=classification_metrics(self.counts.reshape(4,4).cpu().tolist())
            total=int(self.counts.sum())
            report.update(update=state.global_step,weighted_ce=float(self.losses[0])/total,
                          unweighted_ce=float(self.losses[1])/total)
            with (self.out/'action_metrics.jsonl').open('a') as f:
                f.write(json.dumps(report)+'\n')
            print('ACTION_METRICS',json.dumps(report),flush=True)
        self.counts=self.losses=None
