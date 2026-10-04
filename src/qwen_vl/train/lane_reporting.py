"""Small detached parameter diagnostics; never retain an episode's autograd graph."""
import json
import math
from pathlib import Path
import torch
from transformers import TrainerCallback


@torch.no_grad()
def lane_parameter_summary(model):
    result = {}
    for name, module in model.named_modules():
        if not name.endswith('.step_lane'):
            continue
        decay = module.A_log.float().exp() * torch.nn.functional.softplus(module.dt_bias.float())
        result[name] = dict(
            gate_context='zero-input reference, not observed writer gates',
            zero_input_decay_half_life_events=(math.log(2) / decay).cpu().tolist(),
            zero_input_beta=module.in_proj_b.bias.float().sigmoid().cpu().tolist(),
            out_proj_norm=float(module.out_proj.weight.float().norm()),
        )
    return result


class LaneParameterCallback(TrainerCallback):
    def __init__(self, output_dir, every=50):
        self.path = Path(output_dir)/'lane_parameters.jsonl'
        self.every = every

    def _write(self, args, state, model):
        if args.process_index == 0:
            with self.path.open('a') as handle:
                handle.write(json.dumps(dict(update=state.global_step,
                                             layers=lane_parameter_summary(model)))+'\n')

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._write(args, state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 1 or state.global_step % self.every == 0:
            self._write(args, state, model)


class ObservedLaneDiagnostics:
    """Scoped inference hooks; at most max_records detached scalar summaries.

    Intended for append-by-append replay. A single completed group has one writer,
    permitting an exact delta-update norm from the initial and final sidecar state.
    Multiwriter offline calls are rejected rather than mislabeled as one write.
    No hooks exist on the normal training/serving path.
    """
    def __init__(self, model, max_records=128):
        if max_records < 1:
            raise ValueError('Diagnostic bound must be positive')
        self.model, self.max_records = model, max_records
        self.records, self._hooks, self._pending = [], [], {}

    def __enter__(self):
        if self.model.training or torch.is_grad_enabled():
            raise ValueError('Observed diagnostics require inference mode')
        for name, native in self.model.named_modules():
            if not hasattr(native,'step_lane'):
                continue
            def lane_hook(module, args, kwargs, result, name=name):
                if len(self.records)>=self.max_records:
                    return
                x,roles=args[:2]
                initial=args[2] if len(args)>2 else kwargs.get('initial_state')
                branch,final=result
                writers=roles.eq(3) if module.spec.clock_mode=='step_end' else roles.ne(0)
                count=int(writers.sum())
                if count>1:
                    raise ValueError('Observed write diagnostics require at most one writer per append')
                row=dict(layer=name,writer_count=count,tokens=x.shape[1])
                if final is not None:
                    row['state_norm_per_head']=final.float().square().sum((-2,-1)).sqrt()[0].cpu().tolist()
                if count:
                    writer=x[writers]
                    a=module.in_proj_a(writer).float()
                    b=module.in_proj_b(writer)
                    g=-module.A_log.float().exp()*torch.nn.functional.softplus(a+module.dt_bias.float())
                    alpha=g.exp()
                    row['writer_alpha']=alpha.cpu().tolist()
                    row['writer_beta']=b.sigmoid().float().cpu().tolist()
                    if initial is not None and final is not None:
                        update=final-initial*alpha[...,None,None]
                        row['write_magnitude_per_head']=update.square().sum((-2,-1)).sqrt()[0].cpu().tolist()
                self._pending[name]=(branch.detach(),row)
            def native_hook(module,args,result,name=name):
                pending=self._pending.pop(name,None)
                if pending is None:
                    return
                branch,row=pending
                combined=result[0] if isinstance(result,tuple) else result
                lane_rms=branch.float().square().mean().sqrt()
                native_rms=(combined.float()-branch.float()).square().mean().sqrt()
                row['lane_output_rms']=float(lane_rms)
                row['native_output_rms']=float(native_rms)
                row['native_rms_estimate']='combined minus lane in FP32, includes BF16 addition rounding'
                row['lane_native_output_rms_ratio']=float(lane_rms/native_rms) if native_rms>0 else None
                if len(self.records)<self.max_records:
                    self.records.append(row)
            self._hooks.append(native.step_lane.register_forward_hook(lane_hook,with_kwargs=True))
            self._hooks.append(native.register_forward_hook(native_hook))
        return self

    def __exit__(self,*exc):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._pending.clear()
