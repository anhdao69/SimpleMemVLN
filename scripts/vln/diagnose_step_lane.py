"""Fixed-history inference interventions and bounded lane-state diagnostics.

These are teacher-forced observation replays, not closed-loop SR/SPL scores.
Native GDN/KV can retain prior lane influence after an intervention.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import MethodType
import torch
from qwen_vl.train.vln_runtime import load_checkpoint
from qwen_vl.stream.session import StreamSession
from qwen_vl.train.lane_reporting import lane_parameter_summary, ObservedLaneDiagnostics
from PIL import Image


@contextmanager
def lane_output_disabled(model):
    if model.training or torch.is_grad_enabled():
        raise ValueError('Output intervention is inference-only')
    saved=[]
    def zero_output(module, *args, **kwargs):
        output,state=module._diagnostic_original_forward(*args,**kwargs)
        return torch.zeros_like(output),state
    try:
        for module in model.modules():
            if hasattr(module,'step_lane'):
                lane=module.step_lane
                saved.append((lane,lane.forward))
                object.__setattr__(lane,'_diagnostic_original_forward',lane.forward)
                lane.forward=MethodType(zero_output,lane)
        yield
    finally:
        for lane,forward in saved:
            lane.forward=forward
            del lane._diagnostic_original_forward


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--model-path',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--episode-uid',required=True)
    p.add_argument('--steps',type=int,default=17)
    p.add_argument('--freeze-after',type=int,default=8)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    model,serializer=load_checkpoint(args.checkpoint,args.model_path)
    model.eval()
    if not getattr(model,'step_lane_spec',None):
        raise ValueError('Checkpoint does not contain step lanes')
    if model.step_lane_spec.clock_mode!='step_end':
        raise ValueError('Observed single-writer replay diagnostics require step_end clock')
    if serializer.mode!='candidate_logits':
        raise ValueError('This four-score replay diagnostic requires candidate_logits')
    with Path(args.manifest).open() as handle:
        episode=next(json.loads(line) for line in handle if json.loads(line)['episode_uid']==args.episode_uid)
    from contextlib import nullcontext
    records={}
    with torch.inference_mode():
        for intervention in ('normal','output_disabled','reset_each_group','freeze_after_boundary'):
            s=StreamSession(model,serializer)
            rows=[]
            context=lane_output_disabled(model) if intervention=='output_disabled' else nullcontext()
            with context, ObservedLaneDiagnostics(model,max_records=256) as observed:
                s.reset(episode['episode_uid'],episode['instruction'])
                for t,step in enumerate(episode['steps'][:args.steps]):
                    with Image.open(step['rgb_path']) as rgb:
                        block=serializer.encode_observation(rgb,t)
                    s._begin_step(t,block)
                    start=s.positions.logical_token_count
                    freeze = intervention=='freeze_after_boundary' and t>=args.freeze_after
                    h=s._append_with_role(block,0 if freeze else 1)
                    logits=model.action_logits(h[0,-1]).float()
                    # Hold feedback history fixed across interventions.
                    from qwen_vl.contracts import ACTIONS
                    feedback=serializer.feedback_ids(ACTIONS.index(step['action_name']))
                    s._append_with_role(serializer.text_block(feedback),0 if freeze else 2,complete=not freeze)
                    if s.cfg['memory']['mode']=='window8':
                        s.resident.append((t,s.positions.logical_token_count-start))
                    states=s.step_lane_cache._states
                    norms={str(k):v.float().norm().item() for k,v in states.items()}
                    rows.append(dict(step=t,logits=logits.cpu().tolist(),state_norms=norms,
                                     logical_tokens=s.positions.logical_token_count,
                                     lane_state_bytes=sum(v.numel()*v.element_size() for v in states.values())))
                    if intervention=='reset_each_group':
                        for v in states.values():
                            v.zero_()
            records[intervention]=dict(steps=rows,observed_lane_diagnostics=observed.records)
    report=dict(kind='fixed_history_teacher_forced_interventions',episode_uid=args.episode_uid,
                parameters=lane_parameter_summary(model),replays=records,
                limitation='Native GDN/KV may retain earlier lane influence; these interventions do not isolate all past pathways. No closed-loop SR/SPL claim.')
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
