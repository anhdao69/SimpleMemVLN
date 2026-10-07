"""Pinned text-policy gate: zero-init, writer gradients, fixed-history decoding.

Teacher forcing controls only argmax decisions of StreamSession's real text loop;
the captured vocabulary scores and model/cache appends remain unmodified.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import torch
from PIL import Image
from qwen_vl.train.vln_runtime import load_model, resolve_config
from qwen_vl.data.episode_serializer import EpisodeSerializer
from qwen_vl.models.install_step_lane import install_step_lanes
from qwen_vl.research.lane_contract import parse_step_lane_spec
from qwen_vl.stream.session import StreamSession
from qwen_vl.stream.cache import kv_length


def encode(serializer, episode, device):
    return {k:v.to(device) if torch.is_tensor(v) else v
            for k,v in serializer.encode_episode(episode).items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--config',default='configs/vln_dual_window8_text_joint.yaml')
    p.add_argument('--out',required=True)
    args=p.parse_args()
    torch.manual_seed(429)
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text())
    native=deepcopy(cfg); native['model'].pop('step_lane')
    model,serializer=load_model(native,args.model_path)
    device=next(model.parameters()).device
    episodes={}
    for line in Path(args.manifest).open():
        e=json.loads(line)
        if len(e['steps'])>=17 and (e['dataset'] not in episodes or len(e['steps'])<len(episodes[e['dataset']]['steps'])):
            episodes[e['dataset']]=e
    assert len(episodes)==2
    for e in episodes.values():
        e['steps']=e['steps'][:17]; e['steps'][-1]['action_name']='STOP'
    first=next(iter(episodes.values()))
    old=encode(serializer,first,device)
    model.train(); model.gradient_checkpointing_enable()
    result=model(**old)
    native_logits=result['logits'].detach().clone(); native_loss=result['loss_sum'].detach().clone()
    del result
    native['runtime']['window_attention_batch_steps']=1
    serial=model(**old)
    attention_diff=(serial['logits'].detach().float()-native_logits.float()).abs()
    torch.testing.assert_close(serial['logits'],native_logits,atol=.75,rtol=.03)
    attention_rms=float(attention_diff.square().mean().sqrt())
    assert attention_rms<=.15,attention_rms
    attention_comparison=dict(max_logit_error=float(attention_diff.max()),rms_logit_error=attention_rms,
                              loss_sum_difference=float((serial['loss_sum'].detach()-native_loss).abs()))
    del serial,attention_diff
    model.navigation_config=cfg
    install_step_lanes(model,parse_step_lane_spec(cfg,model.config.text_config),init_seed=429)
    serializer=EpisodeSerializer(serializer.processor,cfg)
    data=encode(serializer,first,device)
    for k,v in old.items():
        if torch.is_tensor(v): assert torch.equal(v,data[k]),k
        else: assert v==data[k],k
    out=model(**data)
    torch.testing.assert_close(out['logits'],native_logits,rtol=0,atol=0)
    torch.testing.assert_close(out['loss_sum'],native_loss,rtol=0,atol=0)
    out['loss_sum'].backward()
    lanes={n:p for n,p in model.named_parameters() if '.step_lane.' in n}
    assert sum(p.numel() for p in lanes.values())==21_054_000
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lanes.values())
    opening={n:float(p.grad.float().norm()) for n,p in lanes.items() if n.endswith('out_proj.weight')}
    assert all(v>0 for v in opening.values())
    model.zero_grad(set_to_none=True)
    del out,old,native_logits,native_loss
    with torch.no_grad():
        for n,p in lanes.items():
            if n.endswith('out_proj.weight'): torch.nn.init.normal_(p,std=.0005)
    out=model(**data)
    last=data['response_action_index']==data['num_actions']-1
    torch.nn.functional.cross_entropy(out['logits'][last].float(),data['input_ids'][0,data['response_target_positions'][last]]).backward()
    temporal={n:float(p.grad.float().norm()) for n,p in lanes.items() if n.endswith(('in_proj_qkv.weight','in_proj_b.weight'))}
    assert all(v>0 and __import__('math').isfinite(v) for v in temporal.values())
    assert all(p.grad is None for p in model.backbone.model.visual.parameters())
    model.zero_grad(set_to_none=True); del out,data
    model.eval(); replay=[]
    with torch.inference_mode():
        for dataset,ep in sorted(episodes.items()):
            data=encode(serializer,ep,device)
            offline=model(**data)['logits'].float()
            targets=data['input_ids'][0,data['response_target_positions']].tolist()
            scores=[]; cursor=[0]; head=model.backbone.lm_head; original=head.forward
            def forced(h):
                value=original(h)
                scores.append(value.float().clone())
                token=targets[cursor[0]]; cursor[0]+=1
                controlled=torch.full_like(value,-10000.); controlled[token]=10000.
                return controlled
            session=StreamSession(model,serializer); session.reset(ep['episode_uid'],ep['instruction'])
            assert all(torch.count_nonzero(x)==0 for x in session.step_lane_cache._states.values())
            head.forward=forced
            try:
                for t,step in enumerate(ep['steps']):
                    with Image.open(step['rgb_path']) as rgb: result=session.observe(ep['episode_uid'],t,rgb)
                    assert result['action_name']==step['action_name']
                    before=dict(session.step_lane_cache._processed_tokens)
                    assert session.observe(ep['episode_uid'],t,None) is result
                    assert before==session.step_lane_cache._processed_tokens
                    assert len(session.resident)<=8
                    assert kv_length(session.cache)==session.prefix_length+sum(n for _,n in session.resident)
            finally:
                head.forward=original
            assert cursor[0]==len(targets)
            actual=torch.stack(scores)
            diff=(actual-offline).abs(); rms=float(diff.square().mean().sqrt())
            # Existing BF16 Window8 gate bounds, declared before measurements.
            torch.testing.assert_close(actual,offline,atol=.75,rtol=.03)
            assert rms<=.15,rms
            assert all(n==session.positions.logical_token_count for n in session.step_lane_cache._processed_tokens.values())
            assert int(data['step_lane_roles'].eq(3).sum())==17
            session.reset('fresh',ep['instruction'])
            assert all(torch.count_nonzero(x)==0 for x in session.step_lane_cache._states.values())
            replay.append(dict(dataset=dataset,steps=17,tokens=len(targets),max_logit_error=float(diff.max()),
                               rms_logit_error=rms,argmax_agreement=float((actual.argmax(-1)==offline.argmax(-1)).float().mean())))
            del data,offline,actual,scores,session
    report=dict(status='PASS',output='qwen_text',memory='window8',zero_logits_loss_exact=True,
                serial_vs_batched=attention_comparison,
                first_backward_out_proj_norms=opening,opened_temporal_gradients=temporal,replay=replay,
                note='Fixed expert observations and forced canonical response tokens; no navigation-quality claim.')
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__': main()
