"""Real-model lane gate: zero-output, temporal gradients and fixed-history streaming.

This is a numerical/runtime gate, not a navigation-quality evaluation. Memory-tail
and distributed optimizer checks use the production training scripts separately.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import torch
from PIL import Image
from qwen_vl.train.vln_runtime import load_model, load_checkpoint, resolve_config
from qwen_vl.data.episode_serializer import EpisodeSerializer
from qwen_vl.models.install_step_lane import install_step_lanes
from qwen_vl.research.lane_contract import parse_step_lane_spec
from qwen_vl.stream.session import StreamSession
from qwen_vl.stream.cache import kv_length


def inputs(serializer, episode, device):
    return {k:v.to(device) if torch.is_tensor(v) else v
            for k,v in serializer.encode_episode(episode).items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--memory', choices=['full_context','window8'], required=True)
    p.add_argument('--out', required=True)
    args=p.parse_args()
    torch.manual_seed(429)
    if args.checkpoint:
        model, serializer=load_checkpoint(args.checkpoint,args.model_path)
        cfg=deepcopy(model.navigation_config)
        if cfg['memory']['mode']!=args.memory or cfg['observations'].get('feedback_format')!='none':
            raise ValueError('Checkpoint must already match memory and no-history contract')
    else:
        cfg=resolve_config('configs/vln_r2r_v0_base.yaml','configs/vln_joint_b_2epoch_bs8.yaml',
                           'configs/vln_memory_window8.yaml' if args.memory=='window8' else None,
                           'configs/vln_candidate_logits_no_history.yaml')
        model,serializer=load_model(cfg,args.model_path)
    if getattr(model,'step_lane_spec',None) is not None:
        raise ValueError('Gate expects native parent/base; tests fresh zero-output installation')
    episodes={}
    with Path(args.manifest).open() as handle:
        for line in handle:
            e=json.loads(line)
            if len(e['steps'])>=17 and (e['dataset'] not in episodes or len(e['steps'])<len(episodes[e['dataset']]['steps'])):
                episodes[e['dataset']]=e
    if len(episodes)<2:
        raise ValueError('Gate requires a joint R2R/RxR manifest')
    device=next(model.parameters()).device
    first=next(iter(episodes.values()))
    data=inputs(serializer, first, device)
    model.train()
    model.gradient_checkpointing_enable()
    baseline=model(**data)
    base_logits=baseline['logits'].detach().clone()
    base_loss=baseline['loss_sum'].detach().clone()
    baseline['loss_sum'].backward()
    gradient_names=[n for n,p in model.named_parameters() if p.grad is not None and (
        '.layers.0.linear_attn.in_proj_qkv.weight' in n or
        '.layers.16.linear_attn.in_proj_qkv.weight' in n or
        '.layers.20.mlp.down_proj.weight' in n or
        '.layers.28.linear_attn.out_proj.weight' in n)]
    base_grad={n:dict(model.named_parameters())[n].grad.detach().clone() for n in gradient_names}
    assert len(base_grad)==4
    model.zero_grad(set_to_none=True)
    del baseline
    baseline_repeat=model(**data)
    baseline_repeat['loss_sum'].backward()
    repeat_errors={}
    for name,g in base_grad.items():
        repeat=dict(model.named_parameters())[name].grad
        repeat_errors[name]=dict(max_abs=float((repeat-g).float().abs().max()),
                                 norm_relative=float((repeat-g).float().norm()/g.float().norm()))
    print('NATIVE_REPEAT_GRADIENT_ERROR',json.dumps(repeat_errors),flush=True)
    model.zero_grad(set_to_none=True)
    del baseline_repeat
    cfg=deepcopy(cfg)
    cfg['model']['step_lane']={'enabled':True,'init_seed':429}
    model.navigation_config=cfg
    install_step_lanes(model,parse_step_lane_spec(cfg,model.config.text_config),init_seed=429)
    serializer=EpisodeSerializer(serializer.processor,cfg)
    data=inputs(serializer,first,device)
    out=model(**data)
    torch.testing.assert_close(out['logits'],base_logits,atol=0,rtol=0)
    torch.testing.assert_close(out['loss_sum'],base_loss,atol=0,rtol=0)
    out['loss_sum'].backward()
    named=dict(model.named_parameters())
    native_errors={name:dict(max_abs=float((named[name].grad-g).float().abs().max()),
                            norm_relative=float((named[name].grad-g).float().norm()/g.float().norm()))
                   for name,g in base_grad.items()}
    print('ZERO_LANE_NATIVE_GRADIENT_ERROR',json.dumps(native_errors),flush=True)
    # Native fused BF16 backward itself varies between identical repeated runs.
    # Bound lane-vs-native error by that measured control AND an absolute 3% norm cap.
    for name,error in native_errors.items():
        assert error['norm_relative'] <= .03, (name,error)
        assert error['norm_relative'] <= max(.002,2*repeat_errors[name]['norm_relative']), (name,error)
        assert error['max_abs'] <= max(1e-6,2*repeat_errors[name]['max_abs']), (name,error)
    lane_parameters={n:p for n,p in named.items() if '.step_lane.' in n}
    assert sum(p.numel() for p in lane_parameters.values())==21_054_000
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lane_parameters.values())
    opened_first_grad={n:float(p.grad.float().norm()) for n,p in lane_parameters.items() if n.endswith('out_proj.weight')}
    assert all(v>0 for v in opened_first_grad.values())
    model.zero_grad(set_to_none=True)
    del out,data,base_grad
    # Explicitly open the sole zero bottleneck to test the temporal writer path.
    with torch.no_grad():
        for n,p in lane_parameters.items():
            if n.endswith('out_proj.weight'):
                torch.nn.init.normal_(p,std=0.0005)
    data=inputs(serializer,first,device)
    out=model(**data)
    loss=torch.nn.functional.cross_entropy(out['logits'][-1:].float(),data['action_class_ids'][-1:])
    loss.backward()
    temporal={n:float(p.grad.float().norm()) for n,p in lane_parameters.items() if n.endswith(('in_proj_qkv.weight','in_proj_b.weight'))}
    assert all(v>0 and __import__('math').isfinite(v) for v in temporal.values())
    assert all(p.grad is None for p in model.backbone.model.visual.parameters())
    model.zero_grad(set_to_none=True)
    del out,data,loss
    model.eval()
    replay=[]
    for dataset,episode in sorted(episodes.items()):
        # Fixed expert observations; no-history closure is independent of chosen/gold class.
        ep=deepcopy(episode)
        ep['steps']=ep['steps'][:17]
        ep['steps'][-1]['action_name']='STOP'
        data=inputs(serializer,ep,device)
        changed=deepcopy(ep)
        for step in changed['steps'][:-1]:
            step['action_name']='TURN_LEFT' if step['action_name']!='TURN_LEFT' else 'TURN_RIGHT'
        other=inputs(serializer,changed,device)
        for key in ['input_ids','step_lane_roles','read_positions']:
            assert torch.equal(data[key],other[key])
        del other
        with torch.inference_mode():
            offline=model(**data)['logits'].float()
            s=StreamSession(model,serializer)
            s.reset(ep['episode_uid'],ep['instruction'])
            assert all(torch.count_nonzero(v)==0 for v in s.step_lane_cache._states.values())
            scores=[]
            readout=model.action_logits
            def capture(h):
                result=readout(h)
                scores.append(result.clone())
                return result
            model.action_logits=capture
            started=time.perf_counter()
            for t,step in enumerate(ep['steps']):
                with Image.open(step['rgb_path']) as rgb:
                    result=s.observe(ep['episode_uid'],t,rgb)
                cursor=dict(s.step_lane_cache._processed_tokens)
                assert s.observe(ep['episode_uid'],t,None) is result
                assert s.step_lane_cache._processed_tokens==cursor
                if args.memory=='window8':
                    assert len(s.resident)<=8
                    assert kv_length(s.cache)==s.prefix_length+sum(length for _,length in s.resident)
            elapsed=time.perf_counter()-started
            model.action_logits=readout
            actual=torch.stack(scores).float()
            diff=(actual-offline).abs()
            torch.testing.assert_close(actual,offline,atol=.75,rtol=.03)
            rms=float(diff.square().mean().sqrt())
            assert rms<=.15, rms
            assert all(v==s.positions.logical_token_count for v in s.step_lane_cache._processed_tokens.values())
            assert any(torch.count_nonzero(v)>0 for v in s.step_lane_cache._states.values())
            first_scores=actual[0].clone()
            s.reset('fresh',ep['instruction'])
            assert all(torch.count_nonzero(v)==0 for v in s.step_lane_cache._states.values())
            hidden=s._append_with_role(serializer.encode_observation(Image.open(ep['steps'][0]['rgb_path']),0),1)
            torch.testing.assert_close(readout(hidden[0,-1]).float(),first_scores,atol=0,rtol=0)
            replay.append(dict(dataset=dataset,episode_uid=ep['episode_uid'],steps=17,
                               fixture='first 17 real expert observations; forced terminal label for serializer',
                               max_logit_error=float(diff.max()),rms_logit_error=rms,
                               argmax_agreement=float((actual.argmax(-1)==offline.argmax(-1)).float().mean()),
                               streaming_seconds=elapsed))
    # Alternate two real sessions on one shared model. Each owns lane/native/position state.
    with torch.inference_mode():
        pair=list(episodes.values())[:2]
        def observe_score(session,episode,t):
            scores=[]
            original=model.action_logits
            def record(h):
                value=original(h)
                scores.append(value.detach().clone())
                return value
            model.action_logits=record
            try:
                with Image.open(episode['steps'][t]['rgb_path']) as rgb:
                    session.observe(session.episode_uid,t,rgb)
            finally:
                model.action_logits=original
            return scores[0]
        solo=[]
        for index,episode in enumerate(pair):
            session=StreamSession(model,serializer)
            session.reset(f'solo-{index}',episode['instruction'])
            solo.append([observe_score(session,episode,t) for t in range(10)])
        sessions=[StreamSession(model,serializer) for _ in pair]
        for index,(session,episode) in enumerate(zip(sessions,pair)):
            session.reset(f'interleaved-{index}',episode['instruction'])
        for t in range(10):
            for index,(session,episode) in enumerate(zip(sessions,pair)):
                other=sessions[1-index]
                preserved={k:v.clone() for k,v in other.step_lane_cache._states.items()}
                score=observe_score(session,episode,t)
                torch.testing.assert_close(score,solo[index][t],rtol=0,atol=0)
                for k,v in preserved.items():
                    assert torch.equal(v,other.step_lane_cache._states[k])
    report=dict(status='PASS',memory=args.memory,feedback='none',parent=args.checkpoint,
                interleaved_sessions_exact=True,
                zero_logits_loss_exact=True,native_gradient_names=gradient_names,
                native_repeat_gradient_errors=repeat_errors,zero_lane_native_gradient_errors=native_errors,
                native_gradient_gate='four sampled tensors: relative norm <= max(0.002,2x native repeat) AND <=0.03; max abs <= max(1e-6,2x native repeat)',
                lane_parameter_count=sum(p.numel() for p in lane_parameters.values()),
                first_backward_out_proj_norms=opened_first_grad,opened_temporal_gradients=temporal,
                replay=replay,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                note='Runtime/causality gate; no SR/SPL or longest-episode memory claim')
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
