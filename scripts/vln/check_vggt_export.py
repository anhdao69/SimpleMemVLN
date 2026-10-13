"""Export G1 and audit native Window8 response replay, eviction and reset.

The 17-frame prefix is a serving diagnostic, never a training-length shortcut.
Tolerances are the existing native BF16 replay protocol (.75/.03, RMS .15).
"""
import argparse
import copy
import json
from pathlib import Path
from PIL import Image
import torch
from scripts.vln.export_vggt_g1 import export_checkpoint
from qwen_vl.stream.cache import assert_state_dtypes,kv_length
from qwen_vl.stream.session import StreamSession
from qwen_vl.train.vln_runtime import load_checkpoint


def greedy_outcome(session,uid,instruction,rgb):
    """A three-update base policy need not yet emit a valid bounded action.

    Preserve this quality failure explicitly; all other runtime errors still
    fail the audit. Equivalence must include scores/tokens and the outcome.
    """
    session.reset(uid,instruction)
    try:
        result=session.observe(uid,0,rgb)
    except ValueError as error:
        if str(error)!='Response limit reached without assistant terminator':raise
        return dict(valid_action=False,error=str(error))
    return dict(valid_action=True,response_token_ids=result['response_token_ids'],action_name=result['action_name'])


def greedy_snapshot(model,serializer,episode,uid):
    head=model.backbone.lm_head;original=head.forward;scores=[]
    def capture(hidden):
        value=original(hidden);assert torch.isfinite(value).all()
        scores.append(value.float().cpu());return value
    head.forward=capture
    try:
        session=StreamSession(model,serializer)
        with torch.inference_mode(),Image.open(episode['steps'][0]['rgb_path']) as rgb:
            outcome=greedy_outcome(session,uid,episode['instruction'],rgb)
        return outcome,torch.stack(scores)
    finally:head.forward=original


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True)
    p.add_argument('--output',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--model-path',required=True);a=p.parse_args()
    selected={}
    for line in Path(a.manifest).open():
        ep=json.loads(line)
        if len(ep['steps'])>=17:selected.setdefault(ep['dataset'],ep)
        if len(selected)==2:break
    assert len(selected)==2
    for ep in selected.values():
        ep['steps']=ep['steps'][:17];ep['steps'][-1]['action_name']='STOP'
        ep['episode_uid']+=':g1-serving-audit'
    source_model,source_serializer=load_checkpoint(a.source,a.model_path);source_model.eval()
    source_logits={};source_greedy={}
    with torch.inference_mode():
        for dataset,ep in selected.items():
            data={k:v.cuda() if torch.is_tensor(v) else v for k,v in source_serializer.encode_episode(ep).items()}
            source_logits[dataset]=source_model(**data)['logits'].float().cpu()
            source_greedy[dataset]=greedy_snapshot(source_model,source_serializer,ep,'source-greedy')
    del source_model,source_serializer,data
    import gc
    gc.collect();torch.cuda.empty_cache()
    path=export_checkpoint(a.source,a.output)
    model,serializer=load_checkpoint(path,a.model_path);model.eval()
    assert not hasattr(model,'research_aux') and getattr(model,'step_lane_spec',None) is None
    assert 'research' not in model.navigation_config
    report=[];head=model.backbone.lm_head;original=head.forward
    for dataset,source in selected.items():
        ep=copy.deepcopy(source)
        data={k:v.cuda() if torch.is_tensor(v) else v for k,v in serializer.encode_episode(ep).items()}
        targets=data['input_ids'][0,data['response_target_positions']].tolist()
        scores=[];cursor=[0];controlled=[True]
        def capture(hidden):
            value=original(hidden);assert torch.isfinite(value).all()
            scores.append(value.float().clone())
            if controlled[0]:
                forced=torch.full_like(value,-10000.);forced[targets[cursor[0]]]=10000.
                cursor[0]+=1;return forced
            return value
        session=StreamSession(model,serializer)
        with torch.inference_mode():
            offline=model(**data)['logits'].float();del data
            torch.testing.assert_close(offline.cpu(),source_logits[dataset],atol=0,rtol=0)
            head.forward=capture
            try:
                session.reset(ep['episode_uid'],ep['instruction'])
                for i,step in enumerate(ep['steps']):
                    with Image.open(step['rgb_path']) as rgb:
                        result=session.observe(ep['episode_uid'],i,rgb)
                    assert result['action_name']==step['action_name']
                    assert result['response_token_ids']==serializer.action_ids[result['class_id']]
                    before=len(scores)
                    assert session.observe(ep['episode_uid'],i,None) is result
                    assert len(scores)==before
                    assert_state_dtypes(session.cache)
                    assert [j for j,_ in session.resident]==list(range(max(0,i-7),i+1))
                    assert kv_length(session.cache)==session.prefix_length+sum(n for _,n in session.resident)
                streamed=torch.stack(scores);assert cursor[0]==len(targets)
                torch.testing.assert_close(streamed,offline,atol=.75,rtol=.03)
                diff=streamed-offline;rms=float(diff.square().mean().sqrt());assert rms<=.15
                record=dict(dataset=dataset,response_tokens=len(targets),max_abs=float(diff.abs().max()),rms=rms)
            finally:head.forward=original
        first,first_scores=greedy_snapshot(model,serializer,ep,'greedy')
        second,second_scores=greedy_snapshot(model,serializer,ep,'reset')
        assert first==second==source_greedy[dataset][0]
        torch.testing.assert_close(first_scores,second_scores,atol=0,rtol=0)
        torch.testing.assert_close(first_scores,source_greedy[dataset][1],atol=0,rtol=0)
        record.update(source_export_logits_exact=True,source_export_greedy_exact=True,
            greedy_outcome=first,greedy_tokens=first_scores.argmax(-1).tolist(),
            reset_exact=True,retry_exact=True,window8_eviction=True)
        report.append(record)
    result=dict(passed=True,source=a.source,export=str(path),replay=report)
    (path/'g1_export_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('G1_EXPORT_REPLAY_PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
