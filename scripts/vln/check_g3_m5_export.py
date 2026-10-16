"""Exact combined source/export equality and M5 online/offline replay audit."""
import argparse,gc,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scripts.vln.export_vggt_g1 import export_checkpoint
from scripts.vln.check_vggt_export import greedy_snapshot
from qwen_vl.stream.session import StreamSession
from qwen_vl.stream.cache import kv_length,assert_state_dtypes
from qwen_vl.train.vln_runtime import load_checkpoint


def snapshot(model,serializer,episode):
    data={k:v.cuda() if torch.is_tensor(v) else v for k,v in serializer.encode_episode(episode).items()}
    diagnostics={}
    with torch.inference_mode():offline=model(**data,episodic_diagnostics=diagnostics)['logits'].float().cpu()
    probe_report=None
    if model.g1_spec is not None:
        from qwen_vl.research.vggt_g3 import sample_pairs,observation_anchors
        from qwen_vl.research.vggt_sampling import collection_trajectory_key
        pairs=sample_pairs(collection_trajectory_key(episode),len(episode['steps']))
        assert pairs
        anchors=observation_anchors(data['response_target_positions'],data['response_action_index'],data['num_actions'])
        selected=anchors[torch.tensor([t for s,t in pairs],device=anchors.device)]
        # Targets do not enter queries/navigation. Zero targets here audit
        # only the probe-expanded forward; actual online targets are separately
        # tested and used in every training qualification.
        probe_diag={}
        with torch.inference_mode():
            expanded=model(**data,aux_anchors=selected,
                aux_lags=torch.tensor([t-s for s,t in pairs],device=anchors.device),
                aux_pair_count=len(pairs),aux_targets=torch.zeros(len(pairs),5,2048,device=anchors.device),
                episodic_diagnostics=probe_diag)['logits'].float().cpu()
        torch.testing.assert_close(expanded,offline,atol=.75,rtol=.03)
        rms=float((expanded-offline).square().mean().sqrt());assert rms<=.15
        overlaps=[]
        for layer,plan in diagnostics['plans'].items():
            for t in range(len(episode['steps'])):
                ids=plan.group_ids(t)
                if len(ids):overlaps.append(np.intersect1d(ids,probe_diag['plans'][layer].group_ids(t)).size/len(ids))
        probe_report=dict(pairs=len(pairs),rms=rms,max_abs=float((expanded-offline).abs().max()),
                          mean_selected_overlap=float(np.mean(overlaps)),min_selected_overlap=float(np.min(overlaps)))
    tokens=data['input_ids'][0,data['response_target_positions']].tolist();del data
    scores=[];cursor=[0];head=model.backbone.lm_head;original=head.forward
    def forced(hidden):
        value=original(hidden);assert torch.isfinite(value).all();scores.append(value.float().cpu())
        result=torch.full_like(value,-10000.);result[tokens[cursor[0]]]=10000.;cursor[0]+=1
        return result
    session=StreamSession(model,serializer);session.reset(episode['episode_uid'],episode['instruction'])
    spec=model.episodic_spec;assert spec.recent_groups==4 and spec.budget_tokens==1200
    selections=[];overlap=[];peak=0
    head.forward=forced
    try:
        with torch.inference_mode():
            for t,step in enumerate(episode['steps']):
                with Image.open(step['rgb_path']) as rgb:result=session.observe(episode['episode_uid'],t,rgb)
                assert result['action_name']==step['action_name']
                before=cursor[0];assert session.observe(episode['episode_uid'],t,None) is result;assert cursor[0]==before
                assert_state_dtypes(session.cache);peak=max(peak,kv_length(session.cache));assert peak<=4784
                assert [g.step_id for g in session.episodic.groups]==list(range(max(0,t-3),t+1))
                for layer in spec.attention_layers:
                    ids=session.episodic.running[layer].ids.copy();selections.append(ids)
                    expected=diagnostics['plans'][layer].group_ids(t)
                    if len(expected):overlap.append(np.intersect1d(ids,expected).size/len(expected))
                    recent=session.episodic.groups[-1].end-session.episodic.groups[0].start
                    assert session.episodic.resident[layer].size==session.prefix_length+len(ids)+recent
    finally:head.forward=original
    assert cursor[0]==len(tokens)
    streamed=torch.stack(scores);torch.testing.assert_close(streamed,offline,atol=.75,rtol=.03)
    rms=float((streamed-offline).square().mean().sqrt());assert rms<=.15
    first,greedy=greedy_snapshot(model,serializer,episode,'first')
    second,reset=greedy_snapshot(model,serializer,episode,'reset')
    assert first==second;torch.testing.assert_close(greedy,reset,atol=0,rtol=0)
    return dict(offline=offline,streamed=streamed,selections=selections,greedy=greedy,outcome=first,probe_report=probe_report,
                rms=rms,max_abs=float((streamed-offline).abs().max()),peak_kv=peak,
                mean_overlap=float(np.mean(overlap)),min_overlap=float(np.min(overlap)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--model-path',required=True);a=p.parse_args()
    selected={}
    for line in Path(a.manifest).open():
        ep=json.loads(line)
        if len(ep['steps'])>=17:selected.setdefault(ep['dataset'],ep)
        if len(selected)==2:break
    assert set(selected)=={'r2rce','rxrce'}
    for ep in selected.values():
        ep['steps']=ep['steps'][:17];ep['steps'][-1]['action_name']='STOP';ep['episode_uid']+=':g3-m5-audit'
    model,serializer=load_checkpoint(a.source,a.model_path);model.eval()
    assert model.g1_spec['run_id']=='G3_M5'
    config=model.navigation_config;before={key:snapshot(model,serializer,ep) for key,ep in selected.items()}
    del model,serializer;gc.collect();torch.cuda.empty_cache()
    output=export_checkpoint(a.source,a.output)
    model,serializer=load_checkpoint(output,a.model_path);model.eval()
    assert not hasattr(model,'research_aux') and model.navigation_config['memory']==config['memory']
    report=[]
    for key,ep in selected.items():
        after=snapshot(model,serializer,ep);prior=before[key]
        for field in ('offline','streamed','greedy'):torch.testing.assert_close(after[field],prior[field],atol=0,rtol=0)
        assert after['outcome']==prior['outcome']
        for x,y in zip(after['selections'],prior['selections'],strict=True):np.testing.assert_array_equal(x,y)
        report.append(dict(dataset=key,exact_export=True,probe_forward=prior['probe_report'],
            **{k:after[k] for k in ('rms','max_abs','peak_kv','mean_overlap','min_overlap','outcome')}))
    (output/'g3_m5_export_check.json').write_text(json.dumps(dict(passed=True,replay=report),indent=2))
    print('G3_M5_EXPORT_PASS',json.dumps(report),flush=True)


if __name__=='__main__':main()
