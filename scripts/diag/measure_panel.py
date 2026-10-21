"""Gated full-panel D-a/D-c/D-d/D-f capture; no navigation updates.

State/ridge features are captured for every registered D-a layer. Extra M5
scorer states support the write-score/eviction-error bridge and are discarded
after saving per-token bridge quantities. D-b and probe fitting consume these
immutable artifacts in later separate stages.
"""
import argparse
import csv
import gc
import gzip
import json
from pathlib import Path
import time

M5_MODELS=('M5_E1','G3_M5','TRIPLE')
M5_CAPTURE=(2,6,8,10,14,16,18,22,24,26,30)
WINDOW_CAPTURE=(8,16,22,24,26)
ATTENTION=(3,7,11,15,19,23,27,31)


def validate_gate_evidence(gate,model,weight_hash,panel_uids,manifest_hash=None):
    expected=set(M5_CAPTURE if model in M5_MODELS else WINDOW_CAPTURE)
    rows=gate.get('rows',[])
    if (gate.get('status')!='PASS' or gate.get('model')!=model or
        gate.get('weights_sha256')!=weight_hash or gate.get('panel_episodes_completed',0)<2 or
        set(gate.get('input_capture_layers',[]))!=expected or
        set(gate.get('attention_capture_layers',[]))!=set(ATTENTION)):
        raise ValueError('Matching two-panel all-hook G-1/G-3/D-c evidence required')
    if manifest_hash is not None and [x.get('sha256') for x in gate.get('inputs',[])] != [manifest_hash]:
        raise ValueError('Instrumentation gate manifest identity differs')
    uids=[row['episode_uid'] for row in rows]
    if gate.get('selected_episode_uids') != uids:
        raise ValueError('Gate rows differ from declared selected episode identities')
    if (len(uids)<2 or len(set(uids))!=len(uids) or not set(uids)<=set(panel_uids) or
        any(row.get('input_kind')!='panel' or not all(row.get(k) for k in
            ('g1_pass','g3_pass','attention_pass')) for row in rows)):
        raise ValueError('Gate rows do not prove two distinct panel episodes passed every gate')



def checkpoint_weight(checkpoint):
    """Match load_checkpoint: safetensors wins when both weight files exist."""
    checkpoint=Path(checkpoint)
    safe=checkpoint/'model.safetensors'
    return safe if safe.exists() else checkpoint/'pytorch_model.bin'


def recall_denominators(rows):
    """Counts per region; Global overlaps quadrants and is not added to them."""
    sizes=dict(Q1=70,Q2=70,Q3=80,Q4=80,Global=300)
    return dict(unique_anchor_source_pairs=len({(r['anchor'],r['source']) for r in rows}),
        region_rows=len(rows),regions={name:dict(pairs=sum(r['region']==name for r in rows),
            tokens_per_pair=size,token_evaluations=size*sum(r['region']==name for r in rows))
            for name,size in sizes.items()})


def require_finite_logits(logits):
    """Reject invalid forward evidence; return the identical unmodified tensor."""
    import torch
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite measurement logits; episode cannot be recorded')
    return logits


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=(*M5_MODELS,'G3','G1'),required=True)
    for name in ('checkpoint','model-path','manifest','panel','teacher-cache','g2-result','instrument-result','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--research-config',type=Path)
    p.add_argument('--projection-cache',type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import torch
    from scripts.diag.cache_teacher import sha256,validate_panel
    from scripts.diag.state import LayerInputCapture,layout_from_encoded,reconstruct
    from scripts.diag.attention import AttentionCapture
    from scripts.diag.recall import recall_rows,gaussian_projection,current_view_features,native_error
    from scripts.diag.selection import selection_metrics,spearman
    from qwen_vl.train.vln_runtime import load_checkpoint
    episodes=[json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    panel=json.loads(args.panel.read_text());validate_panel(episodes,panel)
    expected_order=[str(row['episode_id']) for row in panel['episodes']]
    if [e['episode_id'] for e in episodes]!=expected_order:raise ValueError('Panel order changed')
    weight=checkpoint_weight(args.checkpoint)
    digest=sha256(weight);replay=json.loads(args.g2_result.read_text())
    if not replay.get('passed') or replay['weights_sha256']!=digest or replay['model']!=args.model:
        raise ValueError('Matching passed G-2 evidence required')
    gate=json.loads(args.instrument_result.read_text())
    validate_gate_evidence(gate,args.model,digest,{e['episode_uid'] for e in episodes},sha256(args.manifest))
    teacher=json.loads((args.teacher_cache/'identity.json').read_text())
    complete=json.loads((args.teacher_cache/'COMPLETE.json').read_text())
    if (teacher['manifest_sha256']!=sha256(args.manifest) or
        teacher['panel_sha256']!=sha256(args.panel) or complete['episodes']!=240 or
        complete['index_sha256']!=sha256(args.teacher_cache/'index.jsonl')):
        raise ValueError('Complete matching teacher cache required')
    research=None
    if args.model in ('G3_M5','TRIPLE'):
        if args.research_config is None:raise ValueError('D-f requires original research recipe')
        import yaml
        research=yaml.safe_load(args.research_config.read_text())['research']
    args.output.mkdir(parents=True,exist_ok=False)
    provenance=dict(model=args.model,checkpoint=str(args.checkpoint.resolve()),weights_sha256=digest,
        manifest_sha256=sha256(args.manifest),panel_sha256=sha256(args.panel),
        instrument_result_sha256=sha256(args.instrument_result),g2_result_sha256=sha256(args.g2_result),
        teacher_identity_sha256=sha256(args.teacher_cache/'identity.json'),
        teacher_index_sha256=complete['index_sha256'],
        research_config_sha256=sha256(args.research_config) if args.research_config else None,
        state_projection=dict(kind='standard_gaussian',dimensions=4096,seed=0,cpu_generation_dtype='float32',
                              cache_manifest_sha256=sha256(args.projection_cache/'manifest.json')),
        recall_sampling=dict(anchor_stride=2,lags=[0,*range(1,13),14,16,20,24,28,32],
                             regions=['Q1','Q2','Q3','Q4','Global']),
        eviction_bridge=dict(recent_groups=4,anchor='source+4',visual_tokens_per_source=300),
        diagnostics=['D-a frozen features/native error','D-c',
                     'D-d' if args.model in M5_MODELS else 'D-d not applicable',
                     'D-f' if research else 'D-f not applicable'],
        navigation_training=False,model_eval=True,inference_mode=True)
    write_json(args.output/'provenance.json',provenance)
    model,serializer=load_checkpoint(str(args.checkpoint),str(args.model_path))
    model.eval().requires_grad_(False)
    versions={name:p._version for name,p in model.named_parameters()}
    layers=model.backbone.model.language_model.layers
    device=next(model.parameters()).device
    requested=(8,16,22,24,26) if args.model=='TRIPLE' else (22,24,26)
    capture_indices=M5_CAPTURE if args.model in M5_MODELS else WINDOW_CAPTURE
    # Window8 captures five layers for identical G-1 instrumentation, even when
    # only the three registered reference layers are retained for D-a.
    with (args.output/'episodes.jsonl').open('x') as journal:
        for number,episode in enumerate(episodes):
            start=time.monotonic();folder=args.output/f'episode_{number:03d}';folder.mkdir()
            encoded={k:v.to(device) if torch.is_tensor(v) else v for k,v in serializer.encode_episode(episode).items()}
            layout=layout_from_encoded(encoded,serializer.image_id);diagnostics={}
            extra={'episodic_diagnostics':diagnostics} if model.episodic_spec is not None else {}
            with torch.inference_mode(),LayerInputCapture(layers,indices=capture_indices) as capture,\
                 AttentionCapture(layout,episode['episode_uid']) as attention:
                logits=require_finite_logits(model(**encoded,**extra)['logits'].detach().cpu())
            dc=attention.gate_summary();write_json(folder/'attention_gate.json',dc)
            if not dc['passed']:raise RuntimeError('D-c gate failed during full panel')
            with gzip.open(folder/'attention.json.gz','wt') as stream:
                json.dump(attention.records,stream,allow_nan=False)
            del attention
            torch.save(logits,folder/'logits.pt')
            torch.save(dict(layout=layout,diagnostics=diagnostics),folder/'selection_capture.pt')
            dd={str(layer):selection_metrics(plan,layout.visual_matrix(),diagnostics['visual_scores'][layer])
                for layer,plan in diagnostics.get('plans',{}).items()}
            write_json(folder/'selection.json',dd)
            if research is not None:
                from scripts.diag.probe_drift import episode_measurement
                plans={str(layer):[plan.group_ids(t).tolist() for t in range(len(episode['steps']))]
                       for layer,plan in diagnostics['plans'].items()}
                df=episode_measurement(model,serializer,episode,research,encoded=encoded,
                                       baseline=(logits.float().numpy(),plans))
                write_json(folder/'probe_drift.json',df)
                if not df['dummy_invariance_passed']:raise RuntimeError('D-f dummy-head invariance failed')
            del encoded,logits;gc.collect();torch.cuda.empty_cache()
            bridge={};written_layers=[];recall_counts={}
            for layer in capture_indices:
                if layer not in requested and layer+1 not in diagnostics.get('visual_scores',{}):
                    del capture.hidden[layer];continue
                record=reconstruct(layers[layer],capture.hidden.pop(layer),layout,check_fla=False)
                if layer in requested:
                    torch.save(record,folder/f'state_layer_{layer}.pt')
                    rows=recall_rows(record,device=device)
                    recall_counts[str(layer)]=recall_denominators(rows)
                    with (folder/f'recall_layer_{layer}.csv').open('x',newline='') as stream:
                        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
                    features=gaussian_projection(record['states'],device=device,cache=args.projection_cache)
                    torch.save(features,folder/f'projected_state_layer_{layer}.pt')
                    torch.save(current_view_features(record['visual_values']),folder/f'current_view_layer_{layer}.pt')
                    del rows,features;written_layers.append(layer)
                if layer+1 in diagnostics.get('visual_scores',{}):
                    scores=diagnostics['visual_scores'][layer+1].numpy()
                    eps=[]
                    for source in range(max(0,len(record['states'])-4)):
                        eps.append(native_error(record['states'][source+4].to(device),
                            record['visual_keys'][source].to(device),record['visual_values'][source].to(device)).cpu().numpy())
                    eps=np.stack(eps) if eps else np.empty((0,300))
                    np.savez_compressed(folder/f'eviction_bridge_layer_{layer}.npz',epsilon=eps,scores=scores[:len(eps)])
                    bridge[str(layer)]=spearman(scores[:len(eps)].ravel(),eps.ravel())
                del record;gc.collect();torch.cuda.empty_cache()
            write_json(folder/'eviction_bridge.json',bridge)
            if versions!={name:p._version for name,p in model.named_parameters()}:
                raise RuntimeError('Navigation parameter version changed')
            files={str(path.relative_to(folder)):sha256(path) for path in sorted(folder.iterdir()) if path.is_file()}
            row=dict(episode_id=episode['episode_id'],episode_uid=episode['episode_uid'],scene_id=episode['scene_id'],
                frames=len(episode['steps']),layers=written_layers,folder=folder.name,files_sha256=files,
                recall_counts=recall_counts,attention_counts=dc,
                selection_steps={key:len(value['steps']) for key,value in dd.items()},
                seconds=time.monotonic()-start,parameter_versions_unchanged=True)
            write_json(folder/'COMPLETE.json',row);journal.write(json.dumps(row)+'\n');journal.flush()
            print(json.dumps(dict(completed=number+1,total=240,episode_id=episode['episode_id'],seconds=row['seconds'])),flush=True)
            del capture,diagnostics,dd;gc.collect();torch.cuda.empty_cache()
    if sha256(weight)!=digest:raise RuntimeError('Checkpoint file changed')
    write_json(args.output/'COMPLETE.json',dict(episodes=240,weights_sha256=digest,
        journal_sha256=sha256(args.output/'episodes.jsonl'),parameter_versions_unchanged=True))


if __name__=='__main__':main()
