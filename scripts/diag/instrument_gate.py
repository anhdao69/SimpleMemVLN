"""G-1/G-3/D-c runner: transparent capture and native recurrence, no weight updates.

Fixture runs remain PARTIAL even when numerically successful: qualification
requires two distinct panel episodes per model and all final hooks installed.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback
# Direct script execution must not shadow stdlib statistics with diag/statistics.py.
if __name__ == '__main__':
    script_directory=Path(__file__).resolve().parent
    sys.path=[p for p in sys.path if Path(p or '.').resolve()!=script_directory]
import torch


def read_episodes(fixtures,manifest=None,uids=None):
    if bool(fixtures)==bool(manifest): raise ValueError('Supply fixtures OR a manifest')
    if manifest:
        text=Path(manifest).read_text()
        try:
            parsed=json.loads(text)
            rows=parsed if isinstance(parsed,list) else [parsed]
        except json.JSONDecodeError:
            rows=[json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows=[json.loads(Path(p).read_text()) for p in fixtures]
    by_uid={row['episode_uid']:row for row in rows}
    if len(by_uid)!=len(rows): raise ValueError('Duplicate episode UIDs in inputs')
    if uids:
        if len(set(uids))!=len(uids) or not set(uids)<=by_uid.keys():
            raise ValueError('Requested episode UIDs must be unique and present')
        rows=[by_uid[uid] for uid in uids]
    if not rows: raise ValueError('No episodes selected')
    return rows


def bitwise_logits(plain,instrumented):
    if plain.shape!=instrumented.shape or not plain.numel() or plain.dtype!=instrumented.dtype:
        raise ValueError('G-1 logit shape/dtype mismatch')
    if not bool(torch.isfinite(plain).all() and torch.isfinite(instrumented).all()):
        raise ValueError('Nonfinite G-1 logits')
    # Byte comparison also distinguishes signed zero, unlike torch.equal.
    passed=torch.equal(plain.contiguous().view(torch.uint8),instrumented.contiguous().view(torch.uint8))
    delta=plain.double()-instrumented.double()
    return dict(passed=passed,bitwise_equal=passed,shape=list(plain.shape),dtype=str(plain.dtype),
                elements=plain.numel(),max_abs=float(delta.abs().max()),rms=float(delta.square().mean().sqrt()),
                argmax_flips=int((plain.argmax(-1)!=instrumented.argmax(-1)).sum()))


def capture_layers(episodic):
    state=(8,16,22,24,26)
    return tuple(sorted(set(state)|{2,6,10,14,18,22,26,30})) if episodic else state


def raw_attention_steps(frames):
    if frames<1: raise ValueError('Empty episode')
    return tuple(sorted({0,frames-1}))


def tensor_digest(tensor):
    raw=tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(shape=list(tensor.shape),dtype=str(tensor.dtype),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def coverage(rows):
    if len({r['episode_uid'] for r in rows})!=len(rows): raise ValueError('Duplicate measured episodes')
    passed=bool(rows) and all(r['g1_pass'] and r['g3_pass'] and r.get('dc_pass',False) for r in rows)
    panels=sum(r['input_kind']=='panel' for r in rows)
    return dict(status=('FAIL' if rows and not passed else 'PASS' if passed and panels>=2 else 'PARTIAL'),
                passed_for_checked_episodes=passed,episodes_completed=len(rows),
                panel_episodes_completed=panels,required_panel_episodes=2,
                instrumentation_scope='state/scorer-input capture union plus post-RoPE attention wrapper')


def run(args):
    source=Path(args.source).resolve()
    sys.path[:0]=[str(source/'src'),str(source)]
    from scripts.diag.replay_gate import sha256,write_json,relocate_fixture
    from scripts.diag.state import LAYERS,LayerInputCapture,layout_from_encoded,reconstruct
    from scripts.diag.attention import AttentionCapture
    import qwen_vl.train.vln_runtime as runtime
    if not Path(runtime.__file__).resolve().is_relative_to(source/'src'):
        raise RuntimeError('Imported loader outside requested source')
    output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=False)
    summary=dict(model=args.model,status='RUNNING',source=str(source),rows=[],
                 state_gate_layers=list(LAYERS),input_kind=args.input_kind,command=sys.argv)
    def save(): write_json(output/'result.json',summary)
    save()
    try:
        checkpoint=Path(args.checkpoint).resolve()
        weight=checkpoint/'pytorch_model.bin'
        if not weight.exists(): weight=checkpoint/'model.safetensors'
        digest=sha256(weight)
        if digest!=args.weights_sha256: raise ValueError('Checkpoint SHA256 mismatch')
        summary['checkpoint']=str(checkpoint); summary['weights_sha256']=digest
        paths=args.fixture or [args.manifest]
        summary['inputs']=[dict(path=str(Path(p).resolve()),sha256=sha256(p)) for p in paths]
        episodes=read_episodes(args.fixture,args.manifest,args.episode_uid)
        summary['selected_episode_uids']=[x['episode_uid'] for x in episodes]
        summary['selected_episode_ids']=[str(x.get('episode_id','')) for x in episodes];save()
        torch.manual_seed(20261019)
        model,serializer=runtime.load_checkpoint(str(checkpoint),args.model_path or str(checkpoint))
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        layers=model.backbone.model.language_model.layers
        indices=capture_layers(model.episodic_spec is not None)
        summary['input_capture_layers']=list(indices)
        summary['attention_capture_layers']=[3,7,11,15,19,23,27,31];save()
        for number,original in enumerate(episodes):
            folder=output/f'episode_{number:03d}';folder.mkdir()
            episode,mapping=relocate_fixture(original,args.rgb_dir)
            write_json(folder/'inputs.json',dict(episode_uid=episode['episode_uid'],scene_id=episode.get('scene_id'),
                mapping=mapping,action_names=[x['action_name'] for x in episode['steps']]))
            encoded={k:v.cuda() if torch.is_tensor(v) else v for k,v in serializer.encode_episode(episode).items()}
            layout=layout_from_encoded(encoded,serializer.image_id)
            raw_steps=raw_attention_steps(layout.groups)
            raw_folder=folder/'attention_raw';raw_folder.mkdir()
            def raw_sink(layer,step,record):
                metadata={k:tensor_digest(v) if torch.is_tensor(v) else v for k,v in record.items()}
                metadata.update(layer=layer,step=step,raw_saved=step in raw_steps)
                if step in raw_steps:
                    destination=raw_folder/f'layer_{layer}_step_{step}.pt'
                    torch.save(record,destination)
                    metadata['file']=str(destination);metadata['file_sha256']=sha256(destination)
                with (folder/'attention_tensor_hashes.jsonl').open('a') as stream:
                    stream.write(json.dumps(metadata,allow_nan=False)+'\n')
            print('START_EPISODE',number,episode['episode_uid'],flush=True)
            with torch.inference_mode():
                plain=model(**encoded)['logits'].detach().cpu()
                with LayerInputCapture(layers,indices=indices) as capture, AttentionCapture(
                        layout,episode['episode_uid'],raw_sink=raw_sink) as attention:
                    hooked=model(**encoded)['logits'].detach().cpu()
            g1=bitwise_logits(plain,hooked)
            torch.save(dict(plain_logits=plain,instrumented_logits=hooked),folder/'g1_logits.pt')
            write_json(folder/'g1.json',g1)
            row=dict(episode_uid=episode['episode_uid'],input_kind=args.input_kind,
                     steps=len(episode['steps']),g1_pass=g1['passed'],g3_pass=False,
                     dc_pass=attention.gate_summary()['passed'],attention_pass=attention.gate_summary()['passed'],
                     layers={},dc=attention.gate_summary(),
                     raw_attention_steps=list(raw_steps))
            write_json(folder/'attention_measurements.json',attention.records)
            write_json(folder/'dc_gate.json',attention.gate_summary())
            summary['rows'].append(row);save()
            print('G1',json.dumps(g1),flush=True)
            if not g1['passed']: raise RuntimeError('G-1 bitwise parity failed')
            if not row['dc_pass']: raise RuntimeError('G-D-c attention output parity failed')
            print('DC_GATE',json.dumps(row['dc']),flush=True)
            del plain,hooked,attention
            # Free image tensors before the recurrence-only side pass.
            del encoded;gc.collect();torch.cuda.empty_cache()
            for index in LAYERS:
                print('START_G3_LAYER',index,flush=True)
                record=reconstruct(layers[index],capture.hidden.pop(index),layout,check_fla=True)
                torch.save(record,folder/f'state_layer_{index}.pt')
                gate=record['gate_g3']; row['layers'][str(index)]=gate
                write_json(folder/f'g3_layer_{index}.json',gate);save()
                peak=max(x['relative_frobenius_error'] for x in gate['anchors'])
                print('G3_LAYER',index,gate['status'],'max_relative_error',peak,flush=True)
                if gate['status']!='PASS': raise RuntimeError(f'G-3 failed at layer {index}')
                del record;gc.collect();torch.cuda.empty_cache()
            row['g3_pass']=True;save()
            del capture;gc.collect();torch.cuda.empty_cache()
        summary.update(coverage(summary['rows']));save()
        print('COMPLETE',summary['status'],flush=True)
    except BaseException as exc:
        summary.update(status='FAIL' if isinstance(exc,RuntimeError) and str(exc).startswith('G-') else 'ERROR',
                       error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        save()
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','weights-sha256','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--source',default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--model-path')
    inputs=p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--fixture',action='append',default=[])
    inputs.add_argument('--manifest')
    p.add_argument('--episode-uid',action='append',help='Explicit ordered selection from manifest')
    p.add_argument('--rgb-dir',help='Exact filename relocation only; manifest paths otherwise preserved')
    p.add_argument('--input-kind',choices=['fixture','panel'],required=True)
    args=p.parse_args()
    if args.manifest and not args.episode_uid: p.error('Manifest gate runs require explicit --episode-uid selection')
    run(args)

if __name__=='__main__': main()
