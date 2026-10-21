"""D-b frozen bank interventions after completed common gates and baseline panel.

No probe fitting, navigation updates, tolerance changes or native source edits.
Global B0 runs full depth/episode; targeted conditions run through decoder23 and
include the anchor group's response suffix only to satisfy complete-group layout.
Every measured state is sampled after observation, before that response.
"""
import argparse
import csv
import gc
import json
from pathlib import Path
import time
import traceback


def validate_baseline_contract(provenance,complete,model,weight_hash,manifest_hash,panel_hash):
    expected=dict(model=model,weights_sha256=weight_hash,manifest_sha256=manifest_hash,panel_sha256=panel_hash)
    if (any(provenance.get(k)!=v for k,v in expected.items()) or complete.get('episodes')!=240
            or complete.get('weights_sha256')!=weight_hash):
        raise ValueError('Complete matching 240-episode baseline required')


def prefix_gate_anchors(selection,frames):
    anchors=selection['anchors'] or [min(32,frames-1)]
    if frames<1 or any(not 0<=x<frames for x in anchors):raise ValueError('Prefix gate anchor outside episode')
    return list(anchors)


def adapter_gate_passed(rows,episode_ids):
    return ([r['episode_id'] for r in rows]==episode_ids and len(rows)==2 and
        all(r['adapter_bitwise'] and r['plans_equal'] and r.get('prefix_sensitivity')
            for r in rows))


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def write_csv(path,rows):
    if not rows:raise ValueError('No metric rows')
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['M5_E1','G3_M5','TRIPLE'],required=True)
    for name in ('checkpoint','model-path','manifest','panel','baseline-root','instrument-result',
                 'measurement-queue-status','gate-queue-status','projection-cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    import numpy as np
    import torch
    from scripts.diag import bank_ablation as bank
    from scripts.diag.replay_gate import sha256
    from scripts.diag.measure_panel import checkpoint_weight,validate_gate_evidence
    from scripts.diag.cache_teacher import validate_panel
    from scripts.diag.state import reconstruct,relative_state_error
    from scripts.diag.recall import recall_rows,native_error,region_means,current_view_features,gaussian_projection,REGIONS
    from scripts.diag.instrument_gate import bitwise_logits
    from qwen_vl.train.vln_runtime import load_checkpoint
    for status in (args.measurement_queue_status,args.gate_queue_status):
        if json.loads(status.read_text()).get('status')!='complete':
            raise ValueError('All measurement and gate GPU queues must complete before D-b')
    episodes=[json.loads(x) for x in args.manifest.read_text().splitlines() if x.strip()]
    panel=json.loads(args.panel.read_text());validate_panel(episodes,panel)
    ids=[str(x['episode_id']) for x in panel['episodes']]
    if ids!=sorted(ids) or [e['episode_id'] for e in episodes]!=ids:
        raise ValueError('Frozen panel/manifest lexicographic order changed')
    gate_ids=ids[:2]
    weight=checkpoint_weight(args.checkpoint);digest=sha256(weight)
    provenance=json.loads((args.baseline_root/'provenance.json').read_text())
    complete=json.loads((args.baseline_root/'COMPLETE.json').read_text())
    validate_baseline_contract(provenance,complete,args.model,digest,sha256(args.manifest),sha256(args.panel))
    if complete['journal_sha256']!=sha256(args.baseline_root/'episodes.jsonl'):
        raise ValueError('Baseline journal hash mismatch')
    baseline_rows=[json.loads(x) for x in (args.baseline_root/'episodes.jsonl').read_text().splitlines() if x.strip()]
    if len(baseline_rows)!=240 or [r['episode_uid'] for r in baseline_rows]!=[e['episode_uid'] for e in episodes]:
        raise ValueError('Baseline episode order/identity differs')
    validate_gate_evidence(json.loads(args.instrument_result.read_text()),args.model,digest,
                          {e['episode_uid'] for e in episodes},sha256(args.manifest))
    projection_hash=sha256(args.projection_cache/'manifest.json')
    if provenance.get('state_projection',{}).get('cache_manifest_sha256')!=projection_hash:
        raise ValueError('Projection cache differs from baseline feature coordinate system')
    args.output.mkdir(parents=True,exist_ok=False)
    progress=dict(status='running_adapter_gates',model=args.model,completed=0)
    def save():write_json(args.output/'STATUS.json',progress)
    save()
    metadata=dict(model=args.model,weights_sha256=digest,manifest_sha256=sha256(args.manifest),
        panel_sha256=sha256(args.panel),baseline_complete_sha256=sha256(args.baseline_root/'COMPLETE.json'),
        projection_manifest_sha256=projection_hash,gate_episode_ids=gate_ids,
        global_condition='B0 at3,7,11,15,19,23; R4 unchanged; upper banks enabled and recomputed',
        targeted_condition='all baseline plans frozen; removal only query group t; original IDs retained',
        prefix_definition='complete groups0..t through decoder23; GDN24 state snapshot before response_t',
        target_metric1_keys='no-removal prefix source s visual keys/values at the same anchor and truncated sequence length',
        targeted_metric1_reference='no-removal prefix state at the same anchor and truncated sequence length',
        prefix_state_sensitivity='reported descriptively against the full forward; it is not a parity gate because sequence length changes numerical execution',
        action_kl='teacher-forced response vocabulary KL, not semantic action probability KL',
        semantic_action_kl=None,original_head_cosine=None,
        original_head_reason='No identity-verified trained head supplied; no replacement head trained',
        learned_probe_status='pending existing scene-heldout probe prediction; no refit in this driver',
        parameter_updates=False)
    write_json(args.output/'provenance.json',metadata)
    def load_baseline(number):
        row=baseline_rows[number];folder=args.baseline_root/row['folder']
        for name in ('selection_capture.pt','logits.pt','state_layer_24.pt'):
            if sha256(folder/name)!=row['files_sha256'][name]:raise ValueError(f'Baseline artifact changed: {folder/name}')
        # Trusted locally generated objects, verified against the completed journal.
        capture=torch.load(folder/'selection_capture.pt',map_location='cpu',weights_only=False)
        logits=torch.load(folder/'logits.pt',map_location='cpu',weights_only=True)
        record=torch.load(folder/'state_layer_24.pt',map_location='cpu',weights_only=True)
        return capture['layout'],capture['diagnostics']['plans'],logits,record
    try:
        model,serializer=load_checkpoint(str(args.checkpoint),str(args.model_path));model.eval().requires_grad_(False)
        versions={k:v._version for k,v in model.named_parameters()}
        device=next(model.parameters()).device;layer24=model.backbone.model.language_model.layers[24]
        gate_folder=args.output/'adapter_gate';gate_folder.mkdir();gate_rows=[]
        # Mandatory two-panel adapter and prefix gates before counterfactual measurements.
        for number,episode in enumerate(episodes[:2]):
            layout,plans,baseline_logits,baseline_record=load_baseline(number)
            data={k:v.to(device) if torch.is_tensor(v) else v for k,v in serializer.encode_episode(episode).items()}
            prepared=bank.prepare_model_inputs(model,data)
            torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in prepared.items()},gate_folder/f'prepared_{number}.pt')
            full=bank.forward_model(model,data,prepared=prepared)
            copy_logits=torch.from_numpy(full['logits'])
            logit_gate=bitwise_logits(baseline_logits.float(),copy_logits)
            same_plans=all(full['diagnostics']['plans'][l].source_digest==plans[l].source_digest for l in plans)
            full_record=reconstruct(layer24,full['capture']['layer24_input'][0],layout,check_fla=False)
            selection=bank.choose_pairs(plans)
            row=dict(episode_id=episode['episode_id'],adapter_bitwise=logit_gate['passed'],
                     plans_equal=same_plans,logit_gate=logit_gate,prefix_sensitivity=[])
            gate_rows.append(row)
            torch.save(dict(original=baseline_logits,copied=copy_logits),gate_folder/f'logits_{number}.pt')
            if not logit_gate['passed'] or not same_plans:
                write_json(gate_folder/'results.json',dict(passed=False,rows=gate_rows));raise RuntimeError('Copied baseline adapter bitwise/plan gate failed')
            for anchor in prefix_gate_anchors(selection,layout.groups):
                prefix=bank.prefix_layer24_input(model,prepared,layout,plans,anchor)
                record=reconstruct(layer24,prefix['hidden'],prefix['layout'],check_fla=False)
                error=relative_state_error(record['states'][-1],full_record['states'][anchor])
                item=dict(anchor=anchor,relative_frobenius_error=error,
                          interpretation='descriptive_full_vs_truncated_sequence_length_sensitivity',
                          bitwise_equal=torch.equal(record['states'][-1],full_record['states'][anchor]),
                          prefix_tokens=prefix['prefix_tokens'],includes_response_suffix=True)
                row['prefix_sensitivity'].append(item)
                torch.save(dict(prefix=record['states'][-1],full=full_record['states'][anchor]),
                           gate_folder/f'states_episode_{number}_anchor_{anchor}.pt')
                write_json(gate_folder/'results.json',dict(
                    adapter_gate_passed=False,prefix_state_sensitivity_is_gate=False,rows=gate_rows))
                del prefix,record;gc.collect();torch.cuda.empty_cache()
            del data,prepared,full,full_record,baseline_record,baseline_logits;gc.collect();torch.cuda.empty_cache()
        passed=adapter_gate_passed(gate_rows,gate_ids)
        write_json(gate_folder/'results.json',dict(adapter_gate_passed=passed,
            prefix_state_sensitivity_is_gate=False,rows=gate_rows,episode_ids=gate_ids))
        if not passed:raise RuntimeError('Two-panel copied adapter/prefix gate incomplete')
        progress['status']='running_measurements';save()
        with (args.output/'episodes.jsonl').open('x') as journal:
            for number,episode in enumerate(episodes):
                started=time.monotonic();folder=args.output/f'episode_{number:03d}';folder.mkdir()
                layout,plans,baseline_logits,baseline_record=load_baseline(number)
                selection=bank.choose_pairs(plans);write_json(folder/'selection.json',selection)
                data={k:v.to(device) if torch.is_tensor(v) else v for k,v in serializer.encode_episode(episode).items()}
                prepared=(torch.load(gate_folder/f'prepared_{number}.pt',map_location=device,weights_only=True)
                          if number<2 else bank.prepare_model_inputs(model,data))
                global_result=bank.forward_model(model,data,prepared=prepared,bank_off_layers=bank.LOWER)
                write_json(folder/'global_vocabulary_kl.json',bank.logit_kl(baseline_logits.float().numpy(),global_result['logits']))
                global_record=reconstruct(layer24,global_result['capture']['layer24_input'][0],layout,check_fla=False)
                global_folder=folder/'global_bank_off';global_folder.mkdir()
                torch.save(torch.from_numpy(global_result['logits']),global_folder/'logits.pt')
                torch.save(dict(layout=layout,diagnostics=global_result['diagnostics']),global_folder/'selection_capture.pt')
                write_csv(global_folder/'recall_layer_24.csv',recall_rows(global_record,device=device))
                torch.save(current_view_features(global_record['visual_values']),global_folder/'current_view_layer_24.pt')
                state_batch=[global_record['states']];condition_states=[];condition_views=[];conditions=[];epsilon_rows=[]
                del global_result,global_record,baseline_logits;gc.collect();torch.cuda.empty_cache()
                prefix_baselines={}
                for anchor in sorted({p['anchor'] for p in selection['pairs'] if p['available']}):
                    baseline_prefix=bank.prefix_layer24_input(model,prepared,layout,plans,anchor)
                    baseline_prefix_record=reconstruct(layer24,baseline_prefix['hidden'],
                                                       baseline_prefix['layout'],check_fla=False)
                    prefix_baselines[anchor]=dict(
                        state=baseline_prefix_record['states'][-1].clone(),
                        visual_keys=baseline_prefix_record['visual_keys'],
                        visual_values=baseline_prefix_record['visual_values'])
                    del baseline_prefix,baseline_prefix_record;gc.collect();torch.cuda.empty_cache()
                for pair_index,pair in enumerate(selection['pairs']):
                    if not pair['available']:continue
                    anchor,source=pair['anchor'],pair['source']
                    reference=prefix_baselines[anchor]
                    keys=reference['visual_keys'][source].to(device);values=reference['visual_values'][source].to(device)
                    original=region_means(native_error(reference['state'].to(device),keys,values)).cpu()
                    for condition,key in [('targeted','target_ids'),('control','control_ids')]:
                        changed=bank.apply_removals(plans,anchor,pair[key])
                        prefix=bank.prefix_layer24_input(model,prepared,layout,changed,anchor)
                        record=reconstruct(layer24,prefix['hidden'],prefix['layout'],check_fla=False)
                        errors=native_error(record['states'][-1].to(device),keys,values).cpu()
                        means=region_means(errors)
                        condition_states.append(record['states'][-1:])
                        condition_views.append(current_view_features(record['visual_values'][-1:])[0])
                        conditions.append(dict(pair_index=pair_index,condition=condition,anchor=anchor,source=source,
                            lag=anchor-source,control_source=pair['control_source'],feature_index=len(conditions),
                            removed_tokens=pair['removed_tokens'],prefix_tokens=prefix['prefix_tokens']))
                        for region,name in enumerate(REGIONS):
                            epsilon_rows.append(dict(pair_index=pair_index,condition=condition,anchor=anchor,source=source,
                                lag=anchor-source,region=name,epsilon=float(means[region]),
                                baseline_epsilon=float(original[region]),delta=float(means[region]-original[region])))
                        torch.save(errors,folder/f'epsilon_pair_{pair_index}_{condition}.pt')
                        del prefix,record,changed;gc.collect();torch.cuda.empty_cache()
                if condition_states:state_batch.extend(condition_states)
                combined=torch.cat(state_batch)
                projected=gaussian_projection(combined,device=device,cache=args.projection_cache)
                torch.save(projected[:layout.groups],global_folder/'projected_state_layer_24.pt')
                targeted=dict(episode_id=episode['episode_id'],episode_uid=episode['episode_uid'],scene_id=episode['scene_id'],
                    conditions=conditions,projected_state=projected[layout.groups:],
                    current_view=torch.stack(condition_views) if condition_views else torch.empty(0,20480),
                    lag_embedding_appended=False,probe_prediction_status='pending existing heldout folds; never refit')
                torch.save(targeted,folder/'targeted_features.pt')
                if epsilon_rows:write_csv(folder/'targeted_epsilon.csv',epsilon_rows)
                contrasts=[]
                for pair_index in sorted({r['pair_index'] for r in epsilon_rows}):
                    for region in REGIONS:
                        values={r['condition']:r['epsilon'] for r in epsilon_rows if r['pair_index']==pair_index and r['region']==region}
                        contrasts.append(dict(pair_index=pair_index,region=region,targeted_minus_control=values['targeted']-values['control']))
                write_json(folder/'native_error_contrasts.json',contrasts)
                if versions!={k:v._version for k,v in model.named_parameters()}:raise RuntimeError('Navigation parameter version changed')
                row=dict(episode_id=episode['episode_id'],episode_uid=episode['episode_uid'],scene_id=episode['scene_id'],
                    frames=layout.groups,folder=folder.name,selected_pairs=len(selection['pairs']),
                    matched_pairs=sum(p['available'] for p in selection['pairs']),conditions=len(conditions),
                    unavailable=selection['unavailable'],seconds=time.monotonic()-started,parameter_versions_unchanged=True,
                    files_sha256={str(f.relative_to(folder)):sha256(f) for f in sorted(folder.rglob('*')) if f.is_file()})
                write_json(folder/'COMPLETE.json',row);journal.write(json.dumps(row)+'\n');journal.flush()
                progress.update(completed=number+1,last_episode_id=episode['episode_id']);save()
                print(json.dumps(dict(completed=number+1,total=240,episode_id=episode['episode_id'])),flush=True)
                del data,prepared,baseline_record,state_batch,condition_states,condition_views,combined,projected,targeted;gc.collect();torch.cuda.empty_cache()
        if sha256(weight)!=digest:raise RuntimeError('Checkpoint file changed')
        write_json(args.output/'COMPLETE.json',dict(episodes=240,model=args.model,weights_sha256=digest,
            journal_sha256=sha256(args.output/'episodes.jsonl'),adapter_gate_passed=True,
            learned_probe_prediction='pending existing heldout model application',parameter_versions_unchanged=True))
        progress['status']='complete';save()
    except BaseException as exc:
        progress.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc());save();raise

if __name__=='__main__':main()
