"""Strict namespace-only export of a G1 recovery model to teacher-free serving."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from qwen_vl.research.vggt_g1 import serving_config, serving_state_dict
from qwen_vl.train.vln_runtime import load_checkpoint


def export_checkpoint(source, destination):
    source, destination=Path(source),Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    metadata=json.loads((source/'navigation.json').read_text())
    research=metadata['config'].get('research')
    if not research or research['run_id'] not in ('G1','G3','A0','A1','R1'):
        raise ValueError('Expected a G1/G3/A0/A1 training checkpoint')
    state=torch.load(source/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    if research['run_id'] in ('A0','A1'):
        from qwen_vl.research.geoaddress_contract import parse_geoaddress
        if parse_geoaddress(metadata['config']) is None:
            raise ValueError('Checkpoint missing GeoAddress runtime contract')
        prefix='backbone.model.language_model.layers.24.linear_attn.geoaddress.'
        expected={prefix+n+'.weight':shape for n,shape in
                  [('q_down',(8,2560)),('k_down',(8,2560)),('q_up',(256,8)),('k_up',(256,8))]}
        actual={k:tuple(v.shape) for k,v in state.items() if '.geoaddress.' in k}
        if actual!=expected or any(k.startswith('research_aux.') for k in state):
            raise ValueError('Invalid GeoAddress weights or unexpected training head')
    else:
        head='current_projector' if research['run_id']=='G1' else 'state_head'
        if not any(k.startswith(f'research_aux.{head}.') for k in state):
            raise ValueError('Checkpoint missing training head')
        if research['run_id']=='R1':
            from qwen_vl.research.vggt_r1 import validate_r1
            spec=validate_r1(metadata['config'])
            installed={int(k.split('.layers.')[1].split('.')[0]) for k in state if '.step_lane.' in k}
            if installed!=set(spec.layers):raise ValueError('R1 export missing or unexpected installed lane')
    destination.mkdir(parents=True)
    torch.save(serving_state_dict(state),destination/'pytorch_model.bin')
    metadata['config']=serving_config(metadata['config'])
    (destination/'navigation.json').write_text(json.dumps(metadata,indent=2))
    (destination/'research_training.json').write_text(json.dumps(dict(source=str(source),research=research),indent=2))
    for name in ('config.json','generation_config.json','preprocessor_config.json','processor_config.json',
                 'tokenizer_config.json','tokenizer.json','chat_template.jinja'):
        if (source/name).is_file(): shutil.copy2(source/name,destination/name)
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',required=True);a=p.parse_args()
    path=export_checkpoint(a.source,a.output)
    model,_=load_checkpoint(path,a.model)  # strict=True in existing loader, no broad missing-key exemption
    assert not hasattr(model,'research_aux')
    print('STRICT_EXPORT_RELOAD_PASS',path,flush=True)
