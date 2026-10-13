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
    if not research or research['run_id']!='G1': raise ValueError('Expected a G1 training checkpoint')
    state=torch.load(source/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    if not any(k.startswith('research_aux.current_projector.') for k in state):
        raise ValueError('Checkpoint missing training head')
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
