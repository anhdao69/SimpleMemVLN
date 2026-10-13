"""Audit full serialized lengths and choose deterministic complete-episode fixtures."""
import json
from pathlib import Path
from transformers import AutoProcessor
from qwen_vl.data.episode_serializer import EpisodeSerializer
from qwen_vl.data.episode_dataset import EpisodeDataset
from qwen_vl.research.vggt_sampling import stable_hash
from qwen_vl.train.vln_runtime import resolve_config
import argparse

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
cfg=resolve_config('configs/episodic/science_common.yaml','configs/vln_empty.yaml',memory='configs/vln_memory_window8.yaml',runtime='configs/episodic/profile_h2.yaml')
serializer=EpisodeSerializer(AutoProcessor.from_pretrained(a.model),cfg)
data=EpisodeDataset(a.manifest,serializer)
by_dataset={}
for index,e in enumerate(data.episodes): by_dataset.setdefault(e['dataset'],[]).append(index)
fixtures={};audit={}
for name, indices in by_dataset.items():
    order=sorted(indices,key=lambda i:data.encoded_lengths[i]); longest=order[-1]
    # Distribution-stratified sample: 16 episodes spread over empirical lengths.
    representative=[order[min(len(order)-1,int((j+.5)*len(order)/16))] for j in range(16)]
    fixtures[name+'_representative']=[data.episodes[i] for i in representative]
    fixtures[name+'_longest']=[data.episodes[longest]]*8
    fixtures[name+'_short']=[data.episodes[order[len(order)//3]]]*8
    audit[name]=dict(episodes=len(order),actions=sum(len(data.episodes[i]['steps']) for i in order),
                     longest_steps=len(data.episodes[longest]['steps']),longest_tokens=data.encoded_lengths[longest],
                     mean_tokens=sum(data.encoded_lengths[i] for i in order)/len(order),
                     representative_tokens=[data.encoded_lengths[i] for i in representative])
lo=max((i for i,n in enumerate(data.encoded_lengths) if n<65536),key=lambda i:data.encoded_lengths[i])
hi=min((i for i,n in enumerate(data.encoded_lengths) if n>=65536),key=lambda i:data.encoded_lengths[i])
fixtures['threshold']=[data.episodes[i] for i in [lo,hi]*4]
fixtures['mixed_tail']=[data.episodes[i] for i in [lo,hi]]+list(fixtures['r2rce_representative'][:3])+list(fixtures['rxrce_representative'][:3])
all_episodes={e['episode_uid']:e for eps in fixtures.values() for e in eps}
for name,eps in dict(fixtures,cache_union=list(all_episodes.values())).items():
    (out/(name+'.jsonl')).write_text(''.join(json.dumps(e)+'\n' for e in eps))
(out/'length_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
