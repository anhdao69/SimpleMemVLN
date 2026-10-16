"""Fixed 4x8 rectangle query coverage on the actual 15x20 visual grid."""
import torch
from qwen_vl.research.vggt_sampling import stable_hash


def query_cells(trajectory_key, source, read, seed=429):
    cells=[]
    for row in range(4):
        for col in range(8):
            candidates=[y*20+x for y in range(row*15//4,(row+1)*15//4)
                        for x in range(col*20//8,(col+1)*20//8)]
            selected=int(stable_hash([trajectory_key,source,read,row*8+col,seed]),16)%len(candidates)
            cells.append(candidates[selected])
    return torch.tensor(cells,dtype=torch.long)


def address_metadata(episode,result,image_id,run_id,seed):
    from qwen_vl.research.vggt_sampling import collection_trajectory_key,sampled_visual_positions
    from qwen_vl.research.vggt_g3 import sample_pairs
    key=collection_trajectory_key(episode)
    pairs=sample_pairs(key,len(episode['steps']),seed) if run_id in ('A1','G3A1') else ()
    frames=sorted({v for pair in pairs for v in pair})
    visual=sampled_visual_positions(result['input_ids'],image_id,result['step_plan'][1],frames)
    sources=torch.tensor([frames.index(s) for s,t in pairs],dtype=torch.long)
    reads=torch.tensor([frames.index(t) for s,t in pairs],dtype=torch.long)
    cells=torch.stack([query_cells(key,s,t,seed) for s,t in pairs]) if pairs else torch.empty(0,32,dtype=torch.long)
    qpos=visual[reads].gather(1,cells)
    meta=dict(aux_query_positions=qpos,aux_key_positions=visual[sources],aux_pair_count=len(pairs))
    if run_id in ('A1','G3A1'):
        meta.update(aux_teacher_sources=sources,aux_teacher_reads=reads,aux_query_cells=cells)
    return meta,frames
