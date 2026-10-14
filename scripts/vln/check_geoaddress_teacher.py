"""Online Omega relation QC on disjoint TRAIN-calibration/diagnostic panels.

Diagnostic holdout is scene-disjoint from calibration within TRAIN, not a claim
of validation-set evaluation. Only statistics and labeled visualizations persist.
"""
import argparse,json,math
from pathlib import Path
import torch
from PIL import Image,ImageDraw
from qwen_vl.research.geoaddress_teacher import OnlineAddressTeacher,a1_teacher_identity
from qwen_vl.research.vggt_g3 import sample_pairs
from qwen_vl.research.geoaddress_sampling import query_cells
from qwen_vl.research.vggt_sampling import collection_trajectory_key
from qwen_vl.data.vggt_online import preprocess_teacher_images
from qwen_vl.models.geoaddress_loss import teacher_relations


def check_batch_independence(reference,perturbed_neighbors,single):
    # Same B and same first image: changing other B entries must have NO effect.
    # Changing B can choose different FP32 GEMM reductions; record that separately.
    torch.testing.assert_close(reference,perturbed_neighbors,atol=0,rtol=0)
    diff=single.float()-reference.float()
    relative=float(diff.norm()/reference.float().norm().clamp_min(1e-12))
    if relative>1e-4:raise AssertionError('Excessive whole-feature drift across teacher batch sizes')
    return dict(same_batch_exact=True,batch_size_max_abs=float(diff.abs().max()),batch_size_relative_l2=relative)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    episodes=[json.loads(line) for line in Path(a.manifest).read_text().splitlines()]
    selected=[]
    for dataset in sorted({e['dataset'] for e in episodes}):
        candidates=sorted((e for e in episodes if e['dataset']==dataset and len(e['steps'])>=33),key=collection_trajectory_key)
        used_scenes=set();used_keys=set()
        for panel in ('train_calibration','heldout_diagnostic_within_train'):
            for e in candidates:
                key=collection_trajectory_key(e)
                if e['scene_id'] not in used_scenes and key not in used_keys:
                    selected.append((panel,e));used_scenes.add(e['scene_id']);used_keys.add(key);break
            else:raise ValueError('Cannot construct disjoint-scene QC panels')
    root=Path('/mnt/data/vmo-ai-task/anhdh35');source=root/'third_party/vggt-omega';weights=root/'models/vggt-omega/vggt_omega_1b_512.pt'
    spec=dict(teacher_source=str(source),teacher_weights=str(weights),teacher_identity=a1_teacher_identity(source,weights),teacher_batch_size=8,teacher_residency='resident')
    teacher=OnlineAddressTeacher(spec,'cuda');records=[]
    for panel,ep in selected:
        key=collection_trajectory_key(ep);pairs=sample_pairs(key,len(ep['steps']))
        frames=sorted({x for pair in pairs for x in pair});images=preprocess_teacher_images([ep['steps'][i]['rgb_path'] for i in frames])
        features=teacher(images)
        assert features.shape==(len(frames),300,2048) and not features.requires_grad
        probs=[]
        for s,t in pairs:
            cells=query_cells(key,s,t).cuda()
            probs.append(teacher_relations(features[frames.index(t),cells][None],features[frames.index(s)][None])[0])
        probs=torch.stack(probs);entropy=-(probs*probs.clamp_min(1e-30).log()).sum(-1)
        spreads=probs.std(-1)
        assert torch.isfinite(probs).all() and bool((spreads>1e-7).any()),'All teacher relations numerically uniform'
        permutation=torch.randperm(300,device='cuda')
        same=teacher_relations(features[:1,:32],features[:1])
        permuted=teacher_relations(features[:1,:32],features[:1,permutation])
        torch.testing.assert_close(permuted,same[:,:,permutation],atol=1e-6,rtol=1e-5)
        neighbors=images[:min(len(images),8)].clone();neighbors[1:]=1-neighbors[1:]
        isolated=teacher(neighbors)[:1]
        single=teacher(images[:1])
        independence=check_batch_independence(features[:1],isolated,single)
        batched_probs=teacher_relations(features[:1,:32],features[1:2])
        single_probs=teacher_relations(single[:,:32],features[1:2])
        torch.testing.assert_close(single_probs,batched_probs,atol=1e-5,rtol=1e-4)
        independence['batch_size_relation_max_abs']=float((single_probs-batched_probs).abs().max())
        records.append(dict(panel=panel,dataset=ep['dataset'],episode_uid=ep['episode_uid'],scene_id=ep['scene_id'],
            trajectory_key=key,pairs=pairs,frames=frames,entropy_mean=float(entropy.mean()),entropy_min=float(entropy.min()),
            uniform_entropy=math.log(300),probability_std_mean=float(spreads.mean()),maximum_probability=float(probs.max()),
            uniform_rows=int((spreads<=1e-7).sum()),rows=spreads.numel(),same_frame_self_probability=float(same[0,torch.arange(32),torch.arange(32)].mean()),
            batch_independence=independence,permutation_pass=True))
        s,t=pairs[0];cell=int(query_cells(key,s,t)[0]);heat=probs[0,0].reshape(15,20).cpu()
        heat=((heat-heat.min())/(heat.max()-heat.min()).clamp_min(1e-9)*255).to(torch.uint8).numpy()
        with Image.open(ep['steps'][t]['rgb_path']) as im:current=im.copy()
        with Image.open(ep['steps'][s]['rgb_path']) as im:old=im.copy()
        h=Image.fromarray(heat).resize(old.size,Image.Resampling.NEAREST)
        color=Image.merge('RGB',(h,Image.new('L',h.size),Image.new('L',h.size)))
        old=Image.blend(old,color,.45);draw=ImageDraw.Draw(current);y,x=divmod(cell,20)
        draw.rectangle((x*32,y*32,(x+1)*32,(y+1)*32),outline='lime',width=3)
        canvas=Image.new('RGB',(1280,510),'white');canvas.paste(current,(0,30));canvas.paste(old,(640,30))
        ImageDraw.Draw(canvas).text((8,8),'Descriptor similarity only; NOT physical correspondence. Query (left), source relation (right).',fill='black')
        canvas.save(out/f'{ep["dataset"]}-{panel}.png')
    assert len(records)==4 and all(not p.requires_grad for p in teacher.model.parameters())
    report=dict(passed=True,teacher_identity=spec['teacher_identity'],single_view_shape='[B,1,3,448,592]',
        target_dim=2048,no_pca=True,no_position=True,no_dimension_converter=True,panels=records,
        limitation='Held-out diagnostic scenes are disjoint from calibration but drawn from TRAIN; not validation navigation evaluation.')
    (out/'teacher_qc.json').write_text(json.dumps(report,indent=2));print('ADDRESS_TEACHER_QC_PASS',json.dumps(report))

if __name__=='__main__':main()
