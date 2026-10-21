"""Premeasurement panel selection; standard library only, no model imports.

Algorithm v1: scene quotas use constrained Hamilton apportionment. Fix scenes
whose proportional ideal is below the minimum at that minimum, recompute ideals
on the remaining population, floor ideals, then give remaining seats in descending
fractional-remainder order (lexicographic scene key breaks ties). Capacity is never
exceeded. Within each scene, apportion its quota proportionally over the population's
three GLOBAL rank tertiles. Rank by (continuous GT length, string episode ID),
assign tertile min(2, floor(3*rank/N)); equal lengths may cross a boundary, with
lexicographic episode IDs breaking ties. No model outputs inform these choices.

Selection: separate random.Random(seed) streams for shared, D-e and determinism
panels; canonical sorted scene/stratum/ID order before shuffle. Within each fixed
shared-panel stratum shuffle >=34-action and shorter candidates separately, then
take long candidates first. D-e has scene quotas only and no length preference.
The 60 determinism episodes are scene-proportional within the selected 600.
Action count includes terminal STOP; continuous path length is the sum of 3D
Euclidean distances between successive GT locations, not reference-path length
or action count. Missing/malformed GT fails closed instead of silently excluding
episodes. Missing instruction_id defaults to episode_id (VLN-CE convention).
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

DATASET_SHA256='1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3'
SEED=20261019


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_gzip(path, expected_hash=None):
    if expected_hash is not None and sha256(path)!=expected_hash:
        raise ValueError(f'Input SHA-256 mismatch: {path}')
    with gzip.open(path,'rt') as f:
        return json.load(f)


def records(data, gt):
    result=[]; seen=set()
    for ep in data['episodes']:
        eid=str(ep['episode_id'])
        if eid in seen:raise ValueError(f'Duplicate episode ID {eid}')
        seen.add(eid)
        if eid not in gt:raise ValueError(f'Missing GT: {eid}')
        item=gt[eid];loc=item.get('locations');actions=item.get('actions')
        if not loc or not actions or any(len(x)!=3 or not all(math.isfinite(v) for v in x) for x in loc):
            raise ValueError(f'Missing or invalid GT locations/actions: {eid}')
        if any(a not in (0,1,2,3) for a in actions):raise ValueError(f'Invalid GT action: {eid}')
        if actions[-1]!=0 or 0 in actions[:-1]:raise ValueError(f'Invalid terminal STOP: {eid}')
        iid=ep.get('instruction',{}).get('instruction_id',ep.get('instruction_id',eid))
        result.append(dict(episode_id=eid,instruction_id=str(iid),
                           instruction_id_source='explicit' if 'instruction_id' in ep or 'instruction_id' in ep.get('instruction',{}) else 'episode_id_fallback',
                           scene_id=ep['scene_id'],scene=Path(ep['scene_id']).stem,
                           path_length_m=math.fsum(math.dist(a,b) for a,b in zip(loc,loc[1:])),
                           gt_action_count=len(actions),gt_forward_steps=sum(a==1 for a in actions),
                           gt_location_count=len(loc)))
    return result


def apportion(populations, size, minimum=0):
    counts=dict(sorted(populations.items()))
    if size<0 or size>sum(counts.values()) or size<minimum*len(counts) or any(n<minimum for n in counts.values()):
        raise ValueError('Infeasible panel quota or minimum')
    quotas={}; active=dict(counts); remaining=size
    while active:
        total=sum(active.values())
        if not total:
            quotas.update({k:0 for k in active});break
        ideals={k:Fraction(remaining*n,total) for k,n in active.items()}
        fixed=[k for k,v in ideals.items() if v<minimum]
        if fixed:
            for k in fixed:quotas[k]=minimum;remaining-=minimum;del active[k]
            continue
        floors={k:int(v) for k,v in ideals.items()}
        for k in sorted(active,key=lambda k:(-(ideals[k]-floors[k]),k))[:remaining-sum(floors.values())]:floors[k]+=1
        quotas.update(floors);break
    return dict(sorted(quotas.items()))


def select(rows,size,seed,minimum=0,stratify=False,prefer_long=False):
    rng=random.Random(seed);scenes=defaultdict(list)
    for row in sorted(rows,key=lambda x:x['episode_id']):scenes[row['scene']].append(row)
    quotas=apportion({s:len(v) for s,v in scenes.items()},size,minimum)
    selected=[]
    for scene in sorted(scenes):
        groups=defaultdict(list)
        for row in scenes[scene]:groups[str(row['length_tertile']) if stratify else 'all'].append(row)
        sq=apportion({k:len(v) for k,v in groups.items()},quotas[scene])
        for key in sorted(groups):
            pool=groups[key]
            if prefer_long:
                long=[r for r in pool if r['gt_action_count']>=34];short=[r for r in pool if r['gt_action_count']<34]
                rng.shuffle(long);rng.shuffle(short);pool=long+short
            else:
                pool=list(pool);rng.shuffle(pool)
            selected.extend(dict(r) for r in pool[:sq[key]])
    return sorted(selected,key=lambda x:x['episode_id'])


def describe(rows):
    return {'scene_counts':dict(sorted(Counter(r['scene'] for r in rows).items())),
            'stratum_counts':dict(sorted(Counter(f"{r['scene']}:{r['length_tertile']}" for r in rows).items())),
            'long_episodes':sum(r['gt_action_count']>=34 for r in rows)}


def build_panels(rows,seed=SEED):
    if len({r['episode_id'] for r in rows})!=len(rows):raise ValueError('Duplicate episode IDs')
    ranked=sorted(rows,key=lambda x:(x['path_length_m'],x['episode_id']))
    ranked=[dict(r,length_tertile=min(2,3*i//len(rows))) for i,r in enumerate(ranked)]
    shared=select(ranked,240,seed,minimum=8,stratify=True,prefer_long=True)
    closed=select(ranked,600,seed)
    repeat=select(closed,60,seed)
    common={'seed':seed,'algorithm_version':1,'length_tertile_definition':'global rank floor(3*rank/N), ordered by (GT path_length_m, string episode_id); 0/1/2',
            'action_count_includes_terminal_stop':True,'instruction_id_fallback':'episode_id',
            'premeasurement_selection':True}
    a={'metadata':dict(common,**describe(shared),scene_minimum=8,prefer_at_least_actions=34), 'episodes':shared}
    b={'metadata':dict(common,**describe(closed),determinism_episode_ids=[r['episode_id'] for r in repeat],
                       determinism_scene_counts=describe(repeat)['scene_counts'],determinism_selection='scene-proportional 60 of selected 600, independent Random(seed)'), 'episodes':closed}
    return a,b


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    dataset=args.data_root/'val_unseen.json.gz';gtpath=args.data_root/'val_unseen_gt.json.gz'
    sources={str(p):sha256(p) for p in (dataset,gtpath)}
    rows=records(read_gzip(dataset,DATASET_SHA256),read_gzip(gtpath))
    if len(rows)!=1839 or len({r['scene'] for r in rows})!=11:raise ValueError('Expected 1839 episodes in 11 scenes')
    a,b=build_panels(rows)
    outputs={args.output_dir/'panel_ids.json':a,args.output_dir/'de_panel_ids.json':b}
    manifest_path=args.output_dir/'PANEL_MANIFEST.json'
    if any(p.exists() for p in [*outputs,manifest_path]):raise FileExistsError('Refusing to overwrite a frozen panel')
    for p,obj in outputs.items():
        with p.open('x') as f:json.dump(obj,f,indent=2,sort_keys=True);f.write('\n')
    if sources!={str(p):sha256(p) for p in (dataset,gtpath)}:raise RuntimeError('Source changed during selection')
    manifest={'source_gzip_sha256':sources,'dataset_matches_spec':sources[str(dataset)]==DATASET_SHA256,
              'panel_file_sha256':{p.name:sha256(p) for p in outputs},'selector_source_sha256':sha256(__file__),
              'algorithm':__doc__,'population':describe([dict(r,length_tertile=min(2,3*i//len(rows))) for i,r in enumerate(sorted(rows,key=lambda x:(x['path_length_m'],x['episode_id'])))]),
              'population_episodes':len(rows),'seed':SEED,'scope':'D-a through D-f only; no D-g panel',
              'source_bytes_unchanged':True,'measurements_performed':False}
    with manifest_path.open('x') as f:json.dump(manifest,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({'shared':a['metadata'],'de':b['metadata'],'hashes':manifest['panel_file_sha256']},indent=2))


if __name__=='__main__':main()
