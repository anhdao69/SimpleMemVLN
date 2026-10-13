"""Sequential four-H100 G1 qualification; each experiment owns all four GPUs.

Run through g1_omega_env.sh inside an existing allocation. This is a diagnostic
campaign, not a production training launcher. Every failure remains in evidence.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--phase',choices=['safety','select','tuning','recovery'],required=True);a=p.parse_args()
    root=Path(a.root);fixtures=root/'fixtures'
    def has_headroom(result):
        # Unused allocator cache is reclaimable. Active tensors + CUDA context
        # must leave margin; the common gate separately caps reserved memory.
        if not result['passed']:
            return False
        directory=root/'experiments'/result['name']/'train'
        allocated=max(json.loads(line)['peak_allocated_gib'] for p in directory.glob('profile_rank*.jsonl')
                      for line in p.read_text().splitlines())
        return allocated<=72
    def run(name,fixture='mixed_tail',options=(),updates=3,required=True):
        destination=root/'experiments'/name
        if destination.exists():raise FileExistsError(destination)
        fixture_path=Path(fixture) if Path(fixture).is_absolute() else fixtures/f'{fixture}.jsonl'
        command=[sys.executable,'scripts/vln/profile_vggt_g1.py','--root',str(root),'--name',name,
                 '--fixture',str(fixture_path),'--updates',str(updates),*options]
        print('START',name,flush=True)
        with (root/f'{name}.log').open('w') as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
        result=json.loads((destination/'result.json').read_text())
        print('DONE',name,'passed',result['passed'],'seconds',result['steady_seconds'],
              'peak_GiB',result['check']['peak_reserved_gib'],flush=True)
        if required and (status or not result['passed']):raise RuntimeError(f'Failed {name}; inspect its log')
        return result
    if a.phase in ('safety','select'):
        candidates=[]
        for label,options in [('teacher-cpu',['--teacher-residency','cpu_between']),
                              ('optimizer-cpu',['--cpu-optimizer'])]:
            longest=(run(f'online-h4-{label}-longest','rxrce_longest',options,required=False)
                if a.phase=='safety' else json.loads((root/f'experiments/online-h4-{label}-longest/result.json').read_text()))
            if longest['passed']:
                speed=(run(f'online-h4-{label}-mixed',options=options) if a.phase=='safety'
                    else json.loads((root/f'experiments/online-h4-{label}-mixed/result.json').read_text()))
                if has_headroom(longest):
                    candidates.append(dict(label=label,options=options,seconds=speed['steady_seconds']))
        if not candidates:raise RuntimeError('No profile qualified with requested GPU headroom')
        best=min(candidates,key=lambda x:x['seconds'])
        (root/'safety_selection.json').write_text(json.dumps(dict(candidates=candidates,best=best),indent=2))
    elif a.phase=='tuning':
        base=json.loads((root/'safety_selection.json').read_text())['best'];options=base['options']
        variants=[('base',options,base['seconds']),
                  ('teacher16-attention16',options+['--teacher-batch','16','--attention-batch','16'],None),
                  ('threads4-workers2',options+['--threads','4','--workers','2'],None),
                  ('threads4-workers8',options+['--threads','4','--workers','8'],None),
                  ('vision8',options+['--vision-batch','8'],None),
                  ('teacher4',options+['--teacher-batch','4'],None),
                  ('optimizer-cpu-fast',['--cpu-optimizer','--teacher-batch','16',
                    '--attention-batch','32','--vision-batch','8','--threads','4','--workers','8'],None),
                  ('cpu-adam',['--cpu-adam'],None),
                  ('cpu-adam-fast',['--cpu-adam','--teacher-batch','16',
                    '--attention-batch','32','--vision-batch','8','--threads','4','--workers','8'],None)]
        records=[]
        for label,extra,seconds in variants:
            result=None if seconds is not None else run(f'online-h4-tune-{label}',options=extra,required=False)
            if result is not None and not result['passed']:continue
            records.append(dict(label=label,options=extra,seconds=seconds or result['steady_seconds']))
        records.sort(key=lambda x:x['seconds'])
        # A faster mixed fixture alone cannot qualify the longest sequence.
        for best in records:
            if best['label']=='base':break
            longest=run(f'online-h4-best-{best["label"]}-longest','rxrce_longest',best['options'],required=False)
            if has_headroom(longest):break
        else:raise RuntimeError('No tuned profile qualifies')
        (root/'tuning_selection.json').write_text(json.dumps(dict(candidates=records,best=best),indent=2))
        options=best['options']
        run('online-h4-best-r2r-longest','r2rce_longest',options)
        for dataset in ('r2rce','rxrce'):
            run(f'online-h4-best-{dataset}-representative',f'{dataset}_representative',options,updates=8)
        run('online-h4-native-reference',options=options+['--baseline'])
        run('online-h4-best-joint','/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl',
            options,updates=12)
    else:
        options=json.loads((root/'tuning_selection.json').read_text())['best']['options']
        run('online-h4-recovery-reference',options=options+['--save'])
        checkpoint=root/'experiments/online-h4-recovery-reference/train/checkpoint-2'
        run('online-h4-recovery-resumed',options=options+['--save','--resume',str(checkpoint),'--first-update','3'])


if __name__=='__main__':main()
