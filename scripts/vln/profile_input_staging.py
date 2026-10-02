"""Replay update 414 or all-longest GAS slots through the production Trainer."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from torch.utils.data import SequentialSampler
from qwen_vl.train.train_episode import EpisodeTrainer, train_episode


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--case',choices=['failed','worst'],required=True)
    parser.add_argument('--verify-only',action='store_true')
    cli,rest=parser.parse_known_args()
    # Ordered exactly as the two microsteps of full-context update 414.
    suffixes=['25834:56014','31149:22206','18023:112780','54509:99680',
              '3812:2273','44133:55635','38039:103803','10187:122837']
    selected={}
    with open(cli.manifest) as f:
        for line in f:
            ep=json.loads(line)
            suffix=':'.join(ep['episode_uid'].split(':')[-2:])
            if suffix in suffixes and ep['episode_uid'].startswith('rxrce:'):
                if suffix in selected:
                    raise ValueError('Ambiguous diagnostic episode')
                selected[suffix]=ep
    assert len(selected)==8
    episodes=([selected[s] for s in suffixes] if cli.case=='failed'
              else [selected['31149:22206']]*8)
    assert len(episodes[1]['steps'])==627
    out=Path(cli.out)
    if cli.verify_only:
        verify_reports(out, episodes, cli.case)
        return
    if int(os.environ.get('RANK',0))==0:
        out.mkdir(parents=True,exist_ok=False)
        (out/'manifest.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    dist.barrier()
    # Diagnostic-only order override; production sampling is unchanged.
    EpisodeTrainer._get_train_sampler=lambda self,train_dataset=None: SequentialSampler(
        self.train_dataset if train_dataset is None else train_dataset)
    sys.argv=[sys.argv[0],*rest,'--manifest',str(out/'manifest.jsonl'),
              '--output_dir',str(out),'--max-optimizer-updates','3',
              '--profile-only','--debug-repeat-episodes']
    train_episode()
    # train_episode owns teardown. The driver verifies reports only after
    # torchrun has confirmed successful exit of every rank.


def verify_reports(out, episodes, case):
    peaks=[]
    for rank in range(4):
        rows=[json.loads(s) for s in (out/f'profile_rank{rank}.jsonl').read_text().splitlines()]
        assert [r['update'] for r in rows]==[1,2,3]
        peaks.extend(r['peak_reserved_gib'] for r in rows)
        exposures=[json.loads(s) for s in (out/f'exposures_rank{rank}.jsonl').read_text().splitlines()]
        assert [r['episode_uid'] for r in exposures]==[
            episodes[micro*4+rank]['episode_uid'] for _ in range(3) for micro in range(2)]
    assert all(0 < p <= 78 for p in peaks),max(peaks)
    (out/'PASS.json').write_text(json.dumps(dict(passed=True,case=case,
                                  peak_reserved_gib=max(peaks),updates=3)))
    print('INPUT_STAGING_MEMORY_PASS',case,max(peaks),flush=True)


if __name__=='__main__':
    main()
