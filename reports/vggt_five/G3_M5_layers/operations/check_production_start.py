"""Read-only audit of the first queued full-training run and its first save."""
import argparse
import json
from pathlib import Path
import time
from scripts.vln.queue_g3_layers import ARMS,check_seal
from qwen_vl.train.recovery import validate_complete


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root);production=root/'production';run=production/'l8';train=run/'train'
    queue=json.loads((production/'queue_state.json').read_text())
    assert queue['status']=='running' and queue['arm']=='l8' and queue['node']=='worker-2'
    heartbeat=json.loads((production/'heartbeat.json').read_text())
    assert time.time()-heartbeat['unix']<60
    for arm in ARMS:check_seal(queue['sources'][arm],root/'qualification-f4d1c2f/qualification_pass.json')
    checkpoint=train/'checkpoint-10';record=validate_complete(checkpoint)
    assert record['step']==10 and record['world_size']==4
    state=json.loads((checkpoint/'trainer_state.json').read_text())
    assert state['global_step']==10 and state['max_steps']==3852 and state['num_train_epochs']==1
    contract=json.loads((checkpoint/'resume_contract.json').read_text())
    cfg=contract['navigation']['config'];t=cfg['training'];research=cfg['research'];data=contract['data']
    assert t['epochs']==1 and t['warmup_steps']==116 and t['expected_total_steps']==3852
    assert t['seed']==429 and t['gradient_accumulation_steps']==2
    assert t['backbone_lr']==5e-6 and t['weight_decay']==.01
    assert research['student_layers']==[8] and research['lambda_max']==.1 and research['ramp_updates']==771
    assert research['head_lr']==1e-4 and cfg['observations']['append_action_tokens'] is True
    assert data['manifest_sha256']=='5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933'
    assert len(data['episode_uids'])==30815 and data['world_size']==4 and data['accumulation']==2
    supervisors=sorted(run.glob('supervisor-*'))
    owned=json.loads((supervisors[-1]/'owned_process.json').read_text());command=owned['command']
    model=command[command.index('--model_name_or_path')+1]
    assert model.endswith('/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
    assert '--init-policy-checkpoint' not in command and '--max-optimizer-updates' not in command
    assert owned['host']=='worker-2' and str(owned['slurm_job'])=='4925'
    assert all(not (production/arm/'train').exists() for arm in ARMS[1:])
    profiles=[[json.loads(line) for line in (train/f'profile_rank{rank}.jsonl').read_text().splitlines()] for rank in range(4)]
    step=min(rows[-1]['update'] for rows in profiles);assert step>=10
    result=dict(verified=True,utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
        step_id='4925.101',global_step=step,checkpoint_step=10,checkpoint_files=len(record['files']),
        max_steps=state['max_steps'],epochs=t['epochs'],warmup_steps=t['warmup_steps'],seed=t['seed'],
        global_batch=8,episodes=len(data['episode_uids']),model=model,queue=queue,
        per_rank_latest=[rows[-1] for rows in profiles])
    (root/'production_start.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
