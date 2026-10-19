import json,math,time
from pathlib import Path
B=Path('/mnt/data/vmo-ai-task/anhdh35');G=B/'g3-fixed01-seed430-worker2-20261019';R=B/'g3a1-resume-worker2-20261019'
def rows(path):return [json.loads(s) for s in path.read_text().splitlines(keepends=True) if s.endswith('\n')]
p=[rows(R/f'production/train/profile_rank{r}.jsonl') for r in range(4)]
last=min(max(x['update'] for x in stream) for stream in p);assert last>2649
new=[x for stream in p for x in stream if 2649<x['update']<=last]
assert len(new)==4*(last-2649) and min(x['update'] for x in new)==2650
assert all(x['cuda_ooms']==0 for x in new)
e=[x for r in range(4) for x in rows(R/f'production/train/exposures_rank{r}.jsonl')]
old={x['episode_uid'] for x in e if x['update']<2649}
recent=[x for x in e if 2649<=x['update']<last]
assert len(recent)==8*(last-2649) and not old.intersection(x['episode_uid'] for x in recent)
assert len({x['episode_uid'] for x in recent})==len(recent)
assert all(math.isfinite(x[k]) for x in recent for k in ('loss_sum','aux_state_loss_sum','aux_relation_loss_sum'))
queue=json.loads((G/'queue/queue_state.json').read_text());assert queue['status']=='running' and queue['stage']=='g3a1'
config=json.loads((G/'queue/queue.json').read_text());assert [s['name'] for s in config['stages']]==['g3a1','g3_fixed']
heartbeat=json.loads((G/'queue/heartbeat.json').read_text());assert time.time()-heartbeat['unix']<60
result=dict(passed=True,unix=time.time(),resumed_from=2649,current_step=last,full_steps=3852,new_exposures=len(recent),no_repeated_episodes=True,finite_losses=True,cuda_ooms=0,queue=queue,heartbeat=heartbeat,next_run=config['stages'][1]['run'])
(G/'resumed_queue_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
