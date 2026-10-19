"""Snapshot completed resumed updates without changing the running trainer."""
import json, math, time
from pathlib import Path
R=Path('/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019')
Q=Path('/mnt/data/vmo-ai-task/anhdh35/g3a1-then-g1-future-20261019')
train=R/'production/train'
def records(path):
    # The trainer can be appending the final line while this snapshot is read.
    return [json.loads(line) for line in path.read_text().splitlines(keepends=True) if line.endswith('\n')]
profiles=[records(train/f'profile_rank{rank}.jsonl') for rank in range(4)]
last=min(max(x['update'] for x in rows) for rows in profiles)
assert last>=2110
rows=[x for stream in profiles for x in stream if 2100<x['update']<=last]
assert {x['update'] for x in rows}==set(range(2101,last+1))
assert len(rows)==4*(last-2100)
assert all(x['cuda_ooms']==0 for x in rows)
exposures=[x for rank in range(4) for x in records(train/f'exposures_rank{rank}.jsonl')]
old={x['episode_uid'] for x in exposures if x['update']<2100}
new=[x for x in exposures if 2100<=x['update']<last]
assert len(new)==8*(last-2100)
assert len({x['episode_uid'] for x in new})==len(new)
assert not old.intersection(x['episode_uid'] for x in new)
assert all(math.isfinite(x[k]) for x in new for k in ('loss_sum','aux_state_loss_sum','aux_relation_loss_sum'))
assert all(x['aux_weight']==.1 for x in new)
queue=json.loads((Q/'queue_state.json').read_text())
heartbeat=json.loads((Q/'heartbeat.json').read_text())
assert queue['status']=='running' and queue['stage']=='g3a1'
assert time.time()-heartbeat['unix']<60
result=dict(passed=True,unix=time.time(),last_completed_all_ranks=last,remaining_updates=3852-last,completed_new_exposures=len(new),new_episode_uids_unique=True,no_overlap_before_checkpoint=True,first_update=2101,all_losses_finite=True,aux_weight=.1,peak_allocated_gib=max(x['peak_allocated_gib'] for x in rows),peak_reserved_gib=max(x['peak_reserved_gib'] for x in rows),peak_working_set_gib=max(x['working_set_gib'] for x in rows),cuda_ooms=0,queue=queue,heartbeat=heartbeat)
(R/'resumed_progress_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
