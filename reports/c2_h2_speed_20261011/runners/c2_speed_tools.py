from pathlib import Path
import json,os,statistics,hashlib
from scripts.vln import profile_episodic as p
R=Path('/mnt/data/vmo-ai-task/anhdh35'); E=R/'c2-speed-20261011'; F=R/'c2-audit-evidence-20261011/fixtures'
MODEL=str(R/'cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a')
MANIFEST=R/'SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl'
def rows(path):
 return [json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []
def run_case(name,config,arm,fixture,updates,threads,extra=('--profile-only',),first_update=1,guard=460.8):
 os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']=str(threads)
 p.PROFILES['H2']['runtime']=str(config)
 root=E/'experiments'/name
 result=p.run_gate('H2',arm,fixture,root,MODEL,updates,host_guard=guard,extra=extra,first_update=first_update)
 train=root/'H2'/result['gate']/'train'
 profiles=[rows(train/f'profile_rank{rank}.jsonl') for rank in range(2)]
 ends=[max(x[u]['unix_end'] for x in profiles) for u in range(min(map(len,profiles)))]
 intervals=[b-a for a,b in zip(ends,ends[1:])]
 exp=[rows(train/f'exposures_rank{r}.jsonl') for r in range(2)]
 result.update(name=name,threads=threads,config=str(config),steady_intervals=intervals,steady_seconds=statistics.mean(intervals) if intervals else None,
  exposure_sha256=hashlib.sha256(json.dumps([[(e['update'],e['episode_uid'],e['actions']) for e in rank] for rank in exp],sort_keys=True).encode()).hexdigest())
 (root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print('EXPERIMENT',json.dumps({k:result[k] for k in ['name','passed','steady_seconds','wall_seconds']}),flush=True)
 return result
