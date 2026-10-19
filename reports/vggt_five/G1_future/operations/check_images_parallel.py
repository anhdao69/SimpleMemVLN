from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import os,json,time
r=Path('/mnt/data/vmo-ai-task/anhdh35/g3a1-resume-worker2-20261019');alias=json.loads((r/'image_alias.json').read_text());start=time.time()
def check(line):
 e=json.loads(line);missing=[]
 for step in e['steps']:
  old=step['rgb_path'];assert old.startswith(alias['original_root']+'/')
  p=alias['alias_root']+old[len(alias['original_root']):]
  if not os.path.isfile(p):missing.append(p)
 return len(e['steps']),missing
n=steps=0;missing=[]
with (r/'download/data/r2r_rxr15_train.jsonl').open() as f,ThreadPoolExecutor(16) as pool:
 for count,bad in pool.map(check,f):n+=1;steps+=count;missing.extend(bad)
report=dict(passed=not missing,episodes=n,images=steps,missing_count=len(missing),first_missing=missing[:10],seconds=time.time()-start,workers=16)
(r/'image_check_parallel.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True);assert not missing and (n,steps)==(30815,3128624)
