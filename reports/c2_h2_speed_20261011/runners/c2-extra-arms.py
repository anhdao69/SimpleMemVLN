from c2_speed_tools import *
import subprocess,sys
out=E/'extra-arms-e789a13';out.mkdir(exist_ok=True)
# M1 gets the same explicit save/fresh-process-resume gate as V0 and M2.
from scripts.vln import profile_episodic as profile
m1_recovery=profile.checkpoint_gate('H2','m1',F/'longest_rxr.jsonl',out/'m1-recovery',MODEL,460.8)
(out/'m1-recovery.json').write_text(json.dumps(m1_recovery,indent=2)+'\n')
assert m1_recovery['passed']
results=[]
for fixture in ['medium_joint','longest_rxr']:
 for arm in ['m2c','m5','m5c']:
  r=run_case('extra_'+arm+'_'+fixture,'configs/episodic/profile_h2.yaml',arm,F/(fixture+'.jsonl'),3,8)
  results.append(r)
  (out/'training.json').write_text(json.dumps(dict(results=results,passed=all(x['passed'] for x in results)),indent=2)+'\n')
  if not r['passed']:raise SystemExit(1)
# Extend only the runner's list of existing versioned configurations; model code stays unchanged.
from scripts.vln import check_episodic as c
import torch
c.RUNTIME_PROFILE='configs/episodic/profile_h2.yaml'
c.ARMS.update({a:f'configs/episodic/arm_{a}.yaml' for a in ['m5','m5c']})
torch.manual_seed(429)
model,serializer=c.load_model(c.cfg_for('m2'),MODEL);model.eval();device=next(model.parameters()).device
replays=[]
for dataset in ['r2rce','rxrce']:
 ep=c.fixture(MANIFEST,dataset,96,rng_seed=96)
 for arm in ['m2c','m5','m5c']:
  r=c.gate_c(model,serializer,arm,ep,device);replays.append(r)
  print('EXTRA_REPLAY',json.dumps(r),flush=True)
  (out/'replay.json').write_text(json.dumps(dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runtime_sha256=hashlib.sha256(Path(c.RUNTIME_PROFILE).read_bytes()).hexdigest(),replays=replays,passed=all(x['passed'] for x in replays)),indent=2)+'\n')
  if not r['passed']:raise SystemExit(1)
dry=c.gate_e(model,serializer,'m5',c.fixture(MANIFEST,'rxrce',40),device)
(out/'dry_stream.json').write_text(json.dumps(dry,indent=2)+'\n')
assert dry['peak_resident_kv']<=dry['guard']
print('EXTRA_ARMS_DONE',flush=True)
