from pathlib import Path
import json,re,hashlib,subprocess
R=Path('/mnt/data/vmo-ai-task/anhdh35');E=R/'c2-speed-20261011';Q=E/'qualification-e789a13'
commit='e789a13f8b6ea3abc29086b82205a19024a6e7b5'
W=R/'SimpleMemVLN-c2_h2_speed-e789a13'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip()==commit
assert not subprocess.check_output(['git','diff','HEAD','--','.'],cwd=W)
pytest=(Q/'pytest.log').read_text();assert re.search(r'\b324 passed\b',pytest),pytest[-500:]
num=json.loads((Q/'numerical.json').read_text())
assert num['passed'] and num['source_commit']==commit and num['source_dirty'] is False
assert num['runtime_sha256']==hashlib.sha256((W/'configs/episodic/profile_h2.yaml').read_bytes()).hexdigest()
assert len(num['C_replay'])==18
suites=[]
for path,expected in [(Q/'resources/H2_suite.json',14),(Q/'m1/H2_suite.json',2)]:
 x=json.loads(path.read_text());assert x['passed'] and len(x['results'])==expected
 suites.append(x)
extra=E/'extra-arms-e789a13'
x=json.loads((extra/'m1-recovery.json').read_text());assert x['passed']
x=json.loads((extra/'training.json').read_text());assert x['passed'] and len(x['results'])==6
x=json.loads((extra/'replay.json').read_text());assert x['passed'] and len(x['replays'])==6 and x['source_commit']==commit
x=json.loads((extra/'dry_stream.json').read_text());assert x['peak_resident_kv']<=x['guard']
bench=[]
for name in ['representative_baseline','representative_v0','representative_m1','representative_m2','representative_baseline_warm']:
 x=json.loads((E/'experiments'/name/'result.json').read_text())
 assert x['passed'] and x['updates']==20
 bench.append(x)
assert len({x['exposure_sha256'] for x in bench})==1,'benchmark data exposures differ'
# Select the measured candidate only if it actually beats the warmed original settings.
assert bench[1]['steady_seconds'] < bench[-1]['steady_seconds'],'candidate is not faster than warm baseline'
result=dict(passed=True,source_commit=commit,tests=324,exposure_sha256=bench[0]['exposure_sha256'],
 warm_baseline_seconds=bench[-1]['steady_seconds'],v0_seconds=bench[1]['steady_seconds'],
 m1_seconds=bench[2]['steady_seconds'],m2_seconds=bench[3]['steady_seconds'],
 improvement_fraction=1-bench[1]['steady_seconds']/bench[-1]['steady_seconds'])
(E/'release.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
