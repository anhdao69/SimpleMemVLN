import importlib.util
from pathlib import Path
p=Path('reports/vggt_five/G3_M5_layers/operations/launch_after_qualification.py')
spec=importlib.util.spec_from_file_location('gate',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
fn=lambda seal,rec,steps:m.gate_status(seal,rec,steps,'4925.100','4925.101','4925')
assert fn(False,[],['4925.0','4925.100','4925.101'])=='waiting-for-qualification'
assert fn(True,[{'returncode':0}],['4925.0','4925.100','4925.101'])=='waiting-for-idle'
assert fn(True,[{'returncode':0}],['4925.0','4925.101','4925.102'])=='waiting-for-idle'
assert fn(True,[{'returncode':0}],['4925.0','4925.101'])=='ready'
for seal,records,steps in [(True,[{'returncode':1}],['4925.101']),(False,[],['4925.101'])]:
 try:fn(seal,records,steps)
 except RuntimeError:pass
 else:raise AssertionError('unsafe readiness')
print('6 launch-gate safety cases passed')
