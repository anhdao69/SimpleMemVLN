from c2_speed_tools import *
results=[]
for name,config,arm,threads in [('representative_baseline',E/'baseline.yaml','v0',2),('representative_v0',E/'workers4.yaml','v0',8),('representative_m1',E/'workers4.yaml','m1',8),('representative_m2',E/'workers4.yaml','m2',8)]:
 result=run_case(name,config,arm,MANIFEST,20,threads)
 results.append(result)
 (E/'representative-results.json').write_text(json.dumps(results,indent=2)+'\n')
 if not result['passed']:raise SystemExit(1)
print('REPRESENTATIVE_DONE',flush=True)
