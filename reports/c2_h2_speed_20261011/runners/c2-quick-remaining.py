from c2_speed_tools import *
results=[]
for arm in ['m2c','m5','m5c']:
    result=run_case('quick_'+arm,'configs/episodic/profile_h2.yaml',arm,F/'smoke.jsonl',1,8)
    results.append(result)
    (E/'quick-remaining.json').write_text(json.dumps(dict(results=results,passed=all(x['passed'] for x in results)),indent=2)+'\n')
    if not result['passed']:raise SystemExit(1)
print('QUICK_REMAINING_DONE',flush=True)
