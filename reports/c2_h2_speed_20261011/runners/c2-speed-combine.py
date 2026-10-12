from pathlib import Path
import json,time
from c2_speed_tools import *
while not (R/'c2-speed-screen.log').exists() or 'SCREEN_DONE' not in (R/'c2-speed-screen.log').read_text():time.sleep(10)
initial=json.loads((E/'screen-results.json').read_text())
# Combine the two fastest tested batching/loading choices with the best measured thread count.
thread_winner=min((x for x in initial if x['name'] in ['baseline','threads4','threads8'] and x['passed']),key=lambda x:x['steady_seconds'])['threads']
choices=sorted((x for x in initial if x['passed'] and x['name'] in ['workers4','batch16_vision8','batch32_vision16']),key=lambda x:x['steady_seconds'])[:2]
results=[]
for choice in choices:
 results.append(run_case('combined_'+choice['name'],choice['config'],'v0',F/'medium_joint.jsonl',4,thread_winner))
(E/'combined-results.json').write_text(json.dumps(results,indent=2)+'\n')
print('COMBINE_DONE',flush=True)
