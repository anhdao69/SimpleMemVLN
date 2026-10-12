from c2_speed_tools import *
r=run_case('representative_baseline_warm',E/'baseline.yaml','v0',MANIFEST,20,2)
raise SystemExit(0 if r['passed'] else 1)
