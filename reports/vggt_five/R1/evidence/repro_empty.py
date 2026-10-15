import os,json,runpy,faulthandler
from pathlib import Path
rank=os.environ.get('RANK','0')
root=Path(os.environ['R1_DIAG_ROOT'])
root.mkdir(parents=True,exist_ok=True)
trace=(root/f'stacks_rank{rank}.txt').open('w')
faulthandler.dump_traceback_later(45,repeat=True,file=trace)
from qwen_vl.data.episode_dataset import EpisodeLengthSampler
EpisodeLengthSampler.__iter__=lambda self:iter(range(len(self.lengths)))
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
original=DeepSpeedZeroOptimizer.reduce_ipg_grads
def reduced(self):
    record={'micro':self.micro_step_id,'elements':self.elements_in_ipg_bucket,
            'params':[(g,i,p) for g,i,p in self.params_in_ipg_bucket]}
    with (root/f'buckets_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    return original(self)
DeepSpeedZeroOptimizer.reduce_ipg_grads=reduced
import sys,hashlib
import qwen_vl.models.vggt_r1_forward as repaired
from scripts.vln.queue_geoaddress import source_files
module=Path(repaired.__file__).resolve()
fixture=Path(sys.argv[sys.argv.index('--manifest')+1])
identity=dict(module_path=str(module),module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),
              source_files=source_files(module.parents[3]),
              fixture=str(fixture),fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
              argv=sys.argv,slurm_job=os.environ.get('SLURM_JOB_ID'),node=os.environ.get('SLURMD_NODENAME'),
              completed=False)
(root/f'identity_rank{rank}.json').write_text(json.dumps(identity,indent=2))

runpy.run_module('qwen_vl.train.train_qwen',run_name='__main__')
identity['completed']=True
(root/f'identity_rank{rank}.json').write_text(json.dumps(identity,indent=2))
