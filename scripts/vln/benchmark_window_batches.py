"""Matched FA2 Window8 forward/backward microbenchmark; no speed extrapolation."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from qwen_vl.stream.window_attention import step_flash_attention


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True)
    args=p.parse_args(); torch.manual_seed(429); records=[]
    for count in (64,183,627):
        prefix,size=128,320; length=prefix+count*size
        spans=tuple((prefix+i*size,prefix+(i+1)*size) for i in range(count))
        xs=[torch.randn(1,h,length,256,device='cuda',dtype=torch.bfloat16,requires_grad=True) for h in (16,4,4)]
        # Qwen3.5-4B full-attention heads have head_dim 256.
        for batch in (1,4,16,32):
            times=[]
            for repeat in range(3):
                for x in xs: x.grad=None
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                y=step_flash_attention(None,*xs,None,scaling=256**-.5,step_plan=(prefix,spans),window_attention_batch_steps=batch)[0]
                y.backward(torch.full_like(y,1/y.numel()))
                torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                if repeat: times.append(elapsed)
                peak=torch.cuda.max_memory_allocated()/2**30
                del y
            row=dict(groups=count,batch_steps=batch,seconds=statistics.median(times),peak_allocated_gib=peak)
            records.append(row); print(json.dumps(row),flush=True)
        del xs; torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(records,indent=2)+'\n')


if __name__=='__main__': main()
