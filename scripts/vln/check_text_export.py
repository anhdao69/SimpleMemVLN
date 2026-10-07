"""Verify trained lane weights survive strict export loading and multi-step use."""
import argparse
import json
from pathlib import Path
import torch
from qwen_vl.train.vln_runtime import load_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True); p.add_argument('--model-path',required=True)
    p.add_argument('--manifest',required=True); p.add_argument('--out',required=True)
    args=p.parse_args()
    model,serializer=load_checkpoint(args.checkpoint,args.model_path); model.eval()
    assert model.output_mode=='qwen_text' and model.step_lane_spec is not None
    saved=torch.load(Path(args.checkpoint)/'pytorch_model.bin',map_location='cpu',weights_only=True,mmap=True)
    count=0; norms={}
    for name,param in model.named_parameters():
        if '.step_lane.' not in name: continue
        assert torch.equal(param.detach().cpu(),saved[name]),name
        assert torch.isfinite(param).all(),name
        count+=param.numel()
        if name.endswith('out_proj.weight'):
            norms[name]=float(param.float().norm()); assert norms[name]>0
    assert count==21_054_000 and len(norms)==4
    del saved
    episodes={}
    for line in Path(args.manifest).open():
        ep=json.loads(line)
        if len(ep['steps'])>8: episodes.setdefault(ep['dataset'],ep)
    assert len(episodes)==2
    losses={}
    with torch.inference_mode():
        for dataset,ep in episodes.items():
            data={k:v.cuda() if torch.is_tensor(v) else v for k,v in serializer.encode_episode(ep).items()}
            result=model(**data)
            assert torch.isfinite(result['logits']).all() and torch.isfinite(result['loss_sum'])
            losses[dataset]=float(result['loss_sum'])
    report=dict(passed=True,lane_parameters_exact=count,trained_out_proj_norms=norms,multistep_loss_sums=losses)
    Path(args.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__': main()
