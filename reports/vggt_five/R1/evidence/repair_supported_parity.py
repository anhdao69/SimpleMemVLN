"""Campaign repair audit: exact old/new real-FLA lane outputs and gradients."""
import argparse, importlib.util, json, hashlib
from pathlib import Path
import torch
from qwen_vl.models.parallel_step_lane import ParallelStepLane
from qwen_vl.models.vggt_g3_heads import NativeStateHead
from qwen_vl.models.vggt_r1_forward import lane_forward_with_queries
import qwen_vl.models.vggt_r1_forward as current
from qwen_vl.research.lane_contract import StepLaneSpec

p=argparse.ArgumentParser();p.add_argument('--old-source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
path=Path(a.old_source)/'src/qwen_vl/models/vggt_r1_forward.py'
spec=importlib.util.spec_from_file_location('qualified_r1_old',path)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
torch.manual_seed(429)
lane=ParallelStepLane(2560,StepLaneSpec(),device='cuda',dtype=torch.bfloat16,lane_seed=453)
head=NativeStateHead(heads=4,context_heads=32).cuda().bfloat16()
with torch.no_grad():lane.out_proj.weight.normal_(0,.02)
x=torch.randn(1,137,2560,device='cuda',dtype=torch.bfloat16)
roles=torch.ones((1,137),device='cuda',dtype=torch.uint8);roles[:,[31,63,95,127]]=3
records=[]
for count in (2,0):
    anchors=torch.tensor([60,100],device='cuda')[:count]
    lags=torch.tensor([8,32],device='cuda')[:count]
    context=torch.randn(count,32,128,device='cuda',dtype=torch.bfloat16)
    targets=torch.randn(count,5,2048,device='cuda')
    outcomes=[]
    for forward in (old.lane_forward_with_queries,lane_forward_with_queries):
        lane.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
        leaf=x.clone().requires_grad_()
        out,loss=forward(lane,leaf,roles,head,context,anchors,lags,targets)
        (out.float().square().mean()+loss).backward()
        gradients=[leaf.grad]+[p.grad for p in lane.parameters()]+[p.grad for p in head.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        outcomes.append((out.detach().clone(),loss.detach().clone(),[g.clone() for g in gradients]))
    torch.testing.assert_close(outcomes[0][0],outcomes[1][0],atol=0,rtol=0)
    # The uniform empty loss is FP32 (as supported losses already are); both
    # scalar values and all gradients must still be exactly zero/equal.
    torch.testing.assert_close(outcomes[0][1].float(),outcomes[1][1].float(),atol=0,rtol=0)
    for first,last in zip(outcomes[0][2],outcomes[1][2]):
        torch.testing.assert_close(first,last,atol=0,rtol=0)
    records.append(dict(pairs=count,outputs_exact=True,loss_exact=True,all_gradients_exact=True,
                        gradient_tensors=len(outcomes[0][2]),
                        old_loss_dtype=str(outcomes[0][1].dtype),new_loss_dtype=str(outcomes[1][1].dtype)))
result=dict(passed=True,old_source=a.old_source,cases=records,
            old_module_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            new_module_path=str(Path(current.__file__).resolve()),
            new_module_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest())
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print('R1_REPAIR_PARITY_PASS',json.dumps(result))
