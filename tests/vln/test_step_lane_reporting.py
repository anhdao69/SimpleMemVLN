import json
import torch
from torch import nn
from types import SimpleNamespace


def test_parameter_summaries_are_detached_and_labelled():
    from qwen_vl.train.lane_reporting import lane_parameter_summary
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.step_lane = nn.Module()
    lane = model.layer.step_lane
    lane.A_log = nn.Parameter(torch.zeros(4))
    lane.dt_bias = nn.Parameter(torch.tensor([-2.,-3.,-4.,-5.]))
    lane.in_proj_b = nn.Linear(3,4)
    lane.out_proj = nn.Linear(4,3, bias=False)
    summary = lane_parameter_summary(model)
    assert summary['layer.step_lane']['gate_context'] == 'zero-input reference, not observed writer gates'
    assert all(v > 0 for v in summary['layer.step_lane']['zero_input_decay_half_life_events'])
    json.dumps(summary)
    assert all(p.grad is None for p in model.parameters())


def test_observed_diagnostics_are_bounded_and_hooks_leave_outputs_unchanged():
    from qwen_vl.train.lane_reporting import ObservedLaneDiagnostics
    from qwen_vl.models.parallel_step_lane import ParallelStepLane
    from step_lane_test_utils import reference_kernel,tiny_spec
    class Native(nn.Module):
        def __init__(self):
            super().__init__()
            self.step_lane=ParallelStepLane(4,tiny_spec(),device='cpu',dtype=torch.float32,
                chunk_kernel=reference_kernel,recurrent_kernel=reference_kernel,test_only=True)
        def forward(self,x,roles,state):
            branch,final=self.step_lane(x,roles,state,return_final_state=True)
            return x+branch,final
    model=nn.Module()
    model.native=Native()
    model.eval()
    x=torch.randn(1,3,4)
    roles=torch.tensor([[1,2,3]],dtype=torch.uint8)
    state=torch.zeros(1,2,3,5)
    with torch.inference_mode():
        baseline=model.native(x,roles,state)
        with ObservedLaneDiagnostics(model,max_records=1) as diagnostics:
            actual=model.native(x,roles,state)
            model.native(x,roles,state)
        torch.testing.assert_close(actual[0],baseline[0],rtol=0,atol=0)
        torch.testing.assert_close(actual[1],baseline[1],rtol=0,atol=0)
    assert len(diagnostics.records)==1
    row=diagnostics.records[0]
    assert len(row['writer_alpha'][0])==2 and len(row['state_norm_per_head'])==2
    assert row['lane_native_output_rms_ratio']==0
    assert row['write_magnitude_per_head'] and row['writer_count']==1
    json.dumps(row)
    assert not model.native._forward_hooks and not model.native.step_lane._forward_hooks
