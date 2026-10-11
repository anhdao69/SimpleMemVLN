"""The public numerical checker must enforce its helper-state comparison."""
import json
import sys
from unittest.mock import patch
import pytest
import torch
from scripts.vln import check_episodic as check


@pytest.mark.parametrize('finite,error,expected', [(False,.002,1),(True,100.,1),(True,float('nan'),1),(True,.002,0)])
def test_helper_failure_changes_command_exit(tmp_path, finite, error, expected):
    model = torch.nn.Linear(1,1)
    out = tmp_path/'result.json'
    argv = ['check', '--model-path','unused','--manifest','unused','--out',str(out),'--replay-steps','96']
    with patch.object(sys,'argv',argv), patch.multiple(check,
            load_model=lambda *a:(model,None), fixture=lambda *a,**k:{},
            gate_a=lambda *a:dict(logits_bitwise_equal=True,loss_sum_equal=True),
            gate_b=lambda *a:dict(finite=finite,final_state_relative_error_vs_fla=error),
            gate_c=lambda *a:dict(passed=True),
            gate_e=lambda *a:dict(peak_resident_kv=1,guard=2)), pytest.raises(SystemExit) as result:
        check.main()
    assert result.value.code == expected
    assert json.loads(out.read_text())['passed'] == (expected == 0)
