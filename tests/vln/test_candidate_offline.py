import math
import pytest
import torch


def test_offline_summary_uses_action_count_not_episode_mean():
    from scripts.vln.evaluate_actions import summarize_outputs

    outputs = [
        dict(
            predictions=torch.tensor([0, 0, 0]),
            targets=torch.tensor([0, 1, 2]),
            loss_sum=3 * math.log(4),
        ),
        dict(
            predictions=torch.tensor([3]),
            targets=torch.tensor([3]),
            loss_sum=math.log(4),
        ),
    ]
    result = summarize_outputs(outputs)
    assert result["actions"] == 4
    assert result["unweighted_ce"] == pytest.approx(math.log(4))
    assert result["overall_accuracy"] == 0.5
    assert result["per_class"]["STOP"]["recall"] == 1
    assert result["confusion"] == [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
