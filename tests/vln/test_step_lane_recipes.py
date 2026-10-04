import pytest
from qwen_vl.train.vln_runtime import resolve_config


@pytest.mark.parametrize('recipe',['configs/vln_dual_full_no_history_r2r.yaml','configs/vln_dual_window8_no_history_r2r.yaml','configs/vln_dual_full_no_history_joint.yaml','configs/vln_dual_window8_no_history_joint.yaml'])
def test_committed_recipe_learning_rates_are_numbers(recipe):
    cfg=resolve_config(recipe,'configs/vln_empty.yaml')
    for key in ['backbone_lr','classifier_lr','step_lane_lr']:
        assert isinstance(cfg['training'][key],float)
        assert cfg['training'][key]>0
