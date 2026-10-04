import torch
from torch import nn
from transformers import TrainingArguments
from qwen_vl.train.trainer import QwenSFTTrainer


def test_lane_lr_decay_and_exact_unique_coverage():
    class Lane(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(5, 5)
            self.norm = nn.LayerNorm(5)
            self.A_log = nn.Parameter(torch.zeros(4))
            self.dt_bias = nn.Parameter(torch.zeros(4))
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.native = nn.Linear(5, 5)
    model.backbone.step_lane = Lane()
    model.classifier = nn.Linear(5, 4)
    model.frozen = nn.Linear(5, 2).requires_grad_(False)
    model.navigation_config = {'training': {'step_lane_lr': 1e-4}}
    args = TrainingArguments(output_dir='/tmp/lane-opt-test', use_cpu=True, report_to='none', learning_rate=5e-6, weight_decay=0.01)
    args.mm_projector_lr = 5e-5
    trainer = QwenSFTTrainer(model=model, args=args)
    optimizer = trainer.create_optimizer()
    assigned = {id(p): group for group in optimizer.param_groups for p in group['params']}
    assert len(assigned) == sum(len(g['params']) for g in optimizer.param_groups)
    assert set(assigned) == {id(p) for p in model.parameters() if p.requires_grad}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = assigned[id(p)]
        assert group['lr'] == (1e-4 if '.step_lane.' in name else 5e-5 if name.startswith('classifier.') else 5e-6)
        assert group['weight_decay'] == (0.01 if p.ndim > 1 and 'norm' not in name else 0)
