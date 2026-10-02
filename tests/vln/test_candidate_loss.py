import math
import pytest
import torch


@pytest.mark.parametrize("mode", ["none", "sqrt_inverse_frequency", "effective_number"])
def test_distribution_weight_normalization(mode):
    from qwen_vl.models.action_loss import class_weights

    counts = torch.tensor([400.0, 100.0, 80.0, 10.0], dtype=torch.float64)
    w = class_weights(counts.tolist(), mode)
    torch.testing.assert_close(
        (counts / counts.sum() * w).sum(), torch.tensor(1.0, dtype=torch.float64)
    )
    if mode != "none":
        assert w[-1] > w[0] and not torch.isclose(
            w.mean(), torch.tensor(1.0, dtype=torch.float64)
        )


def test_weighted_ce_uses_global_action_denominator():
    from qwen_vl.models.action_loss import class_weights, action_losses
    from qwen_vl.train.trainer import QwenSFTTrainer
    from types import SimpleNamespace

    w = class_weights([8, 2, 2, 1], "sqrt_inverse_frequency")
    logits = torch.randn(13, 4, requires_grad=True)
    target = torch.tensor([0] * 8 + [1] * 2 + [2] * 2 + [3])
    ce, weighted = action_losses(logits, target, w)
    reference = (
        torch.nn.functional.cross_entropy(logits, target, reduction="none")
        * w.float()[target]
    ).mean()
    torch.testing.assert_close(weighted.sum() / 13, reference)
    trainer = object.__new__(QwenSFTTrainer)
    trainer.accelerator = SimpleNamespace(num_processes=2)

    class Model(torch.nn.Module):
        def forward(self, **kw):
            return {"loss_sum": kw["loss_sum"]}

    model = Model()
    # Simulated rank averaging, unequal action counts and a partial GAS tail.
    loss = (
        sum(
            trainer.compute_loss(
                model,
                {"num_actions": len(part), "loss_sum": part.sum()},
                num_items_in_batch=13,
            )
            for part in (weighted[:4], weighted[4:7], weighted[7:])
        )
        / 2
    )
    torch.testing.assert_close(loss, reference)
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits, retain_graph=True)[0],
        torch.autograd.grad(reference, logits)[0],
    )
    _, uniform = action_losses(torch.zeros(13, 4), target, w)
    torch.testing.assert_close(uniform.mean(), torch.tensor(math.log(4)))
    with pytest.raises(ValueError):
        class_weights([10, 2, 1, 0], "sqrt_inverse_frequency")
    with pytest.raises(ValueError):
        class_weights([10, 2, 1, 1], "effective_number", beta=1.0)


def test_all_forward_collapse_visible():
    from qwen_vl.eval.action_metrics import classification_metrics

    report = classification_metrics(
        [[8, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [1, 0, 0, 0]]
    )
    assert report["overall_accuracy"] == 8 / 13
    assert report["macro_accuracy"] == 0.25
    assert report["per_class"]["STOP"]["recall"] == 0
    assert report["predicted_action_counts"]["MOVE_FORWARD"] == 13
    assert report["ground_truth_action_counts"]["STOP"] == 1


def test_direct_rows_use_backbone_lr_only():
    from test_candidate_readout import tiny
    from qwen_vl.train.trainer import QwenSFTTrainer
    from transformers import TrainingArguments

    for mode in ("lm_rows_trainable", "copied_linear"):
        model = tiny(mode)
        args = TrainingArguments(
            output_dir="/tmp/candidate-optimizer-test",
            use_cpu=True,
            report_to="none",
            learning_rate=5e-6,
        )
        args.mm_projector_lr = 5e-5
        trainer = QwenSFTTrainer(model=model, args=args)
        opt = trainer.create_optimizer()
        rates = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
        assert rates[id(model.backbone.lm_head.weight)] == 5e-6
        if mode == "copied_linear":
            assert rates[id(model.classifier.weight)] == 5e-5
