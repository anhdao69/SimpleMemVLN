"""Real pretrained temporal-gradient and checkpoint recomputation check."""
import argparse, json
from pathlib import Path
import torch
from qwen_vl.train.vln_runtime import resolve_config, load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--memory-mode", default="full_context")
    p.add_argument("--out", default="artifacts/gradients.json")
    args = p.parse_args()
    cfg = resolve_config(
        "configs/vln_r2r_v0_base.yaml",
        "configs/vln_r2r_v0_classification.yaml",
        "configs/vln_memory_window8.yaml" if args.memory_mode == "window8" else None,
    )
    torch.manual_seed(429)
    model, serializer = load_model(cfg, args.model_path)
    model.train()
    episode = min(
        [json.loads(l) for l in Path(args.manifest).read_text().splitlines()],
        key=lambda e: len(e["steps"]),
    )
    data = {
        k: v.cuda() if torch.is_tensor(v) else v
        for k, v in serializer.encode_episode(episode).items()
    }
    embed = model.embed
    captures = []

    def capture(*a, **kw):
        result = embed(*a, **kw)
        result.retain_grad()
        captures.append(result)
        return result

    model.embed = capture
    names = [
        "classifier.weight",
        "backbone.model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "backbone.model.language_model.layers.3.self_attn.q_proj.weight",
    ]
    named = dict(model.named_parameters())
    snapshots = []
    report = []
    for checkpointed in [False, False, True, True]:
        if checkpointed:
            model.gradient_checkpointing_enable({"use_reentrant": False})
        model.zero_grad(set_to_none=True)
        output = model(**data)
        # Only final STOP contributes, so early source gradients prove time credit.
        loss = torch.nn.functional.cross_entropy(
            output["logits"][-1:].float(), data["action_class_ids"][-1:]
        )
        loss.backward()
        prefix, spans = data["step_plan"]
        early = captures[-1].grad[0, spans[0][0] : spans[0][1]]
        assert (
            early is not None and torch.isfinite(early).all() and early.abs().sum() > 0
        )
        assert all(p.grad is None for p in model.backbone.model.visual.parameters())
        assert not model.backbone.model.visual.training
        gradients = {
            n: named[n].grad.detach().float().cpu().clone() for n in names if n in named
        }
        assert len(gradients) == 3
        assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients.values())
        snapshots.append(gradients)
        report.append(
            dict(
                checkpointed=checkpointed,
                loss=float(loss.detach()),
                early_source_gradient_norm=float(early.float().norm()),
                gradient_norms={n: float(g.norm()) for n, g in gradients.items()},
            )
        )
        captures.clear()
    comparisons = []
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        for name in snapshots[0]:
            x, y = snapshots[a][name].double(), snapshots[b][name].double()
            comparisons.append(
                dict(
                    a=a,
                    b=b,
                    name=name,
                    max_error=float((x - y).abs().max()),
                    relative_l2=float((x - y).norm() / x.norm()),
                    cosine=float(
                        torch.nn.functional.cosine_similarity(
                            x.flatten(), y.flatten(), dim=0
                        )
                    ),
                )
            )
    # Measured repeatability controls distinguish BF16 backward reduction noise
    # from checkpoint recomputation errors. The original elementwise .002/.03
    # gate failed even between two uncheckpointed backward executions.
    passed = all(c["relative_l2"] <= 0.03 and c["cosine"] >= 0.999 for c in comparisons)
    passed = passed and len({r["loss"] for r in report}) == 1
    Path(args.out).write_text(
        json.dumps(
            dict(
                passed=passed,
                memory_mode=args.memory_mode,
                tolerance=dict(
                    relative_l2=0.03, min_cosine=0.999, forward_loss="exact"
                ),
                results=report,
                comparisons=comparisons,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))
    print(json.dumps(comparisons, indent=2))
    assert passed, "Gradient discrepancy exceeds measured repeatability budget"


if __name__ == "__main__":
    main()
