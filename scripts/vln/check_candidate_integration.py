"""Real-model history replay, causality, reset/retry and temporal-gradient gate."""

import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from qwen_vl.contracts import ACTIONS
from qwen_vl.train.vln_runtime import load_model, load_checkpoint, resolve_config
from qwen_vl.stream.session import StreamSession
from qwen_vl.stream.cache import assert_state_dtypes, kv_length


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--memory", choices=["full_context", "window8"], required=True)
    p.add_argument(
        "--feedback",
        choices=["candidate_token", "canonical_action_text", "none"],
        required=True,
    )
    p.add_argument("--checkpoint")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = resolve_config(
        "configs/vln_r2r_v0_base.yaml",
        "configs/vln_r2r_v0_candidate_logits.yaml",
        "configs/vln_memory_window8.yaml" if args.memory == "window8" else None,
    )
    cfg["observations"]["feedback_format"] = args.feedback
    if args.feedback == "none":
        cfg["observations"]["append_action_tokens"] = False
        cfg["observations"]["serializer_version"] = "vln_candidate_logits_no_action_history_v1"
    cfg["runtime"]["action_diagnostics"] = True
    model, s = (
        load_checkpoint(args.checkpoint, args.model_path)
        if args.checkpoint
        else load_model(cfg, args.model_path)
    )
    assert s.mode == "candidate_logits"
    assert model.navigation_config["memory"]["mode"] == args.memory
    assert model.navigation_config["observations"]["feedback_format"] == args.feedback
    model.navigation_config["runtime"]["action_diagnostics"] = True
    episodes = [
        json.loads(x) for x in Path(args.manifest).read_text().splitlines() if x.strip()
    ]
    ep = min(
        (e for e in episodes if len(e["steps"]) >= 12), key=lambda e: len(e["steps"])
    )
    device = next(model.parameters()).device
    data = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in s.encode_episode(ep).items()
    }
    model.train()
    model.gradient_checkpointing_enable()
    captured = []
    embed = model.embed

    def capture_embed(*a, **kw):
        x = embed(*a, **kw)
        x.retain_grad()
        captured.append(x)
        return x

    model.embed = capture_embed
    output = model(**data)
    initial_ce = float(output["unweighted_loss_sum"] / len(ep["steps"]))
    loss = torch.nn.functional.cross_entropy(
        output["logits"][-1:].float(), data["action_class_ids"][-1:]
    )
    loss.backward()
    start, end = data["step_plan"][1][0]
    early = float(captured[0].grad[0, start:end].float().norm())
    assert early > 0 and all(
        p.grad is None for p in model.backbone.model.visual.parameters()
    )
    named = dict(model.named_parameters())
    gradients = {
        n: float(p.grad.float().norm())
        for n, p in named.items()
        if p.grad is not None
        and (
            ".layers.0.linear_attn.in_proj_b.weight" in n
            or ".layers.3.self_attn.q_proj.weight" in n
        )
    }
    assert len(gradients) == 2 and all(v > 0 for v in gradients.values())
    assert not model.backbone.model.visual.training
    model.embed = embed
    model.zero_grad(set_to_none=True)
    captured.clear()
    del output, data, loss
    model.eval()
    session = StreamSession(model, s)
    blocks = []
    reads = []
    scores = []
    spans = []
    append = session._append
    readout = model.action_logits

    def capture_block(block):
        result = append(block)
        blocks.append(block)
        return result

    def capture_logits(h):
        out = readout(h)
        reads.append(session.positions.logical_token_count - 1)
        scores.append(out.detach().clone())
        return out

    session._append = capture_block
    model.action_logits = capture_logits

    def forbidden(*a, **kw):
        raise AssertionError("Full vocabulary head called during candidate inference")

    original_head = model.backbone.lm_head.forward
    model.backbone.lm_head.forward = forbidden
    with torch.inference_mode():
        session.reset("candidate-integration", ep["instruction"])
        prefix = session.prefix_length
        results = []
        for step in range(12):
            start = session.positions.logical_token_count
            with Image.open(ep["steps"][step]["rgb_path"]) as rgb:
                result = session.observe("candidate-integration", step, rgb)
            assert result["action_name"] in ACTIONS and result["generated_tokens"] == 0
            if args.feedback == "none":
                assert result["feedback_token_ids"] == [s.eos] + s.separator
            assert session.observe("candidate-integration", step, None) is result
            assert_state_dtypes(session.cache)
            spans.append((start, session.positions.logical_token_count))
            results.append(result)
        if args.memory == "window8":
            assert list(session.resident) == [
                (i, spans[i][1] - spans[i][0]) for i in range(4, 12)
            ]
            assert kv_length(session.cache) == prefix + sum(
                e - b for b, e in spans[-8:]
            )
        else:
            assert kv_length(session.cache) == session.positions.logical_token_count
        data = {
            k: torch.cat(
                [b[k] for b in blocks if k in b],
                dim=1 if k in ("input_ids", "mm_token_type_ids") else 0,
            ).to(device)
            for k in (
                "input_ids",
                "mm_token_type_ids",
                "pixel_values",
                "image_grid_thw",
            )
        }
        ref = readout(
            model.hidden(**data, step_plan=(prefix, tuple(spans)))[0, reads]
        ).float()
        actual = torch.stack(scores).float()
        diff = (ref - actual).abs()
        margin = ref.topk(2, -1).values.diff(dim=-1).abs().squeeze(-1)
        flips = ref.argmax(-1) != actual.argmax(-1)
        torch.testing.assert_close(actual, ref, atol=0.75, rtol=0.03)
        assert float(diff.square().mean().sqrt()) <= 0.15
        assert not (flips & (margin > 2 * diff.max(-1).values)).any()
        # Fresh reset removes old episode, KV, GDN and position state.
        session.reset("fresh", ep["instruction"])
        assert (
            session.next_step == 0
            and not session.resident
            and session.last_result is None
        )
        assert (
            session.positions.logical_token_count == prefix
            and kv_length(session.cache) == prefix
        )
        with Image.open(ep["steps"][0]["rgb_path"]) as rgb:
            fresh = session.observe("fresh", 0, rgb)
        torch.testing.assert_close(
            torch.tensor(fresh["action_logits"]),
            torch.tensor(results[0]["action_logits"]),
            rtol=0,
            atol=0,
        )
    model.backbone.lm_head.forward = original_head
    report = dict(
        passed=True,
        memory=args.memory,
        feedback=args.feedback,
        initial_unweighted_ce=initial_ce,
        early_source_gradient_norm=early,
        gradient_norms=gradients,
        steps=12,
        max_logit_error=float(diff.max()),
        rms_logit_error=float(diff.square().mean().sqrt()),
        argmax_agreement=float((~flips).float().mean()),
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        note="Runtime/parity smoke; no navigation score or convergence claim",
    )
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
