"""Replay the exact tokens committed by the real greedy B session offline."""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
from qwen_vl.train.vln_runtime import resolve_config, load_model, load_checkpoint
from qwen_vl.stream.session import StreamSession
from qwen_vl.stream.positions import PositionLedger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument(
        "--memory-mode", choices=["full_context", "window8"], default="window8"
    )
    p.add_argument("--out", default="artifacts/generated_history.json")
    p.add_argument("--checkpoint")
    args = p.parse_args()
    cfg = resolve_config(
        "configs/vln_r2r_v0_base.yaml",
        "configs/vln_r2r_v0_qwen_text.yaml",
        "configs/vln_memory_window8.yaml" if args.memory_mode == "window8" else None,
    )
    torch.manual_seed(429)
    model, serializer = (
        load_checkpoint(args.checkpoint, args.model_path)
        if args.checkpoint
        else load_model(cfg, args.model_path)
    )
    if (
        model.output_mode != "qwen_text"
        or model.navigation_config["memory"]["mode"] != args.memory_mode
    ):
        raise ValueError("Generated-history checkpoint/mode mismatch")
    session = StreamSession(model, serializer)
    ep = min(
        (json.loads(l) for l in Path(args.manifest).read_text().splitlines()),
        key=lambda e: len(e["steps"]),
    )
    blocks = []
    generated_logits = []
    reads = []
    spans = []
    results = []
    original = session._append

    def capture(block):
        result = original(block)
        blocks.append(block)
        return result

    session._append = capture

    def capture_prediction(module, inputs, output):
        # Only actual body/EOS prediction positions, not prefix/newline outputs
        # that the decoder never projects or supervises.
        reads.append(session.positions.logical_token_count - 1)
        generated_logits.append(output.detach())

    hook = model.backbone.lm_head.register_forward_hook(capture_prediction)
    with torch.inference_mode():
        session.reset("generated-history", ep["instruction"])
        prefix = session.positions.logical_token_count
        for step in range(12):
            start = session.positions.logical_token_count
            try:
                with Image.open(ep["steps"][step]["rgb_path"]) as rgb:
                    result = session.observe("generated-history", step, rgb)
                assert session.observe("generated-history", step, None) is result
            except Exception as exc:
                report = dict(
                    passed=False,
                    reason=f"{type(exc).__name__}: {exc}",
                    completed_steps=len(results),
                    results=results,
                    note="Untrained response invalidity is not replaced with STOP",
                )
                Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
                raise
            spans.append((start, session.positions.logical_token_count))
            results.append(result)
        hook.remove()
        data = {
            k: torch.cat(
                [b[k] for b in blocks if k in b],
                dim=1 if k in ("input_ids", "mm_token_type_ids") else 0,
            ).cuda()
            for k in (
                "input_ids",
                "mm_token_type_ids",
                "pixel_values",
                "image_grid_thw",
            )
        }
        positions = PositionLedger().append(
            model.backbone.model,
            data["input_ids"],
            data["mm_token_type_ids"],
            data["image_grid_thw"],
        )
        offline = model.hidden(
            **data, position_ids=positions, step_plan=(prefix, tuple(spans))
        )
        ref = model.backbone.lm_head(offline[0, reads]).float()
        actual = torch.stack(generated_logits).float()
        diff = (actual - ref).abs()
        margins = ref.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
        flips = actual.argmax(-1) != ref.argmax(-1)
        report = dict(
            passed=bool(
                torch.allclose(actual, ref, atol=0.75, rtol=0.03)
                and diff.square().mean().sqrt() <= 0.1
            ),
            memory_mode=args.memory_mode,
            completed_steps=len(results),
            committed_tokens=data["input_ids"].numel(),
            atol=0.75,
            rtol=0.03,
            max_rms=0.1,
            max_logit_error=float(diff.max()),
            rms_logit_error=float(diff.square().mean().sqrt()),
            argmax_agreement=float((~flips).float().mean()),
            high_margin_flips=int((flips & (margins > 2 * diff.max(-1).values)).sum()),
            results=results,
        )
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
        assert report["passed"] and not report["high_margin_flips"]


if __name__ == "__main__":
    main()
