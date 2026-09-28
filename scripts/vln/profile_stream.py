"""Dry streaming resource/lifecycle check, NOT a simulator navigation result."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from qwen_vl.train.vln_runtime import resolve_config, load_model
from qwen_vl.stream.session import StreamSession


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--memory-mode", choices=["full_context", "window8"], required=True)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = resolve_config(
        "configs/vln_r2r_v0_base.yaml",
        "configs/vln_r2r_v0_classification.yaml",
        "configs/vln_memory_window8.yaml" if args.memory_mode == "window8" else None,
    )
    torch.manual_seed(429)
    model, serializer = load_model(cfg, args.model_path)
    ep = min(
        (json.loads(l) for l in Path(args.manifest).read_text().splitlines()),
        key=lambda e: len(e["steps"]),
    )
    session = StreamSession(model, serializer)
    torch.cuda.reset_peak_memory_stats()
    session.reset("dry-profile", ep["instruction"])
    records = []
    for step in range(args.steps):
        # Real RGBs are cyclically replayed solely to exercise the full horizon.
        with Image.open(ep["steps"][step % len(ep["steps"])]["rgb_path"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        result = session.observe("dry-profile", step, rgb)
        assert session.observe("dry-profile", step, rgb) is result
        assert session.next_step == step + 1
        if args.memory_mode == "window8":
            assert len(session.resident) <= 8
        else:
            assert not session.resident
        records.append(
            {**result, "logical_tokens": session.positions.logical_token_count}
        )
        if (step + 1) % 50 == 0:
            print("STREAM_PROGRESS", step + 1, flush=True)
    # Reset must match a fresh session, including all recurrent/conv state.
    with Image.open(ep["steps"][0]["rgb_path"]) as im:
        rgb = np.asarray(im.convert("RGB"))
    session.reset("reset-check", ep["instruction"])
    reset = session.observe("reset-check", 0, rgb)
    torch.testing.assert_close(
        torch.tensor(reset["logits"]),
        torch.tensor(records[0]["logits"]),
        atol=0,
        rtol=0,
    )
    report = dict(
        passed=True,
        kind="cyclic_real_rgb_dry_replay_not_navigation",
        memory_mode=args.memory_mode,
        steps=args.steps,
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        peak_kv_tokens=max(r["retained_kv_tokens"] for r in records),
        final_logical_tokens=records[-1]["logical_tokens"],
        age_buckets=[],
        records=records,
    )
    for start in range(0, args.steps, 100):
        times = [r["model_seconds"] for r in records[start : start + 100]]
        report["age_buckets"].append(
            dict(
                first_step=start,
                count=len(times),
                model_latency_p50_seconds=float(np.quantile(times, 0.5)),
                model_latency_p95_seconds=float(np.quantile(times, 0.95)),
            )
        )
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
