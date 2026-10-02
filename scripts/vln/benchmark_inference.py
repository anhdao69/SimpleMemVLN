"""Measure complete per-action policy inference on one repeated Habitat RGB frame.

Run one process/model at a time. Reset and warmup are outside the measured window.
This is a controlled serving-speed test, not a navigation rollout.
"""

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from PIL import Image


def quantile(values, fraction):
    return float(np.quantile(np.asarray(values, dtype=np.float64), fraction))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("stage", "simple"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--history", choices=("uniform8", "recent"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--components",
        action="store_true",
        help="Instrumented run; not headline latency",
    )
    args = parser.parse_args()
    if args.warmup < 9 or args.steps < 32:
        parser.error("warmup must be at least 9, and measured steps at least 32")
    if args.family == "stage" and not args.history:
        parser.error("Stage policy needs --history")
    if args.family == "simple" and not args.base_model:
        parser.error("SimpleMemVLN needs --base-model")
    import torch
    import transformers

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_device(0)
    if args.family == "stage":
        from qwen_vl.train.train_qwen import _install_qwen35_flash_attention_fix
        from qwen_vl.eval.uniform_session import UniformHistorySession
        from qwen_vl.data.data_qwen import QWEN3_5_NON_THINKING_CHAT_TEMPLATE
        from qwen_vl.data.history import history_indices

        _install_qwen35_flash_attention_fix()
        processor = transformers.AutoProcessor.from_pretrained(args.checkpoint)
        processor.tokenizer.chat_template = QWEN3_5_NON_THINKING_CHAT_TEMPLATE
        backbone = (
            transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
                args.checkpoint,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            .to("cuda:0")
            .eval()
        )
        policy = UniformHistorySession(
            backbone, processor, max_new_tokens=32, history_mode=args.history
        )
        mode = args.history
    else:
        from qwen_vl.train.vln_runtime import load_checkpoint
        from qwen_vl.stream.session import StreamSession

        backbone, serializer = load_checkpoint(args.checkpoint, args.base_model)
        if serializer.mode not in ("qwen_text", "classification", "candidate_logits"):
            raise ValueError("Unsupported navigation output")
        backbone.navigation_config["runtime"]["profile_components"] = args.components
        policy = StreamSession(backbone, serializer)
        mode = backbone.navigation_config["memory"]["mode"]
        if mode not in ("full_context", "window8"):
            raise ValueError(f"Unsupported memory mode {mode}")
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected original uint8 Habitat RGB 640x480")
    instruction = args.instruction.strip()
    uid = "inference-speed-repeated-frame"
    policy.reset(uid, instruction)
    measurements = []
    for step in range(args.warmup + args.steps):
        if args.family == "stage":
            selected = history_indices(step, mode, recent=4)
            expected = 9 if mode == "uniform8" else 5
            if step >= args.warmup and len(selected) != expected:
                raise AssertionError("Selected history frame count changed")
        torch.cuda.synchronize()
        begin = time.perf_counter()
        if args.family == "stage":
            action = policy.observe(step, rgb)
            token_count = int(policy.last_token_ids.numel())
        else:
            result = policy.observe(uid, step, rgb)
            action = result["action_name"]
            token_count = len(result.get("response_token_ids", []))
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        if step >= args.warmup:
            entry = dict(
                step=step, seconds=seconds, action=action, generated_tokens=token_count
            )
            if args.family == "simple":
                entry["retained_kv_tokens"] = result["retained_kv_tokens"]
                if "component_seconds" in result:
                    entry["component_seconds"] = result["component_seconds"]
            measurements.append(entry)
    durations = [r["seconds"] for r in measurements]
    report = dict(
        family=args.family,
        mode=mode,
        instrumented_components=args.components,
        checkpoint=str(Path(args.checkpoint).resolve()),
        image=str(Path(args.image).resolve()),
        instruction=instruction,
        warmup=args.warmup,
        measured_actions=len(measurements),
        total_episode_steps=args.warmup + args.steps,
        test_kind="repeated_real_habitat_frame_not_navigation",
        gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
        python=platform.python_version(),
        torch=torch.__version__,
        transformers=transformers.__version__,
        allocator=os.environ.get("PYTORCH_ALLOC_CONF", ""),
        median_seconds=statistics.median(durations),
        mean_seconds=statistics.fmean(durations),
        p95_seconds=quantile(durations, 0.95),
        tokens_per_action=statistics.fmean(r["generated_tokens"] for r in measurements),
        peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
        records=measurements,
    )
    if args.family == "simple":
        report["output_mode"] = serializer.mode
        report["feedback_format"] = backbone.navigation_config["observations"].get(
            "feedback_format"
        )
    if args.components:
        names = set().union(
            *(r.get("component_seconds", {}).keys() for r in measurements)
        )
        report["component_mean_seconds"] = {
            name: statistics.fmean(
                r.get("component_seconds", {}).get(name, 0.0) for r in measurements
            )
            for name in sorted(names)
        }
        report["component_note"] = (
            "Synchronized diagnostic run; history timer includes feedback language append. Do not sum nested timers or compare to uninstrumented headline latency."
        )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
