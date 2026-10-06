"""Single-session JSON-lines model subprocess; RGB transport is lossless raw bytes or PNG."""
import argparse
import contextlib
import json
import os
import sys
from qwen_vl.stream.transport import decode_rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-path")
    args = p.parse_args()
    protocol_stdout = sys.stdout
    # Library progress/diagnostics must never corrupt the protocol stream.
    with contextlib.redirect_stdout(sys.stderr):
        from qwen_vl.train.vln_runtime import load_checkpoint
        from qwen_vl.stream.session import StreamSession
        import torch

        model, serializer = load_checkpoint(args.checkpoint, args.model_path)
        session = StreamSession(model, serializer)
    import platform
    import transformers
    import importlib.metadata
    runtime = dict(status="ready", python=platform.python_version(), torch=torch.__version__,
                   transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
                   capability=torch.cuda.get_device_capability(),
                   flash_attn=importlib.metadata.version("flash-attn"),
                   fla=importlib.metadata.version("flash-linear-attention"),
                   output_mode=serializer.mode, memory_mode=model.navigation_config["memory"]["mode"],
                   cuda_runtime=torch.version.cuda,
                   allocator_config=os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
                   torch_threads=torch.get_num_threads(),
                   dependencies={name: importlib.metadata.version(name) for name in
                                 ("numpy", "pillow", "tokenizers", "fla-core", "triton")})
    # Record the exact checkpoint policy; never infer it from a repository name.
    metadata = serializer.metadata()
    runtime.update(
        serializer=metadata["serializer"],
        feedback_format=metadata.get("feedback_format"),
        append_action_tokens=model.navigation_config["observations"]["append_action_tokens"],
        kv_window_steps_including_current=model.navigation_config["memory"].get(
            "kv_window_steps_including_current"
        ),
    )
    if serializer.mode == "candidate_logits":
        runtime.update(
            candidate_token_ids=metadata["candidate_token_ids"],
            feedback_token_ids=metadata["feedback_token_ids"],
        )
    protocol_stdout.write(json.dumps(runtime) + "\n")
    protocol_stdout.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request["operation"] == "reset":
                    torch.cuda.reset_peak_memory_stats()
                    session.reset(request["episode_uid"], request["instruction"])
                    response = {"status": "ok"}
                elif request["operation"] == "observe":
                    with decode_rgb(request) as rgb:
                        response = session.observe(
                            request["episode_uid"], request["step_id"], rgb
                        )
                        response = {
                            **response,
                            "peak_allocated_gib": torch.cuda.max_memory_allocated()
                            / 2**30,
                        }
                else:
                    raise ValueError("Unknown operation")
        except Exception as exc:
            # CUDA/runtime failures cannot be repaired by simply resetting a session.
            fatal = isinstance(exc, (torch.OutOfMemoryError, RuntimeError))
            response = {"status": "fatal" if fatal else "failure", "error": f"{type(exc).__name__}: {exc}"}
        protocol_stdout.write(json.dumps(response) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
