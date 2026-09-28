"""Single-session JSON-lines model subprocess; RGB transport is lossless PNG."""
import argparse
import base64
import contextlib
import io
import json
import sys
from PIL import Image


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
    protocol_stdout.write(json.dumps({"status": "ready"}) + "\n")
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
                    with Image.open(
                        io.BytesIO(base64.b64decode(request["rgb_png"]))
                    ) as rgb:
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
            response = {"status": "failure", "error": f"{type(exc).__name__}: {exc}"}
        protocol_stdout.write(json.dumps(response) + "\n")
        protocol_stdout.flush()


if __name__ == "__main__":
    main()
