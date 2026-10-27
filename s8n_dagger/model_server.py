"""Persistent JSON-lines S8N recovery process; stdout contains protocol only."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import torch
from qwen_vl.stream.transport import decode_rgb

from s8n_dagger.policy import RecoverySession


def dispatch(session, request):
    operation = request["operation"]
    uid = request["episode_uid"]
    if operation == "reset":
        return session.reset(uid, request["instruction"])
    if operation == "commit":
        return session.commit(uid, request["step_id"], request["habitat_action_id"])
    if operation in ("propose", "force"):
        with decode_rgb(request) as rgb:
            if operation == "propose":
                return session.propose(uid, request["step_id"], rgb)
            return session.force(uid, request["step_id"], rgb, request["habitat_action_id"])
    raise ValueError(f"Unknown operation: {operation!r}")


def serve(session, input_stream, output_stream):
    for line in input_stream:
        fatal = False
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                response = dispatch(session, request)
        except Exception as exc:
            fatal = isinstance(exc, (torch.OutOfMemoryError, RuntimeError))
            response = {"status": "fatal" if fatal else "failure", "error": f"{type(exc).__name__}: {exc}"}
        output_stream.write(json.dumps(response, allow_nan=False) + "\n")
        output_stream.flush()
        if fatal:
            break


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_metadata(model, serializer, checkpoint, model_path):
    import qwen_vl
    import transformers

    checkpoint = Path(checkpoint).resolve()
    source_root = Path(qwen_vl.__file__).resolve().parent
    weights = checkpoint / "model.safetensors"
    if not weights.exists():
        weights = checkpoint / "pytorch_model.bin"
    metadata = serializer.metadata()
    episodic = model.episodic_spec.to_dict()
    return dict(
        status="ready", protocol="s8n_recovery_proposal_commit_v1",
        python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
        cuda_runtime=torch.version.cuda,
        dependencies={name: importlib.metadata.version(name) for name in (
            "flash-attn", "flash-linear-attention", "numpy", "pillow", "tokenizers", "fla-core", "triton",
        )},
        allocator_config=os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
        torch_threads=torch.get_num_threads(), output_mode=serializer.mode,
        memory_mode=model.navigation_config["memory"]["mode"], episodic=episodic,
        serializer=metadata["serializer"], serializer_metadata=metadata,
        teacher=dict(checkpoint=str(checkpoint), weights_sha256=_sha256(weights),
                     navigation_sha256=_sha256(checkpoint / "navigation.json"),
                     source_root=str(source_root),
                     source_sha256={str(path.relative_to(source_root)): _sha256(path)
                                    for path in sorted(source_root.rglob("*.py"))},
                     model_path=str(Path(model_path).resolve()) if model_path else None),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()
    protocol_stdout = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        from qwen_vl.train.vln_runtime import load_checkpoint
        model, serializer = load_checkpoint(args.checkpoint, args.model_path)
        session = RecoverySession(model, serializer)
        runtime = runtime_metadata(model, serializer, args.checkpoint, args.model_path)
    protocol_stdout.write(json.dumps(runtime, allow_nan=False) + "\n")
    protocol_stdout.flush()
    serve(session, sys.stdin, protocol_stdout)


if __name__ == "__main__":
    main()
