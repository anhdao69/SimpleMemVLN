"""One-model GPU gate: bitwise original/adapter and override/force equivalence."""
import argparse
from collections import deque
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import subprocess

import numpy as np
from PIL import Image
import torch

from qwen_vl.stream.session import StreamSession
from s8n_dagger.common import sha256
from s8n_dagger.policy import RecoverySession


def _git_revision(directory):
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def gate_provenance(checkpoint, model_path, paths, frames, instruction):
    """Bind a gate to its exact assets and imported source without loading weights."""
    import qwen_vl

    checkpoint = Path(checkpoint).resolve()
    source = Path(qwen_vl.__file__).resolve().parent
    adapter = Path(__file__).resolve().parent
    weights = checkpoint / "model.safetensors"
    if not weights.exists():
        weights = checkpoint / "pytorch_model.bin"
    versions = {}
    for name in ("transformers", "flash-attn", "flash-linear-attention", "numpy",
                 "pillow", "tokenizers", "fla-core", "triton"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    runtime = dict(
        python=platform.python_version(), torch=torch.__version__,
        cuda_runtime=torch.version.cuda, dependencies=versions,
        torch_threads=torch.get_num_threads(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        allocator_config=os.environ.get("PYTORCH_ALLOC_CONF", os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")),
    )
    if torch.cuda.is_initialized():
        runtime.update(gpu=torch.cuda.get_device_name(),
                       capability=list(torch.cuda.get_device_capability()))
    return dict(
        schema="s8n_policy_gate_provenance_v1",
        weights_file=weights.name, weights_sha256=sha256(weights),
        navigation_sha256=sha256(checkpoint / "navigation.json"),
        model_path=str(Path(model_path).resolve()),
        instruction=instruction,
        source_root=str(source), source_git_revision=_git_revision(source),
        source_sha256={str(path.relative_to(source)): sha256(path)
                       for path in sorted(source.rglob("*.py"))},
        adapter_root=str(adapter), adapter_git_revision=_git_revision(adapter),
        adapter_sha256={str(path.relative_to(adapter)): sha256(path)
                        for path in sorted(adapter.rglob("*.py"))},
        rgb=[dict(path=str(path.resolve()), file_sha256=sha256(path),
                  decoded_rgb_sha256=hashlib.sha256(frame.tobytes(order="C")).hexdigest(),
                  shape=list(frame.shape), dtype=str(frame.dtype))
             for path, frame in zip(paths, frames, strict=True)],
        runtime=runtime,
    )


def _digest(value):
    digest = hashlib.sha256()
    seen = {}

    def emit(item):
        if isinstance(item, torch.nn.Module):
            # The decoder/weights are immutable and deliberately shared.
            digest.update(f"module:{type(item).__qualname__}".encode())
            return
        if isinstance(item, torch.Tensor):
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}".encode())
            tensor = item.detach().cpu().contiguous()
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            return
        if isinstance(item, np.ndarray):
            digest.update(f"array:{item.dtype}:{item.shape}".encode())
            digest.update(item.tobytes())
            return
        if isinstance(item, (str, int, float, bool, type(None), np.generic, torch.dtype, torch.device)):
            encoded = repr(item).encode()
            digest.update(struct.pack("!Q", len(encoded)))
            digest.update(encoded)
            return
        if id(item) in seen:
            digest.update(f"ref:{seen[id(item)]}".encode())
            return
        seen[id(item)] = len(seen)
        digest.update(f"{type(item).__module__}.{type(item).__qualname__}".encode())
        if isinstance(item, dict):
            for key in sorted(item, key=repr):
                emit(key)
                emit(item[key])
        elif isinstance(item, (list, tuple, deque)):
            for child in item:
                emit(child)
        elif hasattr(item, "__dict__"):
            emit(vars(item))
        else:
            raise TypeError(f"Cannot audit state type: {type(item)}")

    emit(value)
    return digest.hexdigest()


def state_fingerprints(session):
    """Hash every session memory component, including all native/scorer tensors."""
    result = {key: _digest(getattr(session, key)) for key in RecoverySession._STATE_FIELDS
              if key != "last_result"}  # Timing/diagnostics differ across identical runs.
    result["rope_deltas"] = _digest(session.model.backbone.model.rope_deltas)
    return result


def _assert_state(expected, actual, label):
    changed = [key for key in expected if expected[key] != actual[key]]
    if changed:
        raise AssertionError(f"{label}: differing memory state: {changed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rgb-dir", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    paths = sorted(Path(args.rgb_dir).glob("*.png"))[:args.steps]
    if args.steps < 5 or len(paths) != args.steps:
        parser.error("At least five saved RGB frames required to exercise episodic promotion")
    frames = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")).copy())
    from qwen_vl.train.vln_runtime import load_checkpoint
    model, serializer = load_checkpoint(args.checkpoint, args.model_path)
    uid = "policy-gpu-gate"
    report = dict(checkpoint=str(Path(args.checkpoint).resolve()),
                  rgb=[str(path.resolve()) for path in paths], steps=args.steps,
                  episodic=model.episodic_spec.to_dict(),
                  provenance=gate_provenance(args.checkpoint, args.model_path, paths,
                                             frames, args.instruction))

    # Each pass reuses the same immutable model. Only one session cache is live.
    original = StreamSession(model, serializer)
    original.reset(uid, args.instruction)
    originals, expected_states = [], []
    for step, frame in enumerate(frames):
        result = original.observe(uid, step, frame)
        originals.append({key: result[key] for key in ("habitat_action_id", "response_token_ids", "response_text")})
        expected_states.append(state_fingerprints(original))
    del original
    gc.collect()
    torch.cuda.empty_cache()
    adapter = RecoverySession(model, serializer)
    adapter.reset(uid, args.instruction)
    for step, frame in enumerate(frames):
        proposed = adapter.propose(uid, step, frame)
        for key, expected in originals[step].items():
            assert proposed[key] == expected, (step, key, proposed[key], expected)
        adapter.commit(uid, step, proposed["habitat_action_id"])
        _assert_state(expected_states[step], state_fingerprints(adapter), f"learner step {step}")
    report["pure_learner_bitwise_equal"] = True

    # Make every step an actual override. Keep all resulting fingerprints and
    # compare against independent expert-only replay from a clean reset.
    adapter.reset(uid, args.instruction)
    executed, override_states = [], []
    for step, frame in enumerate(frames):
        proposed = adapter.propose(uid, step, frame)
        action = (proposed["habitat_action_id"] + 1) % 4
        committed = adapter.commit(uid, step, action)
        assert committed["overridden"] and committed["habitat_action_id"] == action
        assert committed["response_token_ids"] == serializer.action_ids[committed["class_id"]]
        executed.append(action)
        override_states.append(state_fingerprints(adapter))
    adapter.reset(uid, args.instruction)
    for step, frame in enumerate(frames):
        adapter.force(uid, step, frame, executed[step])
        _assert_state(override_states[step], state_fingerprints(adapter), f"override step {step}")
    report["override_force_bitwise_equal"] = True
    report["executed_override_actions"] = executed

    # A learner continuation after forced expert control must match a learner
    # continuation after the same overridden response history.
    continuation = adapter.propose(uid, args.steps, frames[-1])
    adapter.commit(uid, args.steps, continuation["habitat_action_id"])
    continuation_state = state_fingerprints(adapter)
    adapter.reset(uid, args.instruction)
    for step, frame in enumerate(frames):
        adapter.propose(uid, step, frame)
        adapter.commit(uid, step, executed[step])
    resumed = adapter.propose(uid, args.steps, frames[-1])
    assert resumed["response_token_ids"] == continuation["response_token_ids"]
    adapter.commit(uid, args.steps, resumed["habitat_action_id"])
    _assert_state(continuation_state, state_fingerprints(adapter), "recovery continuation")
    report["continuation_bitwise_equal"] = True
    report["reset_bitwise_equal"] = True
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["status"] = "passed"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
