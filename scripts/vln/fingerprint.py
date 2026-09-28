"""Record the actual interpreter, binaries, toolchain and source hashes."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import torch


def main():
    packages = [
        "torch",
        "torchvision",
        "transformers",
        "flash-attn",
        "flash-linear-attention",
        "fla-core",
        "tilelang",
        "triton",
        "accelerate",
        "deepspeed",
    ]
    report = dict(
        interpreter=sys.executable,
        python=sys.version,
        packages={p: importlib.metadata.version(p) for p in packages},
        torch_cuda_runtime=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_home=os.environ.get("CUDA_HOME"),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], text=True
        ),
        git_revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_dirty=bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        source_sha256={},
    )
    for module in ["torch", "transformers", "flash_attn", "fla", "tilelang"]:
        report.setdefault("import_paths", {})[module] = importlib.import_module(
            module
        ).__file__
    for module in [
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.cache_utils",
        "fla.ops.gated_delta_rule.chunk",
        "flash_attn.flash_attn_interface",
        "flash_attn_2_cuda",
    ]:
        path = Path(importlib.import_module(module).__file__)
        report.setdefault("installed_source_sha256", {})[str(path)] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    files = [
        Path("pyproject.toml"),
        Path("uv.lock"),
        Path("deepspeed.json"),
        *Path("src").rglob("*.py"),
        *Path("scripts/vln").glob("*.py"),
        *Path("tests/vln").glob("*.py"),
        *Path("configs").glob("vln*.yaml"),
    ]
    for path in files:
        report["source_sha256"][str(path)] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    if os.environ.get("CUDA_HOME"):
        report["nvcc"] = subprocess.check_output(
            [str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc"), "--version"], text=True
        )
    Path("artifacts/runtime.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "source_sha256"}, indent=2)
    )


if __name__ == "__main__":
    main()
