"""Shared validated loading and recipe resolution for train and serving."""
import json
import os
from pathlib import Path
import torch
import transformers
import yaml
from qwen_vl.contracts import deep_merge, validate_config
from qwen_vl.data.episode_serializer import EpisodeSerializer
from qwen_vl.models.nav_model import SimpleMemVLNForNavigation
from qwen_vl.stream.window_attention import register_window_attention


def resolve_config(base, output, memory=None):
    cfg = {}
    for path in (base, output, memory):
        if path:
            with open(path) as f:
                cfg = deep_merge(cfg, yaml.safe_load(f))
    return validate_config(cfg)


def load_model(cfg, model_path=None):
    if transformers.__version__ != "5.11.0":
        raise RuntimeError("Native continuation requires verified Transformers 5.11.0")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = 0 if torch.cuda.device_count() == 1 else local_rank
    torch.cuda.set_device(device)
    register_window_attention()
    source = model_path or cfg["model"]["backbone"]
    revision = None if Path(source).exists() else cfg["model"]["revision"]
    backbone = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
        source,
        revision=revision,
        dtype=torch.bfloat16,
        attn_implementation={
            "text_config": cfg["runtime"]["text_attention"],
            "vision_config": cfg["runtime"]["vision_attention"],
        },
        device_map={"": f"cuda:{device}"},
    )
    for layer in backbone.model.language_model.layers:
        if hasattr(layer, "linear_attn"):
            for name in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
                fn = getattr(layer.linear_attn, name)
                if not fn.__module__.startswith("fla."):
                    raise RuntimeError(f"Unapproved GDN fallback: {name} {fn}")
    processor = transformers.AutoProcessor.from_pretrained(source, revision=revision)
    serializer = EpisodeSerializer(processor, cfg)
    return SimpleMemVLNForNavigation(backbone, cfg,
        candidate_token_ids=getattr(serializer, 'candidate_token_ids', None)), serializer


def load_checkpoint(path, model_path=None):
    from safetensors.torch import load_model as load_safetensors

    metadata = json.loads((Path(path) / "navigation.json").read_text())
    model, serializer = load_model(metadata["config"], model_path)
    if serializer.metadata() != metadata:
        raise ValueError("Checkpoint tokenizer/serializer/codec contract differs")
    weights = Path(path) / "model.safetensors"
    if weights.exists():
        load_safetensors(model, str(weights), strict=True)
    else:
        model.load_state_dict(
            torch.load(
                Path(path) / "pytorch_model.bin", map_location="cpu", weights_only=True
            ),
            strict=True,
        )
    return model, serializer
