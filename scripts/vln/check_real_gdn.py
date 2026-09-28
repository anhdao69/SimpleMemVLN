"""Isolate native GDN numerical carry on real observation embeddings."""
import argparse, json
from pathlib import Path
import torch
from qwen_vl.train.vln_runtime import resolve_config, load_model
from qwen_vl.stream.cache import make_stream_cache_fp32


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-path", required=True)
    args = p.parse_args()
    cfg = resolve_config(
        "configs/vln_r2r_v0_base.yaml", "configs/vln_r2r_v0_classification.yaml"
    )
    model, serializer = load_model(cfg, args.model_path)
    model.eval()
    ep = min(
        [json.loads(l) for l in Path(args.manifest).read_text().splitlines()],
        key=lambda e: len(e["steps"]),
    )
    x = {
        k: v.cuda() if torch.is_tensor(v) else v
        for k, v in serializer.encode_episode(ep).items()
    }
    with torch.inference_mode():
        emb = model.embed(x["input_ids"], x["pixel_values"], x["image_grid_thw"])
        layer = model.backbone.model.language_model.layers[0]
        emb = layer.input_layernorm(emb)
        block = layer.linear_attn
        full = block(emb)
        cache = make_stream_cache_fp32(model.config.text_config)
        prefix, spans = x["step_plan"]
        chunks = [
            block(emb[:, s:e], cache_params=cache) for s, e in [(0, prefix), *spans]
        ]
        streamed = torch.cat(chunks, 1)
        diff = full.float() - streamed.float()
        shape_control = block(emb[:, : spans[0][1]])
        out = dict(
            max=float(diff.abs().max()),
            rms=float(diff.square().mean().sqrt()),
            full_rms=float(full.float().square().mean().sqrt()),
            prefix_shape_error=float(
                (shape_control - full[:, : spans[0][1]]).abs().max()
            ),
            prefix_stream_error=float(
                (shape_control - streamed[:, : spans[0][1]]).abs().max()
            ),
            per_step_max=[float(diff[:, s:e].abs().max()) for s, e in spans],
        )
        print(json.dumps(out, indent=2))
        Path("artifacts/real_gdn.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
