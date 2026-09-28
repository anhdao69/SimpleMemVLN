"""Compare a saved navigation model with the pre-save execution reference."""
import argparse, json
from pathlib import Path
import torch
from qwen_vl.train.vln_runtime import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--diagnose-cast", action="store_true")
    args = p.parse_args()
    path = Path(args.checkpoint)
    ref = json.loads((path / "reload_reference.json").read_text())
    model, serializer = load_checkpoint(path, args.model_path)
    model.eval()
    episode = next(
        json.loads(l)
        for l in Path(args.manifest).read_text().splitlines()
        if json.loads(l)["episode_uid"] == ref["episode_uid"]
    )
    data = {
        k: v.cuda() if torch.is_tensor(v) else v
        for k, v in serializer.encode_episode(episode).items()
    }
    if args.diagnose_cast:
        saved = torch.load(
            path / "pytorch_model.bin", map_location="cpu", weights_only=True
        )
        print(
            "STATE_DTYPE_DIFFERENCES",
            [
                (k, str(v.dtype), str(saved[k].dtype))
                for k, v in model.state_dict().items()
                if k in saved and v.dtype != saved[k].dtype
            ],
            flush=True,
        )
        print(
            "FLOAT_BUFFERS",
            [
                (k, str(v.dtype))
                for k, v in model.named_buffers()
                if v.is_floating_point()
            ],
            flush=True,
        )
        model.bfloat16()
    with torch.inference_mode():
        out = model(**data)
    if ref["logits"] is not None:
        expected = torch.tensor(ref["logits"], device="cuda")
        torch.testing.assert_close(
            out["logits"].float(), expected, atol=1e-5, rtol=1e-5
        )
    assert abs(float(out["loss_sum"]) - ref["loss_sum"]) < 1e-4
    report = dict(
        passed=True, episode_uid=ref["episode_uid"], loss_sum=float(out["loss_sum"])
    )
    (path / "reload_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
