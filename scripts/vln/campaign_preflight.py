"""Resource, provenance, memory and completion gates for the approved campaign."""
import argparse
import json
import shutil
from pathlib import Path

from qwen_vl.train.campaign import check_memory_profile, verify_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--manifest", required=True)
    args = p.parse_args()
    root, campaign = Path(args.root), Path(args.campaign)
    provenance = json.loads((campaign / "provenance.json").read_text())
    for name, digest in provenance["source_sha256"].items():
        verify_sha256(root / name, digest)
    verify_sha256(args.manifest, provenance["manifest_sha256"])
    if args.variant == "gate":
        if not json.loads((campaign / "epoch_save_test/PASS.json").read_text())[
            "passed"
        ]:
            raise ValueError("Epoch checkpoint test failed")
        result = check_memory_profile(campaign / "profile_fullcontext_b")
        (campaign / "preflight_pass.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print("PREFLIGHT_PASS", json.dumps(result), flush=True)
        return
    if args.variant.startswith("complete-"):
        run = campaign / args.variant.removeprefix("complete-")
        state = json.loads((run / "trainer_state.json").read_text())
        if state["global_step"] != 4059 or state["epoch"] != 3:
            raise ValueError("Training ended before all three epochs")
        for epoch, step in enumerate((1353, 2706, 4059), 1):
            checkpoint = run / f"checkpoint-{step}"
            saved = json.loads((checkpoint / "trainer_state.json").read_text())
            if (
                saved["epoch"] != epoch
                or not (checkpoint / "pytorch_model.bin").is_file()
            ):
                raise ValueError("Missing completed epoch export")
            if len(list(checkpoint.rglob("*optim_states.pt"))) != 4:
                raise ValueError("Missing optimizer rank checkpoint")
        (run / "TRAINING_COMPLETE.json").write_text(
            json.dumps({"passed": True, "steps": 4059, "epochs": 3}) + "\n"
        )
        return
    free_gib = shutil.disk_usage(campaign).free / 2**30
    required = 600 if args.variant in ("preflight", "fullcontext") else 300
    if free_gib < required:
        raise ValueError(
            f"Insufficient disk: {free_gib:.1f} GiB free, require {required}"
        )
    import torch
    from qwen_vl.train.vln_runtime import resolve_config

    if torch.cuda.device_count() != 4 or not all(
        "H100" in torch.cuda.get_device_name(i) for i in range(4)
    ):
        raise ValueError("Exactly four H100s required")
    cfg = resolve_config(
        root / "configs/vln_r2r_v0_base.yaml",
        root / "configs/vln_r2r_b_3epoch_bs8.yaml",
    )
    episodes = [
        json.loads(line) for line in Path(args.manifest).read_text().splitlines()
    ]
    if len(episodes) != 10819 or sum(len(ep["steps"]) for ep in episodes) != 631244:
        raise ValueError("Unexpected R2R corpus")
    if cfg["training"]["nominal_episodes_per_update"] != 8:
        raise ValueError("Wrong global batch")
    print(
        "RESOURCE_PREFLIGHT",
        json.dumps(
            {
                "free_gib": free_gib,
                "required_gib": required,
                "episodes": len(episodes),
                "expected_steps": 4059,
                "expected_warmup": 122,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
