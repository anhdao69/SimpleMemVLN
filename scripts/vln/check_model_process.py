"""Exercise the real evaluator/model IPC without pretending to run Habitat."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from qwen_vl.eval.habitat_r2r import ModelProcess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="artifacts/model_process.json")
    args = p.parse_args()
    ep = json.loads(Path(args.manifest).read_text().splitlines()[0])
    model = ModelProcess(sys.executable, args.checkpoint, args.model_path)
    try:
        model.reset(ep["episode_uid"], ep["instruction"])
        rows = []
        for step in range(2):
            with Image.open(ep["steps"][step]["rgb_path"]) as im:
                rgb = np.asarray(im)
            result = model.observe(ep["episode_uid"], step, rgb)
            assert result == model.observe(ep["episode_uid"], step, rgb)
            assert result["step_id"] == step and result["habitat_action_id"] in (
                0,
                1,
                2,
                3,
            )
            rows.append(result)
        model.reset(ep["episode_uid"], ep["instruction"])
        with Image.open(ep["steps"][0]["rgb_path"]) as im:
            rgb = np.asarray(im)
        assert model.observe(ep["episode_uid"], 0, rgb)["logits"] == rows[0]["logits"]
        Path(args.out).write_text(
            json.dumps(
                dict(passed=True, kind="model_ipc_without_simulator", results=rows),
                indent=2,
            )
            + "\n"
        )
        print("MODEL_IPC_PASS")
    finally:
        model.close()


if __name__ == "__main__":
    main()
