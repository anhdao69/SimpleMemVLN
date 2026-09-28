"""R2R Habitat loop with model execution isolated in its validated environment.

Uses the JanusVLN/StageVLN benchmark settings documented in the specification;
does not reuse the old reader-cache lifecycle or invalid-output STOP fallback.
Run this module with the simulator interpreter and --model-python for Qwen.
"""
import argparse
import base64
import importlib
import io
import json
import math
import numbers
from pathlib import Path
import selectors
import subprocess
import sys
import time
from PIL import Image
from qwen_vl.eval.metrics import summarize


class ModelProcess:
    def __init__(self, python, checkpoint, model_path=None, timeout=300):
        cmd = [python, "-m", "qwen_vl.stream.server", "--checkpoint", checkpoint]
        if model_path:
            cmd.extend(["--model-path", model_path])
        self.process = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.timeout = timeout
        if self.read().get("status") != "ready":
            raise RuntimeError("Model process failed to initialize")

    def read(self):
        if not self.selector.select(self.timeout):
            self.process.terminate()
            raise TimeoutError("Model request timed out; episode must fail")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Model process exited")
        return json.loads(line)

    def call(self, **request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self.read()
        if result.get("status") != "ok":
            raise RuntimeError(result.get("error", "Model failure"))
        return result

    def reset(self, uid, instruction):
        return self.call(operation="reset", episode_uid=uid, instruction=instruction)

    def observe(self, uid, step, rgb):
        image = Image.fromarray(rgb)
        if image.mode != "RGB" or image.size != (640, 480):
            raise ValueError("Unexpected camera contract")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return self.call(
            operation="observe",
            episode_uid=uid,
            step_id=step,
            rgb_png=base64.b64encode(buffer.getvalue()).decode(),
        )

    def close(self):
        self.selector.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--habitat-config", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--model-python", required=True)
    p.add_argument("--model-path")
    p.add_argument("--simulator-source")
    p.add_argument("--split", default="val_unseen", choices=["val_seen", "val_unseen"])
    p.add_argument(
        "--episode-list", required=True, help="JSON list of official episode ID strings"
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.simulator_source:
        sys.path.append(args.simulator_source)
    # Imported only in the simulator environment, never in model/dataset modules.
    import habitat
    from habitat_baselines.config.default import get_config
    import habitat_extensions.measures

    config = get_config(args.habitat_config)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.dataset.scenes_dir = str(Path(args.data_root) / "scene_datasets")
        config.habitat.dataset.data_path = str(
            Path(args.data_root) / "datasets/r2r/{split}/{split}.json.gz"
        )
        config.habitat.seed = 42
        config.habitat.environment.iterator_options.shuffle = False
    sim = config.habitat.simulator
    camera = sim.agents.main_agent.sim_sensors.rgb_sensor
    if (
        camera.width,
        camera.height,
        camera.hfov,
        sim.forward_step_size,
        sim.turn_angle,
    ) != (640, 480, 79, 0.25, 15):
        raise ValueError("Habitat camera/action contract differs from saved recipe")
    if (
        config.habitat.environment.max_episode_steps != 500
        or config.habitat.task.measurements.success.success_distance != 3.0
    ):
        raise ValueError("Unexpected success distance or rollout cap")
    wanted = [str(e) for e in json.loads(Path(args.episode_list).read_text())]
    if not wanted or len(set(wanted)) != len(wanted):
        raise ValueError("Nonempty unique episode list required")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    journal = out / "episodes.jsonl"
    records = (
        [json.loads(l) for l in journal.read_text().splitlines()]
        if journal.exists()
        else []
    )
    contract = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "episode_ids": wanted,
        "seed": 42,
        "failure_policy": "zero_sr_spl_keep_partial_metrics",
    }
    contract_file = out / "evaluation_contract.json"
    if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
        raise ValueError("Evaluation resume contract changed")
    contract_file.write_text(json.dumps(contract, indent=2) + "\n")
    done = {r["episode_id"] for r in records}
    if done == set(wanted):
        (out / "summary.json").write_text(
            json.dumps(summarize(records), indent=2) + "\n"
        )
        return
    with habitat.Env(config=config) as env:
        lookup = {str(e.episode_id): e for e in env.episodes}
        if set(wanted) - lookup.keys():
            raise ValueError("Requested episodes missing from simulator dataset")
        env.episodes = [lookup[e] for e in wanted if e not in done]
        model = ModelProcess(args.model_python, args.checkpoint, args.model_path)
        try:
            for _ in range(len(env.episodes)):
                record = {
                    "failed": False,
                    "metrics": {},
                    "actions": [],
                    "forced_stop": False,
                }
                positions = []
                try:
                    observation = env.reset()
                    ep = env.current_episode
                    uid = f"{args.split}:{ep.scene_id}:{ep.episode_id}"
                    record.update(
                        episode_id=str(ep.episode_id),
                        scene_id=ep.scene_id,
                        episode_uid=uid,
                    )
                    model.reset(uid, ep.instruction.instruction_text)
                    positions = [[float(x) for x in env.sim.get_agent_state().position]]
                    for step in range(500):
                        start = time.perf_counter()
                        prediction = model.observe(uid, step, observation["rgb"])
                        transport_and_model = time.perf_counter() - start
                        executed = prediction["habitat_action_id"]
                        if step == 499 and executed != 0:
                            executed = 0
                            record["forced_stop"] = True
                        start = time.perf_counter()
                        observation = env.step(executed)
                        record["actions"].append(
                            dict(
                                predicted=prediction["habitat_action_id"],
                                executed=executed,
                                model_seconds=prediction["model_seconds"],
                                transport_and_model_seconds=transport_and_model,
                                environment_seconds=time.perf_counter() - start,
                                retained_kv_tokens=prediction["retained_kv_tokens"],
                            )
                        )
                        record["actions"][-1]["peak_allocated_gib"] = prediction[
                            "peak_allocated_gib"
                        ]
                        positions.append(
                            [float(x) for x in env.sim.get_agent_state().position]
                        )
                        if env.episode_over:
                            break
                except Exception as exc:
                    record["failed"] = True
                    record["failure_reason"] = f"{type(exc).__name__}: {exc}"
                    if "episode_id" not in record:
                        record["episode_id"] = str(env.current_episode.episode_id)
                if positions:
                    import numpy as np

                    record["metrics"]["trajectory_length"] = float(
                        np.linalg.norm(
                            np.diff(np.asarray(positions), axis=0), axis=1
                        ).sum()
                    )
                    record["trajectory"] = positions
                try:
                    for key, value in env.get_metrics().items():
                        if isinstance(value, numbers.Real) and math.isfinite(
                            float(value)
                        ):
                            record["metrics"][key] = float(value)
                except Exception:
                    pass
                if (
                    not {"success", "spl"} <= record["metrics"].keys()
                    and not record["failed"]
                ):
                    record["failed"] = True
                    record["failure_reason"] = "Required simulator metrics unavailable"
                records.append(record)
                with journal.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                (out / "summary.json").write_text(
                    json.dumps(summarize(records), indent=2) + "\n"
                )
        finally:
            model.close()
    if len(records) != len(wanted):
        raise RuntimeError("Evaluation denominator incomplete")


if __name__ == "__main__":
    main()
