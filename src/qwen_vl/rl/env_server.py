"""JSON-lines Habitat 0.2.4 R2R environment server for RL rollouts.

Runs under the simulator interpreter (EVAL_PYTHON). Camera, action, success
and cap settings are asserted exactly as in ``qwen_vl.eval.habitat_r2r``; only
the split differs (RL trains on ``train``). Observations travel as lossless raw
RGB, the same transport the evaluator uses.
"""
import argparse
import json
import math
import numbers
import sys
from pathlib import Path

import numpy as np
from qwen_vl.stream.transport import encode_rgb

HABITAT_STOP = 0


def build_config(habitat_config, split, data_path, scenes_dir, gpu):
    import habitat
    from habitat.config.default import get_config

    config = get_config(habitat_config)
    with habitat.config.read_write(config):
        config.habitat.dataset.split = split
        config.habitat.dataset.scenes_dir = str(Path(scenes_dir).resolve())
        config.habitat.dataset.data_path = data_path
        config.habitat.seed = 42
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu
        config.habitat.environment.iterator_options.shuffle = False
    sim = config.habitat.simulator
    camera = sim.agents.main_agent.sim_sensors.rgb_sensor
    if (camera.width, camera.height, camera.hfov, sim.forward_step_size, sim.turn_angle) != (640, 480, 79, 0.25, 15):
        raise ValueError('Habitat camera/action contract differs from saved recipe')
    if (config.habitat.environment.max_episode_steps != 500
            or config.habitat.task.measurements.success.success_distance != 3.0):
        raise ValueError('Unexpected success distance or rollout cap')
    if config.habitat.dataset.type != 'R2RVLN-v1':
        raise ValueError('RL environment requires the R2RVLN-v1 dataset')
    return config


def scalar_metrics(env):
    return {key: float(value) for key, value in env.get_metrics().items()
            if isinstance(value, numbers.Real) and math.isfinite(float(value))}


class RolloutEnvironment:
    def __init__(self, habitat_config, split, data_path, scenes_dir, gpu, episode_ids=None):
        import habitat
        from habitat.datasets import make_dataset

        self.config = build_config(habitat_config, split, data_path, scenes_dir, gpu)
        dataset = make_dataset(self.config.habitat.dataset.type, config=self.config.habitat.dataset)
        lookup = {str(ep.episode_id): ep for ep in dataset.episodes}
        if len(lookup) != len(dataset.episodes):
            raise ValueError('Dataset has duplicate episode IDs')
        if episode_ids is not None:
            missing = sorted(set(map(str, episode_ids)) - lookup.keys())
            if missing:
                raise ValueError(f'Episodes missing from {split}: {missing[:10]}')
            lookup = {key: lookup[key] for key in map(str, episode_ids)}
        dataset.episodes = sorted(lookup.values(), key=lambda ep: (ep.scene_id, str(ep.episode_id)))
        self.lookup = lookup
        self.env = habitat.Env(config=self.config, dataset=dataset)
        self.episode = None
        self.steps = 0

    def position(self):
        return [float(x) for x in self.env.sim.get_agent_state().position]

    def bearing_error(self):
        """Signed angle (rad) from the agent heading to the first shortest-path point >= 0.5 m away.

        Privileged training-time geometry for the state baseline (plans/RL/research); it never
        reaches the policy. None when the navmesh has no path.
        """
        from habitat_sim.nav import ShortestPath
        from habitat_sim.utils.common import quat_rotate_vector

        state = self.env.sim.get_agent_state()
        here = np.asarray(state.position, dtype=np.float64)
        path = ShortestPath()
        path.requested_start = here
        path.requested_end = np.asarray(self.episode.goals[0].position, dtype=np.float64)
        if not self.env.sim.pathfinder.find_path(path) or len(path.points) < 2:
            return None
        target = next((np.asarray(p) for p in path.points[1:] if np.linalg.norm(np.asarray(p) - here) >= 0.5),
                      np.asarray(path.points[-1]))
        heading = quat_rotate_vector(state.rotation, np.array([0.0, 0.0, -1.0]))
        direction = target - here
        angle = lambda v: math.atan2(-v[0], -v[2])
        return float((angle(direction) - angle(heading) + math.pi) % (2 * math.pi) - math.pi)

    def reset(self, episode_id):
        episode = self.lookup[str(episode_id)]
        self.env.current_episode = episode
        observation = self.env.reset()
        self.episode, self.steps = episode, 0
        metrics = scalar_metrics(self.env)
        return dict(status='ok', episode_id=str(episode.episode_id), scene_id=episode.scene_id,
                    instruction=episode.instruction.instruction_text.strip(),
                    geodesic_distance=float(episode.info['geodesic_distance']),
                    distance_to_goal=metrics.get('distance_to_goal'),
                    position=self.position(), bearing_error=self.bearing_error(),
                    **encode_rgb(observation['rgb']))

    def step(self, action):
        if self.episode is None or self.env.episode_over:
            raise ValueError('Step without an active episode')
        if action not in (0, 1, 2, 3):
            raise ValueError(f'Invalid Habitat action {action!r}')
        observation = self.env.step(action)
        self.steps += 1
        metrics = scalar_metrics(self.env)
        result = dict(status='ok', episode_over=bool(self.env.episode_over),
                      distance_to_goal=metrics.get('distance_to_goal'),
                      position=self.position(), steps=self.steps,
                      bearing_error=None if self.env.episode_over else self.bearing_error())
        if self.env.episode_over:
            result['metrics'] = metrics
        else:
            result.update(encode_rgb(observation['rgb']))
        return result

    def close(self):
        self.env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--habitat-config', required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--data-path', required=True)
    p.add_argument('--scenes-dir', required=True)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--episode-list', help='JSON list of episode IDs to admit')
    args = p.parse_args()
    protocol = sys.stdout
    sys.stdout = sys.stderr  # simulator logging must never corrupt the protocol
    wanted = json.loads(Path(args.episode_list).read_text()) if args.episode_list else None
    env = RolloutEnvironment(args.habitat_config, args.split, args.data_path,
                             args.scenes_dir, args.gpu, wanted)
    import habitat
    protocol.write(json.dumps(dict(status='ready', episodes=len(env.lookup),
                                   habitat=getattr(habitat, '__version__', 'unknown'))) + '\n')
    protocol.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            op = request['operation']
            if op == 'reset':
                response = env.reset(request['episode_id'])
            elif op == 'step':
                response = env.step(request['action'])
            elif op == 'close':
                protocol.write(json.dumps(dict(status='ok')) + '\n')
                protocol.flush()
                break
            else:
                raise ValueError(f'Unknown operation {op!r}')
        except Exception as exc:
            response = dict(status='failure', error=f'{type(exc).__name__}: {exc}')
        protocol.write(json.dumps(response, allow_nan=False) + '\n')
        protocol.flush()
    env.close()


if __name__ == '__main__':
    main()
