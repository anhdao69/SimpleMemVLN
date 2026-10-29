"""Model-side client for the Habitat RL environment server subprocess."""
import base64
import json
import os
import selectors
import subprocess

import numpy as np


class EnvironmentError_(RuntimeError):
    """Broken simulator transport; the caller restarts the server."""


class EnvClient:
    def __init__(self, python, habitat_config, data_path, scenes_dir, gpu, episode_list,
                 split='train', pythonpath=None, timeout=900, log_path=None):
        cmd = [python, '-u', '-m', 'qwen_vl.rl.env_server', '--habitat-config', str(habitat_config),
               '--split', split, '--data-path', data_path, '--scenes-dir', str(scenes_dir),
               '--gpu', str(gpu), '--episode-list', str(episode_list)]
        env = dict(os.environ)
        # Habitat resolves gpu_device_id as a CUDA ordinal of the simulator process.
        # Under a scheduler mask (Slurm sets CUDA_VISIBLE_DEVICES) the trainer's device
        # `gpu` is the gpu-th masked entry, so the simulator sees only that device, as
        # ordinal 0, like the evaluator; without a mask the ordinal is the physical GPU
        # (verified by nvidia-smi in the environment smoke test).
        visible = [d for d in env.pop('CUDA_VISIBLE_DEVICES', '').split(',') if d.strip()]
        if visible:
            env['CUDA_VISIBLE_DEVICES'] = visible[int(gpu)]
            cmd[cmd.index('--gpu') + 1] = '0'
        if pythonpath:
            env['PYTHONPATH'] = pythonpath
        self.log = open(log_path, 'a') if log_path else None
        self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log or subprocess.DEVNULL, text=True,
                                        bufsize=1, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.timeout = timeout
        try:
            self.info = self._read()
            if self.info.get('status') != 'ready':
                raise EnvironmentError_(f'Environment failed to start: {self.info}')
        except BaseException:
            self.close()
            raise

    def _read(self):
        if not self.selector.select(self.timeout):
            raise EnvironmentError_('Environment request timed out')
        line = self.process.stdout.readline()
        if not line:
            raise EnvironmentError_(f'Environment process exited: {self.process.poll()}')
        return json.loads(line)

    def call(self, **request):
        try:
            self.process.stdin.write(json.dumps(request) + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EnvironmentError_('Environment pipe closed') from exc
        result = self._read()
        if result.get('status') != 'ok':
            raise EnvironmentError_(result.get('error', 'Environment failure'))
        return result

    def reset(self, episode_id):
        result = self.call(operation='reset', episode_id=str(episode_id))
        result['rgb'] = _rgb(result)
        return result

    def step(self, action):
        result = self.call(operation='step', action=int(action))
        if 'rgb_raw' in result:
            result['rgb'] = _rgb(result)
        return result

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(json.dumps(dict(operation='close')) + '\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=30)
                except Exception:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
        finally:
            self.selector.close()
            for stream in (self.process.stdin, self.process.stdout):
                try:
                    stream.close()
                except Exception:
                    pass
            if self.log:
                self.log.close()


def _rgb(result):
    raw = base64.b64decode(result.pop('rgb_raw'), validate=True)
    if len(raw) != 480 * 640 * 3:
        raise EnvironmentError_('Invalid raw RGB byte count')
    return np.frombuffer(raw, dtype=np.uint8).reshape(480, 640, 3).copy()
