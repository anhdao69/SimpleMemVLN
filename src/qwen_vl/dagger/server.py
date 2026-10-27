"""JSON-lines model process for DAgger collection (model interpreter).

Operations: ``reset`` (episode_uid, instruction) and ``step`` (episode_uid, step_id, rgb_raw |
rgb_png, forced_class | null, intercept_with | null).  Loading, numerics and failure semantics
follow ``qwen_vl.stream.server`` (the evaluator's model process): status ``fatal`` for CUDA or
runtime failures, ``failure`` for a recoverable request error (the episode is abandoned).
"""
import argparse
import contextlib
import hashlib
import json
import os
import sys

from qwen_vl.stream.transport import decode_rgb

PROTOCOL = 'simplememvln_dagger_step_v1'


def _optional_class(request, key):
    value = request.get(key)
    if value is not None and (type(value) is not int or value not in range(4)):
        raise ValueError(f'{key} must be null or an action class 0..3')
    return value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model-path')
    args = p.parse_args()
    protocol_stdout = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        import torch
        from qwen_vl.dagger.session import DaggerSession
        from qwen_vl.train.vln_runtime import load_checkpoint

        model, serializer = load_checkpoint(args.checkpoint, args.model_path)
        session = DaggerSession(model, serializer)
    import importlib.metadata
    import platform
    import transformers
    metadata = serializer.metadata()
    runtime = dict(status='ready', protocol=PROTOCOL, python=platform.python_version(),
                   torch=torch.__version__, transformers=transformers.__version__,
                   gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
                   flash_attn=importlib.metadata.version('flash-attn'),
                   fla=importlib.metadata.version('flash-linear-attention'),
                   cuda_runtime=torch.version.cuda,
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   output_mode=serializer.mode, memory_mode=model.navigation_config['memory']['mode'],
                   serializer=metadata['serializer'], action_token_ids=metadata['action_token_ids'],
                   stop_first_token=session.layout.stop_first,
                   append_action_tokens=model.navigation_config['observations']['append_action_tokens'],
                   navigation_sha256=hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest())
    if getattr(model, 'episodic_spec', None) is not None:
        runtime.update(episodic=model.episodic_spec.to_dict())
    protocol_stdout.write(json.dumps(runtime) + '\n')
    protocol_stdout.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                operation = request['operation']
                if operation == 'reset':
                    torch.cuda.reset_peak_memory_stats()
                    session.reset(request['episode_uid'], request['instruction'])
                    response = dict(status='ok')
                elif operation == 'step':
                    forced = _optional_class(request, 'forced_class')
                    intercept = _optional_class(request, 'intercept_with')
                    with decode_rgb(request) as rgb:
                        response = session.step(request['episode_uid'], request['step_id'], rgb,
                                                forced_class=forced, intercept_with=intercept)
                    response['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
                else:
                    raise ValueError('Unknown operation')
        except Exception as exc:
            fatal = isinstance(exc, (torch.OutOfMemoryError, RuntimeError))
            response = dict(status='fatal' if fatal else 'failure', error=f'{type(exc).__name__}: {exc}')
        protocol_stdout.write(json.dumps(response) + '\n')
        protocol_stdout.flush()


if __name__ == '__main__':
    main()
