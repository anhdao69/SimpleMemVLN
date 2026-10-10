"""Multi-process rollouts that read the learner's live weights through CUDA IPC.

Rollouts at batch size 1 are bound by Python kernel-launch overhead, so threads in the
trainer process serialize on the GIL. Each helper here is a separate process on the
rank's GPU holding a weightless model skeleton whose every parameter and buffer is a CUDA
IPC view of the trainer's own tensor (ZeRO-2 bf16 flat buffers for trainable weights).
DeepSpeed updates those buffers in place, so helpers always sample from the current
policy, with no copy or synchronization step beyond ``torch.cuda.synchronize``.
"""
import os
import queue
import traceback

import torch
import torch.multiprocessing as mp


def shared_tensors(model):
    """Every parameter/buffer name (aliases included, e.g. the tied LM head) -> tensor."""
    tensors = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        tensors[name] = parameter.data
    for name, buffer in model.named_buffers(remove_duplicate=False):
        if buffer is not None:
            tensors[name] = buffer
    return tensors


def storage_fingerprint(model):
    return tuple((name, int(t.data_ptr())) for name, t in sorted(shared_tensors(model).items()))


def build_skeleton(navigation_config, model_path, device):
    """``vln_runtime.load_model`` without reading weights: modules are created on meta."""
    import transformers
    from qwen_vl.models.nav_model import SimpleMemVLNForNavigation
    from qwen_vl.stream.window_attention import register_window_attention

    register_window_attention()
    cfg = navigation_config
    config = transformers.AutoConfig.from_pretrained(model_path)
    with torch.device('meta'):
        backbone = transformers.Qwen3_5ForConditionalGeneration._from_config(
            config, dtype=torch.bfloat16,
            attn_implementation={'text_config': cfg['runtime']['text_attention'],
                                 'vision_config': cfg['runtime']['vision_attention']})
    for layer in backbone.model.language_model.layers:
        if hasattr(layer, 'linear_attn'):
            for name in ('chunk_gated_delta_rule', 'recurrent_gated_delta_rule'):
                if not getattr(layer.linear_attn, name).__module__.startswith('fla.'):
                    raise RuntimeError(f'Unapproved GDN fallback: {name}')
    return SimpleMemVLNForNavigation(backbone, cfg)


def attach(model, tensors):
    """Replace every parameter/buffer of the skeleton with the shared tensor of that name."""
    seen = set()
    for name, parameter in list(model.named_parameters(remove_duplicate=False)):
        module_name, _, leaf = name.rpartition('.')
        module = model.get_submodule(module_name) if module_name else model
        module._parameters[leaf] = torch.nn.Parameter(tensors[name], requires_grad=False)
        seen.add(name)
    for name, buffer in list(model.named_buffers(remove_duplicate=False)):
        module_name, _, leaf = name.rpartition('.')
        module = model.get_submodule(module_name) if module_name else model
        if name not in tensors:
            raise KeyError(f'Missing shared buffer {name}')
        module._buffers[leaf] = tensors[name]
        seen.add(name)
    missing = set(tensors) - seen
    if missing:
        raise KeyError(f'Shared tensors without a skeleton slot: {sorted(missing)[:5]}')
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError('Skeleton still holds meta tensors')
    # Tied embeddings must stay tied after attachment.
    lm = model.backbone.lm_head.weight
    emb = model.backbone.model.language_model.embed_tokens.weight
    if lm.data_ptr() != emb.data_ptr():
        raise RuntimeError('Tied LM head lost its tie')
    return model.eval()


def helper_main(index, device_index, tensors, navigation_metadata, model_path, env_kwargs,
                tasks, results):
    try:
        torch.cuda.set_device(device_index)
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        import transformers
        from qwen_vl.data.episode_serializer import EpisodeSerializer
        from qwen_vl.rl.env_client import EnvClient, EnvironmentError_
        from qwen_vl.rl.rollout import run_episode
        from qwen_vl.rl.sampling import SamplingSession

        model = attach(build_skeleton(navigation_metadata['config'], model_path, device_index), tensors)
        processor = transformers.AutoProcessor.from_pretrained(model_path)
        serializer = EpisodeSerializer(processor, navigation_metadata['config'])
        if serializer.metadata() != navigation_metadata:
            raise ValueError('Helper serializer contract differs')
        session = SamplingSession(model, serializer)
        make_env = lambda: EnvClient(**env_kwargs)
        env = make_env()
        device = torch.device('cuda', device_index)
        results.put(('ready', index, None))
        while True:
            task = tasks.get()
            if task is None:
                break
            key, item, kwargs = task
            for _ in range(3):
                try:
                    episode = run_episode(session, env, item, device=device, **kwargs)
                    break
                except EnvironmentError_ as exc:
                    try:
                        env.close()
                    except Exception:
                        pass
                    env = make_env()
            else:
                raise RuntimeError('Environment failed three times on one episode')
            torch.cuda.synchronize()
            results.put(('done', key, episode))
        env.close()
    except BaseException:
        results.put(('error', index, traceback.format_exc()))


class RolloutPool:
    def __init__(self, model, navigation_metadata, model_path, env_kwargs, device_index, size):
        ctx = mp.get_context('spawn')
        self.tasks, self.results = ctx.Queue(), ctx.Queue()
        self.fingerprint = storage_fingerprint(model)
        tensors = shared_tensors(model)
        self.procs = [ctx.Process(target=helper_main, daemon=True,
                                  args=(i, device_index, tensors, navigation_metadata, model_path,
                                        dict(env_kwargs, log_path=f"{env_kwargs['log_path']}.h{i}"),
                                        self.tasks, self.results))
                      for i in range(size)]
        for proc in self.procs:
            proc.start()
        self._model = model
        ready = 0
        while ready < size:
            kind, index, payload = self._get()
            if kind != 'ready':
                raise RuntimeError(f'Unexpected helper message {kind}')
            ready += 1

    def _get(self, timeout=3600):
        while True:
            try:
                kind, key, payload = self.results.get(timeout=60)
            except queue.Empty:
                dead = [p.pid for p in self.procs if not p.is_alive()]
                if dead:
                    raise RuntimeError(f'Rollout helper(s) died: {dead}')
                timeout -= 60
                if timeout <= 0:
                    raise TimeoutError('Rollout helpers stalled')
                continue
            if kind == 'error':
                raise RuntimeError(f'Rollout helper {key} failed:\n{payload}')
            return kind, key, payload

    def submit(self, key, item, **kwargs):
        self.tasks.put((key, item, kwargs))

    def get(self):
        kind, key, episode = self._get()
        return key, episode

    def check_storage(self):
        if storage_fingerprint(self._model) != self.fingerprint:
            raise RuntimeError('Learner tensors were reallocated; helpers would read stale weights')

    def close(self):
        for _ in self.procs:
            self.tasks.put(None)
        for proc in self.procs:
            proc.join(timeout=120)
            if proc.is_alive():
                proc.terminate()
