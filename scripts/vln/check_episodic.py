"""Real pinned-model episodic gates (one H100; no training, no navigation claim).

A. Feature disabled: V0 (episodic R8/B0) offline logits/loss equal legacy Window8.
B. Real-data committed-write helper versus the dispatched FLA GDN kernel state.
C. Offline training forward versus the real StreamSession text loop with forced
   canonical responses over real R2R/RxR prefixes that cross many admissions:
   every response-token logit, decoded actions, per-step selected IDs and raw
   source scores; inherited BF16 budget atol .75 / rtol .03 / RMS <= .15.
D. Retry idempotency, reset, bounded resident KV with no history archive.
E. >=500-step cyclic real-RGB dry stream: bounded cache (not a navigation score).
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from qwen_vl.train.vln_runtime import load_model, resolve_config
from qwen_vl.data.episode_serializer import EpisodeSerializer
from qwen_vl.stream.session import StreamSession
from qwen_vl.stream.cache import kv_length, make_stream_cache_fp32
from qwen_vl.models.committed_write import CommittedWriteScorer
from qwen_vl.research.episodic_contract import parse_episodic_spec, derived_kv_guard

ARMS = {a: f'configs/episodic/arm_{a}.yaml' for a in ('v0', 'm1', 'm2', 'm2c')}


def cfg_for(arm):
    return resolve_config('configs/episodic/science_common.yaml', ARMS[arm],
                          runtime='configs/episodic/profile_h4.yaml')


def encode(serializer, episode, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in serializer.encode_episode(episode).items()}


def fixture(manifest, dataset, steps, rng_seed=0):
    """A real episode prefix; its final label is set to STOP only for this replay fixture."""
    rng = np.random.default_rng(rng_seed)
    pool = []
    for line in Path(manifest).open():
        e = json.loads(line)
        if e['dataset'] == dataset and len(e['steps']) >= steps + 1:
            pool.append(e)
            if len(pool) >= 64:
                break
    e = deepcopy(pool[int(rng.integers(len(pool)))])
    e['steps'] = e['steps'][:steps]
    e['steps'][-1]['action_name'] = 'STOP'
    e['episode_uid'] += f':replay{steps}'
    return e


def switch(model, serializer, cfg):
    """Same weights, different navigation contract (no reload)."""
    model.navigation_config = cfg
    model.episodic_spec = parse_episodic_spec(cfg, model.config.text_config)
    return EpisodeSerializer(serializer.processor, cfg)


def gate_a(model, serializer, episode, device):
    legacy = resolve_config('configs/episodic/science_common.yaml', 'configs/vln_memory_window8.yaml',
                            runtime='configs/episodic/profile_h4.yaml')
    legacy['memory']['kv_guard_tokens'] = 4096
    s_legacy = switch(model, serializer, legacy)
    with torch.inference_mode():
        old = model(**encode(s_legacy, episode, device))
        s_v0 = switch(model, serializer, cfg_for('v0'))
        data = encode(s_v0, episode, device)
        new = model(**data)
    return dict(tokens=int(data['input_ids'].numel()), groups=len(episode['steps']),
                logits_bitwise_equal=bool(torch.equal(old['logits'], new['logits'])),
                loss_sum_equal=bool(torch.equal(old['loss_sum'], new['loss_sum'])),
                max_logit_error=float((old['logits'].float() - new['logits'].float()).abs().max()))


def gate_b(model, serializer, episode, device):
    """Helper carried state versus FLA chunk kernel final state on real hidden input."""
    s = switch(model, serializer, cfg_for('m2'))
    data = encode(s, episode, device)
    lm = model.backbone.model.language_model
    captured = {}
    layer_index = 2
    def keep(module, args, kwargs):
        captured.setdefault('hidden', args[0].detach().clone())
    hook = lm.layers[layer_index].register_forward_pre_hook(keep, with_kwargs=True)
    with torch.inference_mode():
        model(**data)
    hook.remove()
    hidden = captured['hidden']
    layout = data['episodic_layout']
    layer = lm.layers[layer_index]
    torch.cuda.synchronize(); start = time.perf_counter()
    scorer = CommittedWriteScorer(layer, device)
    with torch.inference_mode():
        scores = scorer.score_sequence(hidden[0], layout.score_segments())
    torch.cuda.synchronize(); helper_seconds = time.perf_counter() - start
    cache = make_stream_cache_fp32(model.config.text_config)
    with torch.inference_mode():
        normed = layer.input_layernorm(hidden)
        layer.linear_attn(normed, cache_params=cache)
    native_state = cache.layers[layer_index].recurrent_states[0].float()
    rel = float((scorer.state - native_state).norm() / native_state.norm())
    visual = scores[layout.visual_matrix().to(device)]
    return dict(layer=layer_index, tokens=int(hidden.shape[1]), helper_seconds=helper_seconds,
                fla_backend=getattr(layer.linear_attn.chunk_gated_delta_rule, '__module__', '?'),
                final_state_relative_error_vs_fla=rel,
                visual_score_quantiles=[float(x) for x in torch.quantile(
                    visual.flatten().float(), torch.tensor([0, .1, .5, .9, 1.], device=device))],
                finite=bool(torch.isfinite(scores).all()))


def gate_c(model, serializer, arm, episode, device):
    cfg = cfg_for(arm)
    s = switch(model, serializer, cfg)
    spec = model.episodic_spec
    data = encode(s, episode, device)
    diagnostics = {}
    with torch.inference_mode():
        offline = model(**data, episodic_diagnostics=diagnostics)['logits'].float()
    targets = data['input_ids'][0, data['response_target_positions']].tolist()
    plans = diagnostics.get('plans', {})
    head = model.backbone.lm_head
    original = head.forward
    scores, cursor = [], [0]

    def forced(h):
        value = original(h)
        scores.append(value.float().clone())
        token = targets[cursor[0]]; cursor[0] += 1
        controlled = torch.full_like(value, -10000.); controlled[token] = 10000.
        return controlled
    session = StreamSession(model, s)
    session.reset(episode['episode_uid'], episode['instruction'])
    mismatched, compared, max_rows, decoded, overlaps = 0, 0, 0, [], []
    score_error = []
    head.forward = forced
    guard = derived_kv_guard(cfg, spec)
    try:
        with torch.inference_mode():
            for t, step in enumerate(episode['steps']):
                with Image.open(step['rgb_path']) as rgb:
                    result = session.observe(episode['episode_uid'], t, rgb)
                decoded.append(result['action_name'] == step['action_name'])
                promoted = len(session.episodic.promotions)
                assert session.observe(episode['episode_uid'], t, None) is result      # idempotent retry
                assert len(session.episodic.promotions) == promoted
                kv = kv_length(session.cache)
                max_rows = max(max_rows, kv)
                assert kv <= guard, (kv, guard)
                for layer in spec.attention_layers:
                    online = session.episodic.running[layer].ids
                    if layer in plans:
                        expected = plans[layer].group_ids(t)
                        compared += 1
                        mismatched += int(not np.array_equal(online, expected))
                        if expected.size:
                            overlaps.append(np.intersect1d(online, expected).size / expected.size)
                    group = session.episodic.groups[-1]
                    if 'visual_scores' in diagnostics and layer in group.scores:
                        off = diagnostics['visual_scores'][layer][t].numpy()
                        score_error.append(float(np.abs(group.scores[layer] - off).max() /
                                                 max(1e-6, float(np.abs(off).max()))))
                resident = session.episodic.resident[spec.attention_layers[0]]
                recent_tokens = (session.episodic.groups[-1].end - session.episodic.groups[0].start)
                assert resident.size == session.prefix_length + session.episodic.running[
                    spec.attention_layers[0]].ids.size + recent_tokens   # no hidden archive
    finally:
        head.forward = original
    assert cursor[0] == len(targets)
    actual = torch.stack(scores)
    diff = (actual - offline).abs()
    rms = float(diff.square().mean().sqrt())
    allclose = bool(torch.allclose(actual, offline, atol=.75, rtol=.03))
    session.reset('fresh', episode['instruction'])
    assert session.episodic.running[spec.attention_layers[0]].ids.size == 0
    return dict(arm=arm, dataset=episode['dataset'], steps=len(episode['steps']), tokens=len(targets),
                max_logit_error=float(diff.max()), rms_logit_error=rms,
                argmax_agreement=float((actual.argmax(-1) == offline.argmax(-1)).float().mean()),
                decoded_actions_match=float(np.mean(decoded)),
                plan_comparisons=compared, plan_id_mismatches=mismatched,
                mean_selected_overlap=float(np.mean(overlaps)) if overlaps else None,
                min_selected_overlap=float(np.min(overlaps)) if overlaps else None,
                median_cutoff_margin=(float(np.nanmedian(np.concatenate([pl.cutoff_margins for pl in plans.values()])))
                                      if plans and spec.needs_write_scores else None),
                max_relative_score_error=max(score_error) if score_error else None,
                max_resident_kv=max_rows, kv_guard=guard,
                passed=allclose and rms <= .15)


def gate_e(model, serializer, arm, episode, device, steps=520):
    cfg = cfg_for(arm)
    s = switch(model, serializer, cfg)
    spec = model.episodic_spec
    head = model.backbone.lm_head
    original = head.forward
    action_tokens = [ids for ids in s.action_ids[:3]]
    plan, cursor = [], [0]

    def forced(h):
        value = original(h)
        token = plan[cursor[0]]; cursor[0] += 1
        controlled = torch.full_like(value, -10000.); controlled[token] = 10000.
        return controlled
    session = StreamSession(model, s)
    session.reset('dry-stream', episode['instruction'])
    frames = [f['rgb_path'] for f in episode['steps']]
    peak, start = 0, time.perf_counter()
    head.forward = forced
    try:
        with torch.inference_mode():
            for t in range(steps):
                plan.extend(action_tokens[t % 3])
                with Image.open(frames[t % len(frames)]) as rgb:
                    session.observe('dry-stream', t, rgb)
                peak = max(peak, kv_length(session.cache))
    finally:
        head.forward = original
    return dict(arm=arm, steps=steps, peak_resident_kv=peak, guard=derived_kv_guard(cfg, spec),
                seconds=time.perf_counter() - start, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                final=session.episodic.summary())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--replay-steps', type=int, nargs='+', default=[64, 128])
    p.add_argument('--arms', nargs='+', default=['v0', 'm2', 'm2c'])
    args = p.parse_args()
    torch.manual_seed(429)
    model, serializer = load_model(cfg_for('m2'), args.model_path)
    model.eval()
    device = next(model.parameters()).device
    report = dict(note='Fixed expert observations and forced canonical responses; no navigation-quality claim.')
    short = fixture(args.manifest, 'r2rce', 24)
    report['A_feature_disabled'] = gate_a(model, serializer, short, device)
    print('GATE_A', json.dumps(report['A_feature_disabled']), flush=True)
    report['B_helper_vs_fla'] = [gate_b(model, serializer, fixture(args.manifest, d, 96), device)
                                 for d in ('r2rce', 'rxrce')]
    print('GATE_B', json.dumps(report['B_helper_vs_fla']), flush=True)
    replay = []
    for steps in args.replay_steps:
        for dataset in ('r2rce', 'rxrce'):
            episode = fixture(args.manifest, dataset, steps, rng_seed=steps)
            for arm in args.arms:
                replay.append(gate_c(model, serializer, arm, episode, device))
                print(json.dumps(replay[-1]), flush=True)
    report['C_replay'] = replay
    report['E_dry_stream'] = [gate_e(model, serializer, arm, fixture(args.manifest, 'rxrce', 40), device)
                              for arm in ('m2', 'v0')]
    a = report['A_feature_disabled']
    report['passed'] = bool(a['logits_bitwise_equal'] and a['loss_sum_equal']
                            and all(r['passed'] for r in replay)
                            and all(e['peak_resident_kv'] <= e['guard'] for e in report['E_dry_stream']))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'C_replay'}, indent=2), flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
