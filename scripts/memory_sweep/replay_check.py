"""Real-model serving qualification of GDN-sweep arms on fixed expert histories.

For each arm on one checkpoint (default: the primary M5_E1 runtime swap, an
explicit ``qualified=false`` diagnostic, never a claim about that
checkpoint's own qualification), the offline training forward is compared
with the real ``StreamSession`` loop under forced canonical expert responses
over real R2R/RxR prefixes that cross many admissions and saturate the bank.
The inherited BF16 gates are unchanged: atol .75, rtol .03, RMS <= .15.

Also recorded (fixed-history replay, not navigation performance): per-step
selected-ID agreement offline versus serving, priority errors, bounded KV,
retry idempotency, reset, the teacher-forced action NLL, and selected-set
overlap fraction/Jaccard against the same checkpoint's legacy S0 plans.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from qwen_vl.research.episodic_contract import derived_kv_guard
from qwen_vl.stream.cache import kv_length
from qwen_vl.stream.session import StreamSession
from qwen_vl.train.vln_runtime import load_checkpoint
from scripts.vln.check_episodic import encode, fixture, switch

BF16_ATOL, BF16_RTOL, BF16_RMS = .75, .03, .15


def arm_config(base, memory_path):
    cfg = deepcopy(base)
    cfg['memory'] = yaml.safe_load(Path(memory_path).read_text())['memory']
    return cfg


def overlap(plans, reference):
    """Per-layer overlap |A∩B|/|B| and Jaccard against reference plans (equal-sized banks)."""
    fractions, jaccards = [], []
    for layer, plan in plans.items():
        ref = reference[layer]
        for t in range(plan.groups):
            a, b = plan.group_ids(t), ref.group_ids(t)
            if b.size:
                inter = np.intersect1d(a, b).size
                fractions.append(inter / b.size)
                jaccards.append(inter / np.union1d(a, b).size)
    if not fractions:
        return None
    return dict(mean_overlap=float(np.mean(fractions)), min_overlap=float(np.min(fractions)),
                p10_overlap=float(np.quantile(fractions, .1)), mean_jaccard=float(np.mean(jaccards)),
                min_jaccard=float(np.min(jaccards)))


def sweep_gate(model, serializer, cfg, name, episode, device):
    s = switch(model, serializer, cfg)
    spec = model.episodic_spec
    data = encode(s, episode, device)
    diagnostics = {}
    with torch.inference_mode():
        result = model(**data, episodic_diagnostics=diagnostics)
    offline = result['logits'].float()
    nll = float(result['loss_sum']) / result['num_actions']
    targets = data['input_ids'][0, data['response_target_positions']].tolist()
    plans = diagnostics.get('plans', {})
    head = model.backbone.lm_head
    original = head.forward
    logits, cursor = [], [0]

    def forced(h):
        value = original(h)
        logits.append(value.float().clone())
        token = targets[cursor[0]]; cursor[0] += 1
        controlled = torch.full_like(value, -10000.); controlled[token] = 10000.
        return controlled
    session = StreamSession(model, s)
    session.reset(episode['episode_uid'], episode['instruction'])
    mismatched, compared, max_rows, decoded, overlaps, score_error = 0, 0, 0, [], [], []
    guard = derived_kv_guard(cfg, spec)
    head.forward = forced
    try:
        with torch.inference_mode():
            for t, step in enumerate(episode['steps']):
                with Image.open(step['rgb_path']) as rgb:
                    out = session.observe(episode['episode_uid'], t, rgb)
                decoded.append(out['action_name'] == step['action_name'])
                promoted = len(session.episodic.promotions)
                assert session.observe(episode['episode_uid'], t, None) is out        # idempotent retry
                assert len(session.episodic.promotions) == promoted
                kv = kv_length(session.cache)
                max_rows = max(max_rows, kv)
                assert kv <= guard, (kv, guard)
                group = session.episodic.groups[-1]
                for layer in spec.attention_layers:
                    online = session.episodic.running[layer].ids
                    expected = plans[layer].group_ids(t)
                    compared += 1
                    mismatched += int(not np.array_equal(online, expected))
                    if expected.size:
                        overlaps.append(np.intersect1d(online, expected).size / expected.size)
                    if layer in group.scores:
                        if getattr(spec, 'is_frame', False):     # legacy EpisodicSpec: tokens only
                            off = float(diagnostics['frame_scores'][layer][t])
                            score_error.append(abs(float(group.scores[layer]) - off) / max(1e-6, abs(off)))
                        else:
                            off = diagnostics['visual_scores'][layer][t].numpy()
                            score_error.append(float(np.abs(group.scores[layer] - off).max()
                                                     / max(1e-6, float(np.abs(off).max()))))
                first = spec.attention_layers[0]
                recent_tokens = session.episodic.groups[-1].end - session.episodic.groups[0].start
                assert session.episodic.resident[first].size == (
                    session.prefix_length + session.episodic.running[first].ids.size + recent_tokens)
    finally:
        head.forward = original
    assert cursor[0] == len(targets)
    actual = torch.stack(logits)
    diff = (actual - offline).abs()
    rms = float(diff.square().mean().sqrt())
    allclose = bool(torch.allclose(actual, offline, atol=BF16_ATOL, rtol=BF16_RTOL))
    session.reset('fresh', episode['instruction'])
    assert session.episodic.running[spec.attention_layers[0]].ids.size == 0
    record = dict(arm=name, score=spec.score, version=spec.version, dataset=episode['dataset'],
                  steps=len(episode['steps']), tokens=len(targets), action_nll=nll,
                  max_logit_error=float(diff.max()), rms_logit_error=rms,
                  argmax_agreement=float((actual.argmax(-1) == offline.argmax(-1)).float().mean()),
                  decoded_actions_match=float(np.mean(decoded)), plan_comparisons=compared,
                  plan_id_mismatches=mismatched,
                  mean_selected_overlap=float(np.mean(overlaps)) if overlaps else None,
                  min_selected_overlap=float(np.min(overlaps)) if overlaps else None,
                  max_relative_score_error=max(score_error) if score_error else None,
                  max_resident_kv=max_rows, kv_guard=guard,
                  passed=allclose and rms <= BF16_RMS and float(np.mean(decoded)) == 1.0 and max_rows <= guard)
    return record, plans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--arm', nargs=2, action='append', metavar=('NAME', 'MEMORY_YAML'), required=True,
                   help='Arm name and memory config; the legacy checkpoint memory is S0_legacy')
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.manual_seed(429)
    model, serializer = load_checkpoint(args.checkpoint, args.model_path)
    model.eval()
    device = next(model.parameters()).device
    base = deepcopy(model.navigation_config)
    if base['memory'].get('episodic_score') != 'committed_mean_l2_v1':
        raise ValueError('The reference checkpoint must carry the legacy M5 (S0) memory')
    report = dict(checkpoint=str(args.checkpoint), qualified=False,
                  note='Runtime-swap diagnostic on fixed expert histories; no navigation-quality claim.',
                  gates=dict(atol=BF16_ATOL, rtol=BF16_RTOL, rms=BF16_RMS), replay=[])
    for dataset in ('r2rce', 'rxrce'):
        episode = fixture(args.manifest, dataset, args.steps, rng_seed=args.steps)
        reference, ref_plans = sweep_gate(model, serializer, base, 'S0_legacy', episode, device)
        report['replay'].append(reference)
        print(json.dumps(reference), flush=True)
        for name, memory in args.arm:
            record, plans = sweep_gate(model, serializer, arm_config(base, memory), name, episode, device)
            record['overlap_vs_S0_legacy'] = overlap(plans, ref_plans)
            report['replay'].append(record)
            print(json.dumps(record), flush=True)
    report['passed'] = all(r['passed'] for r in report['replay'])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
