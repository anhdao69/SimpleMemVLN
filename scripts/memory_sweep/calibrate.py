"""Training-only S6/S8 calibration: frozen slow-head masks and drift floors.

128 training episodes (64 R2R + 64 English RxR15deg) are chosen by a fixed
length-stratified, scene-diverse hash rule before any navigation result is
inspected; no validation episode can enter.  For each checkpoint role the
policy runs its own qualified M5 memory (R4/B1200 committed writes) over the
complete expert episode.  Forward pre-hooks on the eight scoring GDN layers
only read the layer input and never alter execution; they record

* the per-episode mean native log decay ``g`` per value head over every real
  token (prepared exactly as the scorers prepare it), and
* ``||S_before||_F`` at every observation boundary from the WY boundary scan.

``tau_h = log 2 / max(-mean g_h, 1e-8)`` with equal R2R/RxR domain weight; the
eight largest-tau heads of each scoring layer (ties: lower head index) form
the S6 mask.  ``eps_l = max(1e-6, 1e-3 * median ||S_before_l||)`` is the S8
floor.  Masks/floors are frozen afterwards; tau is a decay-timescale estimate,
not a measured information lifetime.

Run with ``torchrun --standalone --nproc_per_node=4 -m scripts.memory_sweep.calibrate``.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist

TAG = 'gdn_sweep_calibration_v1'
PER_DOMAIN, STRATA, SLOW_HEADS, BOOTSTRAP = 64, 4, 8, 1000
CHUNK_TOKENS = 8192


def _hash(uid):
    return hashlib.sha256(f'{TAG}|{uid}'.encode()).hexdigest()


def select_episodes(manifest, per_domain=PER_DOMAIN, strata=STRATA):
    """Deterministic length-stratified, scene-diverse selection per domain."""
    rows = {'r2rce': [], 'rxrce': []}
    with open(manifest) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e['split'] != 'train':
                    raise ValueError('Calibration uses training episodes only')
                rows[e['dataset']].append((len(e['steps']), e['episode_uid'], e['scene_id']))
    chosen = []
    for domain in ('r2rce', 'rxrce'):
        ranked = sorted(rows[domain])                                   # (steps, uid): length rank
        bounds = [len(ranked) * k // strata for k in range(strata + 1)]
        used, quota = set(), per_domain // strata
        for stratum in range(strata):
            pool = sorted(ranked[bounds[stratum]:bounds[stratum + 1]], key=lambda r: _hash(r[1]))
            picked = []
            for row in pool:                                            # distinct scenes first
                if len(picked) < quota and row[2] not in used:
                    picked.append(row); used.add(row[2])
            for row in pool:                                            # then hash order
                if len(picked) < quota and row not in picked:
                    picked.append(row)
            if len(picked) != quota:
                raise ValueError(f'Stratum {domain}/{stratum} too small')
            chosen += [dict(dataset=domain, stratum=stratum, episode_uid=uid, steps=steps, scene_id=scene)
                       for steps, uid, scene in picked]
    return chosen


def boundary_spec(text_config):
    from qwen_vl.research.gdn_sweep_contract import ARM_REGISTRY, FIXED, GdnSweepSpec
    from qwen_vl.research.episodic_contract import SCORE_SEGMENTATION, scoring_pairs_from_layer_types
    unit, reduction, backend, statistic = ARM_REGISTRY['S8N']
    return GdnSweepSpec(
        version='episodic_window_gdn_sweep_v1', recent_groups=4, budget_tokens=1200, granularity=unit,
        score='S8N', scoring_pairs=scoring_pairs_from_layer_types(list(text_config.layer_types)),
        selector_seed=429, per_source_cap=None, score_backend=backend,
        score_segmentation=SCORE_SEGMENTATION, head_reduction=reduction, statistic=statistic, **FIXED)


def collect(model, serializer, episodes, device):
    from qwen_vl.models.committed_write import empty_conv_tail, prepare_gdn_inputs
    from qwen_vl.models.gdn_features import GdnSweepScorer
    from qwen_vl.stream.gdn_selectors import observation_segment_indices
    lm = model.backbone.model.language_model
    spec = boundary_spec(model.config.text_config)
    out = []
    for episode in episodes:
        data = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in serializer.encode_episode(episode).items()}
        layout = data['episodic_layout']
        segments, observations = layout.score_segments(), observation_segment_indices(layout)
        record = dict(episode_uid=episode['episode_uid'], dataset=episode['dataset'],
                      steps=len(episode['steps']), tokens=int(data['input_ids'].numel()), layers={})
        hooks = []
        for gdn, _ in spec.scoring_pairs:
            def hook(module, args, kwargs, gdn=gdn):
                hidden = (args[0] if args else kwargs['hidden_states']).detach()[0]
                tail = empty_conv_tail(module, hidden.device)
                total = torch.zeros(module.linear_attn.num_v_heads, dtype=torch.float64, device=hidden.device)
                for start in range(0, hidden.shape[0], CHUNK_TOKENS):
                    _, _, g, _, tail = prepare_gdn_inputs(module, hidden[start:start + CHUNK_TOKENS], tail)
                    if not bool(torch.isfinite(g).all()):
                        raise ValueError('Nonfinite native decay')
                    total += g.double().sum(0)
                _, stats = GdnSweepScorer(spec, gdn, module, hidden.device).score_sequence(
                    hidden, segments, observations)
                stats = stats.double().cpu()
                if not bool(torch.isfinite(stats).all()):
                    raise ValueError('Nonfinite boundary statistics')
                record['layers'][str(gdn)] = dict(
                    mean_g=(total / hidden.shape[0]).cpu().tolist(),
                    before_norm=stats[:, 2].tolist(), net_drift=stats[:, 0].tolist(),
                    transported_write=stats[:, 1].tolist())
            hooks.append(lm.layers[gdn].register_forward_pre_hook(hook, with_kwargs=True))
        try:
            with torch.inference_mode():
                result = model(**data)
        finally:
            for handle in hooks:
                handle.remove()
        record['action_nll'] = float(result['loss_sum']) / result['num_actions']
        out.append(record)
        del data, result
    return out


def slow_mask(mean_g, heads=SLOW_HEADS):
    tau = math.log(2) / np.maximum(-np.asarray(mean_g, dtype=np.float64), 1e-8)
    return sorted(sorted(range(tau.size), key=lambda h: (-tau[h], h))[:heads]), tau


def aggregate(records):
    from qwen_vl.research.gdn_sweep_contract import canonical_sha256
    layers = sorted(records[0]['layers'], key=int)
    domains = ('r2rce', 'rxrce')
    index = {d: [i for i, r in enumerate(records) if r['dataset'] == d] for d in domains}
    if any(len(v) != PER_DOMAIN for v in index.values()):
        raise ValueError('Calibration needs 64 episodes per domain')
    rng = np.random.default_rng(429)
    draws = [{d: rng.integers(0, len(index[d]), len(index[d])) for d in domains} for _ in range(BOOTSTRAP)]
    mask, floors, detail = {}, {}, {}
    for layer in layers:
        g = {d: np.asarray([records[i]['layers'][layer]['mean_g'] for i in index[d]]) for d in domains}
        domain_mean = {d: g[d].mean(0) for d in domains}
        heads, tau = slow_mask(0.5 * (domain_mean['r2rce'] + domain_mean['rxrce']))
        frequency, jaccard = np.zeros(tau.size), []
        for draw in draws:
            boot, _ = slow_mask(0.5 * sum(g[d][draw[d]].mean(0) for d in domains))
            frequency[boot] += 1
            jaccard.append(len(set(boot) & set(heads)) / len(set(boot) | set(heads)))
        before = np.concatenate([records[i]['layers'][layer]['before_norm'] for i in range(len(records))])
        first = np.asarray([records[i]['layers'][layer]['before_norm'][0] for i in range(len(records))])
        median = float(np.median(before))
        eps = max(1e-6, 1e-3 * median)
        mask[layer], floors[layer] = heads, eps
        detail[layer] = dict(
            mean_g_by_domain={d: domain_mean[d].tolist() for d in domains}, tau=tau.tolist(),
            slow_heads=heads, bootstrap_head_frequency=(frequency / BOOTSTRAP).tolist(),
            bootstrap_jaccard_mean=float(np.mean(jaccard)), bootstrap_jaccard_p05=float(np.quantile(jaccard, .05)),
            bootstrap_exact_fraction=float(np.mean(np.asarray(jaccard) == 1.0)),
            before_norm_median=median, before_norm_median_by_domain={
                d: float(np.median(np.concatenate([records[i]['layers'][layer]['before_norm'] for i in index[d]])))
                for d in domains},
            boundaries=int(before.size), floor=eps, floor_hit_fraction=float(np.mean(before < eps)),
            first_boundary_norm_min=float(first.min()), first_boundary_norm_median=float(np.median(first)))
    return dict(slow_head_mask=mask, slow_head_mask_sha256=canonical_sha256(mask),
                drift_denominator_floor=floors, drift_denominator_floor_sha256=canonical_sha256(floors),
                layers=detail, episodes=len(records),
                mean_action_nll=float(np.mean([r['action_nll'] for r in records])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--model-path', required=True, help='Original Qwen3.5-4B snapshot')
    p.add_argument('--m5-e1', help='M5_E1 export directory (navigation.json + pytorch_model.bin)')
    p.add_argument('--roles', nargs='+', default=['base', 'm5_e1'], choices=['base', 'm5_e1'])
    p.add_argument('--memory-config', required=True, help='Legacy qualified M5 memory (arm_m5.yaml)')
    p.add_argument('--runtime-profile', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    out = Path(args.out)
    from qwen_vl.train.vln_runtime import load_checkpoint, load_model, resolve_config
    root = Path(__file__).resolve().parents[2]
    selection = select_episodes(args.manifest)
    wanted = {row['episode_uid'] for row in selection}
    episodes = {}
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e['episode_uid'] in wanted:
                    episodes[e['episode_uid']] = e
    ordered = [episodes[row['episode_uid']] for row in selection]
    mine = ordered[rank::world]
    report = dict(version=TAG, rule=('64 R2R + 64 RxR training episodes; per domain four equal-count length '
                                     'strata; within a stratum distinct scenes first then SHA-256(tag|uid) order'),
                  manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest() if rank == 0 else None,
                  selection=selection,
                  selection_sha256=hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest(),
                  reference_memory='episodic_window_v1 committed_mean_l2_v1 R4/B1200 (qualified M5)',
                  checkpoints={})
    for role in args.roles:
        started = time.time()
        if role == 'base':
            cfg = resolve_config(root / 'configs/episodic/science_common.yaml', root / 'configs/vln_empty.yaml',
                                 args.memory_config, root / 'configs/vggt/g3_joint_one_epoch.yaml',
                                 args.runtime_profile)
            model, serializer = load_model(cfg, args.model_path)
            source = args.model_path
        else:
            model, serializer = load_checkpoint(args.m5_e1, args.model_path)
            source = args.m5_e1
        model.eval()
        memory = model.navigation_config['memory']
        if (memory['mode'], memory.get('episodic_score'), memory['recent_groups_including_current'],
                memory['episodic_budget_tokens']) != ('episodic_window_v1', 'committed_mean_l2_v1', 4, 1200):
            raise ValueError('Calibration must run under the qualified M5 memory')
        device = next(model.parameters()).device
        rows = collect(model, serializer, mine, device)
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        del model
        torch.cuda.empty_cache()
        if rank == 0:
            records = sorted((r for part in gathered for r in part),
                             key=lambda r: [row['episode_uid'] for row in selection].index(r['episode_uid']))
            out.mkdir(parents=True, exist_ok=True)
            (out / f'calibration_rows_{role}.json').write_text(json.dumps(records) + '\n')
            summary = aggregate(records)
            summary.update(source=str(source), seconds=time.time() - started)
            report['checkpoints'][role] = summary
            print('CALIBRATED', role, json.dumps(dict(mask=summary['slow_head_mask'],
                                                      floors=summary['drift_denominator_floor'])), flush=True)
        dist.barrier()
    if rank == 0:
        if set(report['checkpoints']) == {'base', 'm5_e1'}:
            a, b = (report['checkpoints'][r]['slow_head_mask'] for r in ('base', 'm5_e1'))
            report['mask_overlap_base_vs_m5_e1'] = {layer: len(set(a[layer]) & set(b[layer])) for layer in a}
        (out / 'calibration.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print('CALIBRATION_PASS', out / 'calibration.json', flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
