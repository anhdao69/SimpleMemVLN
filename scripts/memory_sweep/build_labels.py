"""Phase-B labels: SHADOW_FIXED_QUERY_LOO utilities and gdn_raw_v2 features (``gdn_utility_near_tail_v2``).

``select`` freezes the scene partitions and the labeled training episodes
before any utility exists:

* every training scene goes wholly to ``ranker_fit``/``ranker_dev``/
  ``ranker_holdout`` (~70/15/15 of episodes, greedy deficit assignment in
  SHA-256 scene order), so all instructions of a physical trajectory share a
  partition;
* per partition, per domain (R2R/RxR) and per global length quartile a fixed
  quota (fit 90, dev 19, holdout 19 -> 1,024 episodes) is drawn in a
  scene-round-robin SHA-256 order; the S6/S8 calibration episodes never enter
  the holdout;
* the 256-episode pilot is the first 26 (fit) / 6 (dev) of every cell; the
  64-episode dense-quality subset is the first 4 fit + 4 dev of every cell;
* the logger of each episode is fixed by hash: random_hash_v1 seed 7001,
  seed 7002, or S0 (M5_E1's own qualified memory), all on frozen M5_E1.

``collect`` (torchrun, one GPU per rank) replays each episode teacher-forced
through frozen M5_E1 under its logger.  Passive hooks never alter execution:

* each of the 24 GDN layers: ``GdnFeatureExtractor`` gdn_raw_v2 fields at the
  sampled rows (the same extractor the learned arms use); at the eight
  scoring layers also the qualified legacy S0 score;
* each full-attention layer: the native BF16 Q/gate/K/V (recomputed with the
  module's own projections, norms and rotary embedding on the full
  sequence), its logger plan, and the attention-layer input hidden state at
  the sampled rows (hidden-state controls).  For every requested pre-action
  query the reference context is rebuilt and the unmodified module output is
  reconstructed in FP64 against the actual module output (and against a
  bank-ablated context, to show the check discriminates); then every sampled
  candidate of every source that requests this query gets its exact FP64
  shadow LOO utility.

Labels are per (source, attention layer, sampled row, requested query); a
value that cannot be computed is NaN with a reason, never zero.  Horizon
plans, sampled offsets, T and residency are label metadata only.

    torchrun --standalone --nproc_per_node=4 -m scripts.memory_sweep.build_labels collect ...
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch

from qwen_vl.research.gdn_label_horizons import (
    PROTOCOL, RECENT_GROUPS, displaced_row, make_horizon_plan, plan_sha256, sample_patch_positions)

TAG = 'gdn_utility_near_tail_v2_selection'
LABEL_SEED = 429
CONTEXT_SEEDS = (429, 430)            # primary; second only on the dense-quality subset
PARTITIONS = ('ranker_fit', 'ranker_dev', 'ranker_holdout')
FRACTIONS = dict(ranker_fit=0.70, ranker_dev=0.15, ranker_holdout=0.15)
QUOTA = dict(ranker_fit=90, ranker_dev=19, ranker_holdout=19)
PILOT = dict(ranker_fit=26, ranker_dev=6)
DENSE = dict(ranker_fit=4, ranker_dev=4)
DENSE_SOURCES = 4
DOMAINS, STRATA = ('r2rce', 'rxrce'), 4
LOGGERS = (('C_RANDOM', 7001), ('C_RANDOM', 7002), ('S0', 429))
STORE_FIELDS = ('beta', 'log_decay_g', 'value_norm', 'error_norm', 'committed_norm',
                'prediction_norm', 'prediction_dot_value', 'cosine_error', 'key_norm_sq')
GRID = (15, 20)
ROWS_PER_SOURCE = 24
MAX_QUERIES = 8


def _hash(*parts):
    return hashlib.sha256('|'.join(str(p) for p in parts).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def logger_for(uid):
    return LOGGERS[int(_hash(PROTOCOL, 'logger', uid), 16) % len(LOGGERS)]


# ---------------------------------------------------------------- selection
def partition_scenes(counts):
    """scene -> partition, greedy largest-deficit in SHA-256 scene order (ties: fit, dev, holdout)."""
    total = sum(counts.values())
    assigned, filled = {}, dict.fromkeys(PARTITIONS, 0)
    for scene in sorted(counts, key=lambda s: _hash(TAG, 'scene', s)):
        part = max(PARTITIONS, key=lambda p: (FRACTIONS[p] * total - filled[p], -PARTITIONS.index(p)))
        assigned[scene] = part
        filled[part] += counts[scene]
    return assigned


def round_robin(rows):
    """Scene round-robin: (rank of the episode inside its scene by hash, hash)."""
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row['scene_id']].append(row)
    ordered = []
    for scene_rows in by_scene.values():
        scene_rows.sort(key=lambda r: _hash(TAG, 'episode', r['episode_uid']))
        ordered += [(rank, _hash(TAG, 'episode', r['episode_uid']), r) for rank, r in enumerate(scene_rows)]
    return [r for _, _, r in sorted(ordered, key=lambda x: (x[0], x[1]))]


def select(manifest, calibration_selection=()):
    rows = []
    with open(manifest) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e['split'] != 'train':
                    raise ValueError('Labels use training episodes only')
                rows.append(dict(episode_uid=e['episode_uid'], dataset=e['dataset'], scene_id=e['scene_id'],
                                 steps=len(e['steps'])))
    counts = defaultdict(int)
    for r in rows:
        counts[r['scene_id']] += 1
    partition = partition_scenes(counts)
    bounds = {}
    for domain in DOMAINS:
        lengths = sorted(r['steps'] for r in rows if r['dataset'] == domain)
        bounds[domain] = [lengths[len(lengths) * k // STRATA] for k in range(1, STRATA)]
    excluded = set(calibration_selection)
    cells = defaultdict(list)
    for r in rows:
        r['partition'] = partition[r['scene_id']]
        r['stratum'] = int(np.searchsorted(bounds[r['dataset']], r['steps'], side='right'))
        if r['steps'] <= RECENT_GROUPS:                       # no eligible future query at all
            continue
        if r['partition'] == 'ranker_holdout' and r['episode_uid'] in excluded:
            continue
        cells[(r['partition'], r['dataset'], r['stratum'])].append(r)
    chosen = []
    for part in PARTITIONS:
        for domain in DOMAINS:
            for stratum in range(STRATA):
                pool = round_robin(cells[(part, domain, stratum)])
                if len(pool) < QUOTA[part]:
                    raise ValueError(f'Cell {part}/{domain}/{stratum} has {len(pool)} < {QUOTA[part]} episodes')
                for rank, r in enumerate(pool[:QUOTA[part]]):
                    model, seed = logger_for(r['episode_uid'])
                    chosen.append(dict(r, cell_rank=rank, pilot=rank < PILOT.get(part, 0),
                                       dense=rank < DENSE.get(part, 0), logger=model, logger_seed=seed))
    scene_counts = {p: sum(1 for s in partition.values() if s == p) for p in PARTITIONS}
    episode_counts = {p: sum(c for s, c in counts.items() if partition[s] == p) for p in PARTITIONS}
    return dict(version=TAG, protocol=PROTOCOL, label_seed=LABEL_SEED, context_seeds=list(CONTEXT_SEEDS),
                partition=partition, scene_counts=scene_counts, training_episode_counts=episode_counts,
                length_bounds=bounds, quotas=QUOTA, pilot=PILOT, dense=DENSE,
                loggers=[list(x) for x in LOGGERS], excluded_from_holdout=sorted(excluded), episodes=chosen)


# ---------------------------------------------------------------- collection
def episode_file(out, uid):
    return Path(out) / 'episodes' / f'{_hash("episode-file", uid)[:24]}.npz'


def logger_model_config(base_config, logger, seed):
    """Navigation config of the logger on frozen M5_E1 weights (S0 = the checkpoint's own memory)."""
    from copy import deepcopy
    from qwen_vl.research.gdn_sweep_contract import arm_memory_config
    cfg = deepcopy(base_config)
    if logger == 'S0':
        return cfg
    cfg['memory'] = arm_memory_config(logger, selector_seed=seed)['memory']
    return cfg


def dense_source_choice(labeled):
    """Fixed spanning sources (before any utility): quantiles 0, 1/3, 2/3, 1 of the labeled sources."""
    n = len(labeled)
    picks = sorted({labeled[round(k * (n - 1) / (DENSE_SOURCES - 1))] for k in range(DENSE_SOURCES)})
    return picks


class EpisodeCollector:
    """Passive per-episode hooks; all writes go to host arrays owned by this object."""

    def __init__(self, model, row, layout, data, device):
        self.model, self.row, self.layout, self.device = model, row, layout, device
        text = model.backbone.model.language_model
        self.text = text
        config = text.config
        self.attention_layers = [i for i, t in enumerate(config.layer_types) if t == 'full_attention']
        self.gdn_layers = [i for i, t in enumerate(config.layer_types) if t == 'linear_attention']
        self.scoring = {a - 1: a for a in self.attention_layers}
        if any(config.layer_types[g] != 'linear_attention' for g in self.scoring):
            raise ValueError('Every attention layer must follow a GDN layer')
        uid = row['episode_uid']
        self.uid = uid
        T = layout.groups
        self.T = T
        self.recent = RECENT_GROUPS
        self.anchors = np.asarray(layout.observation_ends, dtype=np.int64) - 1
        targets = data['response_target_positions'].cpu().numpy()
        index = data['response_action_index'].cpu().numpy()
        for t in range(T):
            first = int(targets[index == t].min())
            if first != layout.observation_ends[t]:
                raise ValueError('Pre-action anchor is not the position before the first response token')
        grid = data['image_grid_thw'].cpu().numpy()
        if grid.shape != (T, 3) or not np.all(grid == np.asarray([1, 2 * GRID[0], 2 * GRID[1]])):
            raise ValueError(f'Unexpected image grids {np.unique(grid, axis=0).tolist()}')
        # Labeled sources: positive eligible future support M = T - (s + R) > 0.
        self.sources = list(range(max(T - self.recent, 0)))
        self.plans = {s: make_horizon_plan(uid, s, T, LABEL_SEED) for s in self.sources}
        self.positions = np.asarray([sample_patch_positions(LABEL_SEED, uid, s) for s in self.sources],
                                    dtype=np.int64).reshape(len(self.sources), ROWS_PER_SOURCE)
        starts = np.asarray(layout.visual_starts, dtype=np.int64)
        self.token_ids = starts[self.sources][:, None] + self.positions if self.sources else np.empty((0, 24), np.int64)
        self.dense_sources = dense_source_choice(self.sources) if row.get('dense') and self.sources else []
        self.dense_plans = {s: make_horizon_plan(uid, s, T, LABEL_SEED, dense=True) for s in self.dense_sources}
        dense_ids = (starts[self.dense_sources][:, None] + np.arange(300)) if self.dense_sources else np.empty((0, 300), np.int64)
        self.dense_ids = dense_ids
        # Rows whose features are archived (sparse rows plus dense frames).
        self.feature_rows = np.unique(np.concatenate([self.token_ids.reshape(-1), dense_ids.reshape(-1)]))
        self.row_index = {int(t): i for i, t in enumerate(self.feature_rows)}
        S, A, G = len(self.sources), len(self.attention_layers), len(self.gdn_layers)
        n = self.feature_rows.size
        self.features = np.full((n, G, 32, len(STORE_FIELDS)), np.nan, dtype=np.float32)
        self.s0 = np.full((n, A), np.nan, dtype=np.float32)
        self.hidden = np.zeros((S, ROWS_PER_SOURCE, A, config.hidden_size), dtype=np.uint16)
        self.utility = np.full((S, A, ROWS_PER_SOURCE, MAX_QUERIES), np.nan, dtype=np.float64)
        self.resident = np.full((S, A, ROWS_PER_SOURCE, MAX_QUERIES), -1, dtype=np.int8)
        self.displaced = np.full((S, A, MAX_QUERIES), -1, dtype=np.int64)
        dq = max([len(p.offsets) for p in self.dense_plans.values()], default=0)
        D = len(self.dense_sources)
        self.dense_utility = np.full((D, len(CONTEXT_SEEDS), A, 300, dq), np.nan, dtype=np.float64)
        self.dense_resident = np.full((D, A, 300, dq), -1, dtype=np.int8)
        self.missing = defaultdict(int)
        self.dense_missing = defaultdict(int)
        self.recon = {}
        self.fallbacks = 0
        self.timing = defaultdict(float)
        self.segments = layout.score_segments()
        # query step -> [(source slot, query slot)] (sparse) and dense equivalents
        self.requests = defaultdict(list)
        for slot, s in enumerate(self.sources):
            for q, step in enumerate(self.plans[s].query_steps()):
                self.requests[step].append((slot, q))
        self.dense_requests = defaultdict(list)
        for slot, s in enumerate(self.dense_sources):
            for q, step in enumerate(self.dense_plans[s].query_steps()):
                self.dense_requests[step].append((slot, q))
        self.handles = []

    # -- hooks ----------------------------------------------------------------
    def install(self):
        layers = self.text.layers
        for gdn in self.gdn_layers:
            self.handles.append(layers[gdn].register_forward_pre_hook(self._gdn_hook(gdn), with_kwargs=True))
        for attention in self.attention_layers:
            self.handles.append(layers[attention].self_attn.register_forward_hook(
                self._attention_hook(attention), with_kwargs=True))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _gdn_hook(self, gdn):
        from qwen_vl.models.committed_write import CommittedWriteScorer
        from qwen_vl.models.gdn_features import GdnFeatureExtractor
        column = self.gdn_layers.index(gdn)

        def hook(module, args, kwargs):
            started = time.time()
            hidden = (args[0] if args else kwargs['hidden_states']).detach()[0]
            fields = GdnFeatureExtractor(module, hidden.device).sequence(hidden, self.segments)
            rows = torch.from_numpy(self.feature_rows).to(hidden.device)
            stacked = torch.stack([fields[name][rows] for name in STORE_FIELDS], -1)      # [n, 32, F]
            if not bool(torch.isfinite(stacked).all()):
                raise ValueError(f'Nonfinite gdn_raw_v2 feature at layer {gdn}')
            self.features[:, column] = stacked.cpu().numpy()
            if gdn in self.scoring:
                scores = CommittedWriteScorer(module, hidden.device).score_sequence(hidden, self.segments)
                self.s0[:, self.attention_layers.index(self.scoring[gdn])] = scores[rows].float().cpu().numpy()
            self.timing['features'] += time.time() - started
        return hook

    def _attention_hook(self, layer):
        from qwen_vl.research.kv_utility_labels import FixedQueryContext, native_qkv, visible_rows
        a_index = self.attention_layers.index(layer)

        def hook(module, args, kwargs, output):
            started = time.time()
            hidden = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            if module.o_proj.bias is not None:
                raise ValueError('Utility labels assume a bias-free o_proj (a bias would cancel anyway)')
            plan = kwargs.get('episodic_plan')
            if plan is None or plan.layer_idx != layer:
                raise ValueError(f'Missing logger plan for attention layer {layer}')
            plan.verify()
            query, gate, K, V = native_qkv(module, hidden, kwargs['position_embeddings'])
            anchors = torch.from_numpy(self.anchors).to(hidden.device)
            Q = query[:, anchors].transpose(0, 1).double()                              # [T, Hq, D]
            Z = torch.sigmoid(gate[anchors].double())                                   # [T, Hq, D]
            actual = output[0][0][anchors].double()                                     # [T, d_model]
            w_o = module.o_proj.weight.double()
            if self.sources:
                rows = torch.from_numpy(self.token_ids.reshape(-1)).to(hidden.device)
                self.hidden[:, :, a_index] = (hidden[0][rows].contiguous().view(torch.int16).cpu().numpy()
                                              .view(np.uint16).reshape(len(self.sources), ROWS_PER_SOURCE, -1))
            del query, gate
            errors, ablated = [], []
            layout = self.layout
            for u in range(self.T):
                bank = np.asarray(plan.group_ids(u), dtype=np.int64)
                first = layout.group_spans[max(0, u - self.recent + 1)][0]
                p = int(self.anchors[u])
                ids = visible_rows(layout.prefix, bank, first, p)
                index = torch.from_numpy(ids).to(hidden.device)
                ctx = FixedQueryContext(Q[u], Z[u], K[index].double(), V[index].double(), module.scaling, w_o)
                y = ctx.output()
                errors.append(float((y - actual[u]).norm() / actual[u].norm().clamp_min(1e-30)))
                if bank.size:
                    keep = torch.from_numpy(np.concatenate((np.arange(layout.prefix),
                                                            np.arange(first, p + 1)))).to(hidden.device)
                    ab = FixedQueryContext(Q[u], Z[u], K[keep].double(), V[keep].double(), module.scaling, w_o)
                    ablated.append((errors[-1], float((ab.output() - actual[u]).norm()
                                                      / actual[u].norm().clamp_min(1e-30))))
                if u in self.requests:
                    self._label(ctx, layer, a_index, u, bank, K, V, self.requests[u], dense=False)
                if u in self.dense_requests:
                    self._label(ctx, layer, a_index, u, bank, K, V, self.dense_requests[u], dense=True)
                self.fallbacks += ctx.fallbacks
            pairs = np.asarray(ablated).reshape(-1, 2)
            self.recon[layer] = dict(
                max_rel_error=max(errors), mean_rel_error=float(np.mean(errors)),
                median_rel_error=float(np.median(errors)), queries=len(errors),
                bank_queries=int(pairs.shape[0]),
                bank_ablated_mean_rel_error=float(pairs[:, 1].mean()) if pairs.size else None,
                bank_ablated_median_rel_error=float(np.median(pairs[:, 1])) if pairs.size else None,
                # fraction of bank queries where dropping the bank moves the output further from
                # the actual module output than the reference reconstruction error
                bank_ablation_detected=float((pairs[:, 1] > pairs[:, 0]).mean()) if pairs.size else None)
            self.timing['labels'] += time.time() - started
        return hook

    def _label(self, ctx, layer, a_index, u, bank, K, V, requests, dense):
        prefix = self.layout.prefix
        budget = self.model.episodic_spec.budget_tokens
        seeds = CONTEXT_SEEDS if dense else CONTEXT_SEEDS[:1]
        missing = self.dense_missing if dense else self.missing
        first = self.layout.group_spans[max(0, u - self.recent + 1)][0]
        for slot, q in requests:
            source = (self.dense_sources if dense else self.sources)[slot]
            ids = self.dense_ids[slot] if dense else self.token_ids[slot]
            if u < source + self.recent:
                raise RuntimeError('Requested query precedes eligibility')
            if ids.max() >= first or ids.min() < prefix:
                raise RuntimeError('Candidate is not an eligible old-bank row')
            resident = np.isin(ids, bank)
            res_slots, shadow_slots = np.nonzero(resident)[0], np.nonzero(~resident)[0]
            res_util = (ctx.resident(prefix + np.searchsorted(bank, ids[res_slots])).cpu().numpy()
                        if res_slots.size else None)
            for c, seed in enumerate(seeds):
                values = np.full(ids.size, np.nan)
                if res_slots.size:                  # resident rows keep the reference context
                    values[res_slots] = res_util
                if shadow_slots.size:
                    if bank.size != budget:         # impossible under top-B; counted, never zero
                        missing['nonresident_unsaturated_bank'] += int(shadow_slots.size)
                    else:
                        d = displaced_row(seed, self.uid, source, u, layer, bank)
                        rows = torch.from_numpy(ids[shadow_slots]).to(K.device)
                        values[shadow_slots] = ctx.shadow(
                            K[rows].double(), V[rows].double(),
                            prefix + int(np.searchsorted(bank, d))).cpu().numpy()
                        if not dense:
                            self.displaced[slot, a_index, q] = d
                if dense:
                    self.dense_utility[slot, c, a_index, :, q] = values
                else:
                    self.utility[slot, a_index, :, q] = values
            (self.dense_resident if dense else self.resident)[slot, a_index, :, q] = resident.astype(np.int8)

    # -- output ---------------------------------------------------------------
    def arrays(self):
        S = len(self.sources)
        near = np.full((S, 4), -1, np.int16)
        tail = np.full((S, 4), -1, np.int16)
        for slot, s in enumerate(self.sources):
            plan = self.plans[s]
            near[slot, :len(plan.near_offsets)] = plan.near_offsets
            tail[slot, :len(plan.tail_offsets)] = plan.tail_offsets
        index = np.asarray([[self.row_index[int(t)] for t in row] for row in self.token_ids], dtype=np.int64) \
            if S else np.empty((0, ROWS_PER_SOURCE), np.int64)
        out = dict(
            sources=np.asarray(self.sources, np.int32), positions=self.positions.astype(np.int16),
            token_ids=self.token_ids, near_offsets=near, tail_offsets=tail,
            near_support=np.asarray([self.plans[s].near_support for s in self.sources], np.int16),
            tail_support=np.asarray([self.plans[s].tail_support for s in self.sources], np.int16),
            plan_sha256=np.asarray([plan_sha256(self.plans[s]) for s in self.sources]),
            utility=self.utility, resident=self.resident, displaced=self.displaced,
            features=self.features[index] if S else np.empty((0, ROWS_PER_SOURCE) + self.features.shape[1:], np.float32),
            s0=self.s0[index] if S else np.empty((0, ROWS_PER_SOURCE, len(self.attention_layers)), np.float32),
            hidden_bf16=self.hidden)
        if self.dense_sources:
            dindex = np.asarray([[self.row_index[int(t)] for t in row] for row in self.dense_ids], dtype=np.int64)
            dq = self.dense_utility.shape[-1]
            offsets = np.full((len(self.dense_sources), dq), -1, np.int16)
            for slot, s in enumerate(self.dense_sources):
                offs = self.dense_plans[s].offsets
                offsets[slot, :len(offs)] = offs
            out.update(dense_sources=np.asarray(self.dense_sources, np.int32), dense_offsets=offsets,
                       dense_near_support=np.asarray([self.dense_plans[s].near_support for s in self.dense_sources], np.int16),
                       dense_utility=self.dense_utility, dense_resident=self.dense_resident,
                       dense_features=self.features[dindex], dense_s0=self.s0[dindex])
        return out

    def check_complete(self):
        """Every requested (source, layer, row, query) has a finite label or a counted missing reason."""
        expected = 0
        for slot, s in enumerate(self.sources):
            expected += len(self.plans[s].offsets) * len(self.attention_layers) * ROWS_PER_SOURCE
        finite = int(np.isfinite(self.utility).sum())
        missing = sum(self.missing.values())
        if not bool(np.isfinite(self.features).all()) or not bool(np.isfinite(self.s0).all()):
            raise RuntimeError('Missing features or S0 scores')
        if finite + missing != expected:
            raise RuntimeError(f'Label coverage mismatch: {finite} finite + {missing} missing != {expected}')
        if np.any(self.utility[np.isfinite(self.utility)] < 0):
            raise RuntimeError('Negative utility')
        return dict(expected=expected, finite=finite, missing=dict(self.missing),
                    dense_finite=int(np.isfinite(self.dense_utility).sum()), dense_missing=dict(self.dense_missing))


def collect(args):
    import torch.distributed as dist
    from qwen_vl.train.vln_runtime import load_checkpoint
    from qwen_vl.research.episodic_contract import parse_episodic_spec
    dist.init_process_group('gloo')
    rank, world = dist.get_rank(), dist.get_world_size()
    selection = json.loads(Path(args.selection).read_text())
    episodes = [e for e in selection['episodes'] if args.subset == 'full' or e['pilot']]
    if args.limit:
        episodes = episodes[:args.limit]
    if args.pick:                                   # smoke/profiling: explicit indices into the subset
        episodes = [episodes[int(i)] for i in args.pick.split(',')]
    wanted = {e['episode_uid']: e for e in episodes}
    data_rows = {}
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                if e['episode_uid'] in wanted:
                    data_rows[e['episode_uid']] = e
    if set(data_rows) != set(wanted):
        raise ValueError('Selected episodes missing from the manifest')
    # Longest-first greedy balance across ranks (deterministic).
    load = [0] * world
    mine = []
    for e in sorted(episodes, key=lambda e: (-e['steps'], e['episode_uid'])):
        owner = min(range(world), key=lambda r: (load[r], r))
        load[owner] += e['steps'] ** 1.3
        if owner == rank:
            mine.append(e)
    out = Path(args.out)
    (out / 'episodes').mkdir(parents=True, exist_ok=True)
    model, serializer = load_checkpoint(args.m5_e1, args.model_path)
    model.eval()
    base = model.navigation_config
    memory = base['memory']
    if (memory['mode'], memory.get('episodic_score'), memory['recent_groups_including_current'],
            memory['episodic_budget_tokens']) != ('episodic_window_v1', 'committed_mean_l2_v1', 4, 1200):
        raise ValueError('The frozen logger checkpoint must be M5_E1 (qualified M5 memory)')
    device = next(model.parameters()).device
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    configs = {}
    done = 0
    for e in mine:
        path = episode_file(out, e['episode_uid'])
        if path.exists():
            done += 1
            continue
        started = time.time()
        key = (e['logger'], e['logger_seed'])
        if key not in configs:
            configs[key] = logger_model_config(base, *key)
        cfg = configs[key]
        model.navigation_config = cfg
        model.episodic_spec = parse_episodic_spec(cfg, model.config.text_config)
        data = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in serializer.encode_episode(data_rows[e['episode_uid']]).items()}
        layout = data['episodic_layout']
        if layout.groups != e['steps']:
            raise ValueError('Layout groups differ from recorded decisions')
        collector = EpisodeCollector(model, e, layout, data, device)
        collector.install()
        try:
            with torch.inference_mode():
                result = model(**data)
        finally:
            collector.remove()
        coverage = collector.check_complete()
        meta = dict(
            protocol=PROTOCOL, episode=e, tokens=int(data['input_ids'].numel()), groups=layout.groups,
            layout=layout.to_dict(), logger=e['logger'], logger_seed=e['logger_seed'],
            memory=cfg['memory'], label_seed=LABEL_SEED, context_seeds=list(CONTEXT_SEEDS),
            attention_layers=collector.attention_layers, gdn_layers=collector.gdn_layers,
            store_fields=list(STORE_FIELDS), grid=list(GRID), rows_per_source=ROWS_PER_SOURCE,
            reconstruction=collector.recon, stable_fallbacks=collector.fallbacks, coverage=coverage,
            action_nll=float(result['loss_sum']) / result['num_actions'],
            seconds=time.time() - started, timing=dict(collector.timing),
            trajectory_end_reason='recorded_expert_stop', trajectory_truncated=False,
            m5_e1=str(args.m5_e1), dense=bool(collector.dense_sources))
        arrays = collector.arrays()
        tmp = path.with_suffix('.tmp.npz')
        np.savez(tmp, meta=np.asarray(json.dumps(meta, sort_keys=True)), **arrays)
        os.replace(tmp, path)
        done += 1
        print(f'rank{rank} {done}/{len(mine)} {e["episode_uid"]} T={layout.groups} tok={meta["tokens"]} '
              f'{meta["seconds"]:.1f}s recon_max={max(r["max_rel_error"] for r in collector.recon.values()):.4f} '
              f'recon_med={np.median([r["median_rel_error"] for r in collector.recon.values()]):.4f} '
              f'ablation_detected={min((r["bank_ablation_detected"] if r["bank_ablation_detected"] is not None else 1) for r in collector.recon.values()):.3f} '
              f'missing={sum(coverage["missing"].values())}', flush=True)
        del data, result, collector, arrays
        torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        files = [episode_file(out, e['episode_uid']) for e in episodes]
        absent = [str(f) for f in files if not f.exists()]
        if absent:
            raise RuntimeError(f'{len(absent)} episode files missing')
        print('LABELS_COMPLETE', args.subset, len(files), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('select')
    s.add_argument('--manifest', required=True)
    s.add_argument('--calibration', help='S-campaign calibration.json (its episodes never enter the holdout)')
    s.add_argument('--out', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--selection', required=True)
    c.add_argument('--manifest', required=True)
    c.add_argument('--m5-e1', required=True)
    c.add_argument('--model-path', required=True)
    c.add_argument('--subset', choices=('pilot', 'full'), required=True)
    c.add_argument('--limit', type=int, default=0)
    c.add_argument('--pick', default='', help='comma-separated indices into the subset (smoke only)')
    c.add_argument('--out', required=True)
    args = p.parse_args()
    if args.command == 'select':
        calibration = []
        if args.calibration:
            calibration = [r['episode_uid'] for r in json.loads(Path(args.calibration).read_text())['selection']]
        record = select(args.manifest, calibration)
        record['manifest_sha256'] = sha_file(args.manifest)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(record, indent=1, sort_keys=True) + '\n'
        out.write_text(text)
        cells = defaultdict(int)
        for e in record['episodes']:
            cells[(e['partition'], e['dataset'])] += 1
        print(json.dumps(dict(sha256=hashlib.sha256(text.encode()).hexdigest(), scenes=record['scene_counts'],
                              training_episodes=record['training_episode_counts'],
                              episodes=len(record['episodes']),
                              pilot=sum(e['pilot'] for e in record['episodes']),
                              dense=sum(e['dense'] for e in record['episodes']),
                              cells={f'{a}/{b}': n for (a, b), n in sorted(cells.items())}), indent=1))
    else:
        collect(args)


if __name__ == '__main__':
    main()
