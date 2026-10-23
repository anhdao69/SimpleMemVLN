"""Ranker datasets from collected label episodes (``gdn_utility_near_tail_v2``).

``build`` turns per-episode label files into columnar arrays, one row per
(episode, labeled source, sampled patch):

* ``rows.npz``: identities (episode index, source, position, token ID),
  partition/domain/logger codes, ``labeled_sources`` (for sample-relative
  budget metrics), S0 scores per attention layer, and the raw
  near/tail/remaining aggregates per attention layer with their statuses;
* ``raw_a{layer}.npy``: gdn_raw_v2 fields [N, window GDN layers, Hv, F] of the
  three GDN layers preceding attention ``layer`` (L0/L1 read the last one);
* ``hidden_a{layer}.npy``: BF16 bits of the attention-layer input hidden state
  [N, d_model] (hidden-state controls only).

Aggregation is on raw utilities with strict coverage (missing is never zero);
targets are transformed only by the trainer, with fit-partition constants.
Partitions come from the frozen selection; dev/holdout rows never influence
a normalizer, kappa or weight.
"""
import json
from pathlib import Path
import numpy as np

PARTITION_CODES = {'ranker_fit': 0, 'ranker_dev': 1, 'ranker_holdout': 2}
DOMAIN_CODES = {'r2rce': 0, 'rxrce': 1}
LOGGER_CODES = {'C_RANDOM:7001': 0, 'C_RANDOM:7002': 1, 'S0:429': 2}
STATUS = {'complete': 0, 'missing': 1, 'not_applicable': 2, 'not_eligible': 3}
WINDOW = 3


def aggregate(utility, near_counts, tail_counts, near_support, tail_support):
    """Raw near/tail/remaining means for one episode.

    ``utility`` [S, A, R, Q] (query slots: near offsets then tail offsets).
    Returns float64 arrays [S, A, R] (NaN when null) and int8 statuses.
    """
    S, A, R, _ = utility.shape
    near = np.full((S, A, R), np.nan)
    tail = np.full((S, A, R), np.nan)
    remaining = np.full((S, A, R), np.nan)
    near_status = np.full((S, A, R), STATUS['missing'], np.int8)
    tail_status = np.full((S, A, R), STATUS['missing'], np.int8)
    rem_status = np.full((S, A, R), STATUS['missing'], np.int8)
    for s in range(S):
        n, f = int(near_counts[s]), int(tail_counts[s])
        N, F = int(near_support[s]), int(tail_support[s])
        if N + F == 0:
            near_status[s] = rem_status[s] = STATUS['not_eligible']
            tail_status[s] = STATUS['not_applicable']
            continue
        block = utility[s]
        nv = block[..., :n]
        ok_near = np.isfinite(nv).all(-1)
        near[s][ok_near] = nv[ok_near].mean(-1)
        near_status[s][ok_near] = STATUS['complete']
        if F == 0:
            tail_status[s] = STATUS['not_applicable']
            remaining[s][ok_near] = near[s][ok_near]
            rem_status[s][ok_near] = STATUS['complete']
            continue
        tv = block[..., n:n + f]
        ok_tail = np.isfinite(tv).all(-1)
        tail[s][ok_tail] = tv[ok_tail].mean(-1)
        tail_status[s][ok_tail] = STATUS['complete']
        ok = ok_near & ok_tail
        remaining[s][ok] = (N / (N + F)) * near[s][ok] + (F / (N + F)) * tail[s][ok]
        rem_status[s][ok] = STATUS['complete']
    if np.any(near[np.isfinite(near)] < 0):
        raise ValueError('Negative raw utility aggregate')
    return near, tail, remaining, near_status, tail_status, rem_status


def check_partitions(episodes):
    """Scenes, and therefore physical trajectories, never cross ranker partitions."""
    owner = {}
    for e in episodes:
        for key in (('scene', e['scene_id']), ('trajectory', e['dataset'], e['episode_uid'].rsplit(':', 1)[-1])):
            if owner.setdefault(key, e['partition']) != e['partition']:
                raise ValueError(f'{key} appears in {owner[key]} and {e["partition"]}')
    return owner


def episode_files(label_dir, selection, subset='full'):
    from scripts.memory_sweep.build_labels import episode_file
    episodes = [e for e in selection['episodes'] if subset == 'full' or e['pilot']]
    return episodes, [episode_file(label_dir, e['episode_uid']) for e in episodes]


def build(label_dir, out_dir, subset='full', with_hidden=True):
    label_dir, out = Path(label_dir), Path(out_dir)
    selection = json.loads((label_dir / 'selection.json').read_text())
    episodes, files = episode_files(label_dir, selection, subset)
    check_partitions(episodes)
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} label episodes missing, e.g. {missing[0]}')
    # Pass 1: sizes.
    sizes, metas = [], []
    for f in files:
        with np.load(f) as z:
            meta = json.loads(str(z['meta']))
            sizes.append(int(z['sources'].shape[0]) * int(meta['rows_per_source']))
            metas.append(meta)
    attention = metas[0]['attention_layers']
    gdn = metas[0]['gdn_layers']
    fields = metas[0]['store_fields']
    for meta, e in zip(metas, episodes):
        if (meta['attention_layers'], meta['gdn_layers'], meta['store_fields']) != (attention, gdn, fields):
            raise ValueError('Label episodes disagree on layer/field layout')
        if meta['episode'] != e or sum(meta['coverage']['missing'].values()):
            raise ValueError(f'Label file of {e["episode_uid"]} does not match the selection or is incomplete')
    total = sum(sizes)
    out.mkdir(parents=True, exist_ok=True)
    windows = {a: [gdn.index(g) for g in range(a - WINDOW, a)] for a in attention}
    raw = {a: np.lib.format.open_memmap(out / f'raw_a{a}.npy', mode='w+', dtype=np.float32,
                                        shape=(total, WINDOW, 32, len(fields))) for a in attention}
    hidden = {a: np.lib.format.open_memmap(out / f'hidden_a{a}.npy', mode='w+', dtype=np.uint16,
                                           shape=(total, metas[0].get('hidden_size', 2560)))
              for a in attention} if with_hidden else {}
    A = len(attention)
    cols = {k: [] for k in ('episode', 'source', 'position', 'token_id', 'partition', 'domain', 'logger',
                            'labeled_sources', 'steps')}
    s0, near_all, tail_all, rem_all, st = [], [], [], [], {k: [] for k in ('near', 'tail', 'remaining')}
    resident_any = []
    cursor = 0
    for index, (f, e, meta) in enumerate(zip(files, episodes, metas)):
        with np.load(f) as z:
            S, R = z['sources'].shape[0], meta['rows_per_source']
            n = S * R
            if n == 0:
                continue
            near_counts = (z['near_offsets'] >= 0).sum(-1)
            tail_counts = (z['tail_offsets'] >= 0).sum(-1)
            near, tail, remaining, ns, ts, rs = aggregate(
                z['utility'], near_counts, tail_counts, z['near_support'], z['tail_support'])
            feats = z['features']                                       # [S, R, G, Hv, F]
            for a in attention:
                raw[a][cursor:cursor + n] = feats[:, :, windows[a]].reshape(n, WINDOW, 32, len(fields))
            if with_hidden:
                hid = z['hidden_bf16']                                  # [S, R, A, d]
                for k, a in enumerate(attention):
                    hidden[a][cursor:cursor + n] = hid[:, :, k].reshape(n, -1)
            cols['episode'].append(np.full(n, index, np.int32))
            cols['source'].append(np.repeat(z['sources'], R).astype(np.int32))
            cols['position'].append(z['positions'].reshape(-1).astype(np.int16))
            cols['token_id'].append(z['token_ids'].reshape(-1))
            cols['partition'].append(np.full(n, PARTITION_CODES[e['partition']], np.int8))
            cols['domain'].append(np.full(n, DOMAIN_CODES[e['dataset']], np.int8))
            cols['logger'].append(np.full(n, LOGGER_CODES[f"{e['logger']}:{e['logger_seed']}"], np.int8))
            cols['labeled_sources'].append(np.full(n, S, np.int32))
            cols['steps'].append(np.full(n, e['steps'], np.int32))
            s0.append(z['s0'].reshape(n, A))
            # [S, A, R] -> rows [S*R, A]
            to_rows = lambda x: np.transpose(x, (0, 2, 1)).reshape(n, A)
            near_all.append(to_rows(near))
            tail_all.append(to_rows(tail))
            rem_all.append(to_rows(remaining))
            st['near'].append(to_rows(ns))
            st['tail'].append(to_rows(ts))
            st['remaining'].append(to_rows(rs))
            resident_any.append(to_rows((z['resident'] == 1).any(-1)))
            cursor += n
    if cursor != total:
        raise RuntimeError('Row count changed between passes')
    for array in list(raw.values()) + list(hidden.values()):
        array.flush()
    rows = {k: np.concatenate(v) for k, v in cols.items()}
    rows.update(s0=np.concatenate(s0), near=np.concatenate(near_all), tail=np.concatenate(tail_all),
                remaining=np.concatenate(rem_all), near_status=np.concatenate(st['near']),
                tail_status=np.concatenate(st['tail']), remaining_status=np.concatenate(st['remaining']),
                logger_resident_any=np.concatenate(resident_any))
    np.savez(out / 'rows.npz', **rows)
    manifest = dict(
        version='gdn_ranker_dataset_v1', subset=subset, rows=int(total), episodes=len(files),
        attention_layers=attention, gdn_layers=gdn, window=WINDOW, store_fields=fields,
        windows={str(a): list(range(a - WINDOW, a)) for a in attention},
        selection_sha256=json.loads((label_dir / 'LABEL_SOURCE.json').read_text())['selection_sha256']
        if (label_dir / 'LABEL_SOURCE.json').exists() else None,
        episode_uids=[e['episode_uid'] for e in episodes], partitions=PARTITION_CODES, domains=DOMAIN_CODES,
        loggers=LOGGER_CODES, status=STATUS, hidden=with_hidden)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return manifest


class RankerData:
    """Read-only view of a built dataset."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.manifest = json.loads((self.dir / 'manifest.json').read_text())
        with np.load(self.dir / 'rows.npz') as z:
            self.rows = {k: z[k] for k in z.files}
        self.attention_layers = self.manifest['attention_layers']
        self.fields = self.manifest['store_fields']

    def raw(self, layer):
        return np.load(self.dir / f'raw_a{layer}.npy', mmap_mode='r')

    def hidden(self, layer):
        return np.load(self.dir / f'hidden_a{layer}.npy', mmap_mode='r')

    def layer_column(self, layer):
        return self.attention_layers.index(layer)

    def mask(self, partition, layer, target='near'):
        """Rows of a partition with a complete target at this attention layer."""
        k = self.layer_column(layer)
        return ((self.rows['partition'] == PARTITION_CODES[partition])
                & (self.rows[f'{target}_status'][:, k] == STATUS['complete']))

    def weights(self, index):
        """Equal total weight per domain, equal weight per episode, rows equal within an episode."""
        episode, domain = self.rows['episode'][index], self.rows['domain'][index]
        weight = np.zeros(index.size, np.float64)
        for d in np.unique(domain):
            in_domain = domain == d
            episodes, counts = np.unique(episode[in_domain], return_counts=True)
            per_row = dict(zip(episodes.tolist(), (1.0 / counts).tolist()))
            weight[in_domain] = [per_row[x] / episodes.size for x in episode[in_domain].tolist()]
        return weight / weight.sum()

    def fields_torch(self, layer, index, window_layers):
        """dict field -> float32 torch [n, len(window_layers), Hv] for encode_features."""
        import torch
        window = self.manifest['windows'][str(layer)]
        cols = [window.index(g) for g in window_layers]
        block = np.asarray(self.raw(layer)[np.sort(index)][:, cols])
        order = np.argsort(np.argsort(index))                 # undo the sort for fancy indexing
        block = block[order]
        return {name: torch.from_numpy(np.ascontiguousarray(block[..., i])) for i, name in enumerate(self.fields)}
