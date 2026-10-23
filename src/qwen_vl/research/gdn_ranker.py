"""GDN-only learned rankers L0-L3 (Phase B): models, embedded artifacts, frozen scoring.

One predictor per full-attention layer scores every old visual row once,
from birth-time native GDN features of the preceding GDN layer(s), with one
scalar per logical KV row (shared by all KV heads, as M5 evicts rows):

    L0  F0_v1, previous GDN layer          Linear(96, 1)
    L1  F0_v1, previous GDN layer          Linear(96, 64) -> SiLU -> Linear(64, 1)
    L2  F0_v1, previous three GDN layers   Linear(288, 64) -> SiLU -> Linear(64, 1)
    L3  F1_v2, previous three GDN layers   Linear(576, 64) -> SiLU -> Linear(64, 1)

Inputs are flattened in increasing GDN-layer order, then native value head,
then the declared channel order, and standardized with the frozen
fit-partition mean/std (std floor 1e-4).  No age, birth time, episode length,
future offset or label enters a feature.  The ranker is fitted to
``log1p(Y / kappa[layer])`` of the declared target, but only its ordering is
used at runtime.

The artifact (normalizers + weights, little-endian base64 of exact FP32
bytes) is embedded in the memory contract with its canonical SHA-256, so a
checkpoint's navigation config is self-contained and a missing/altered
artifact fails closed.  Scoring runs under no_grad in FP64 on fixed
300-row groups (one observation frame) in offline training and serving
alike, and the result is rounded once to the FP32 priority dtype.
"""
import base64
import hashlib
import json
import math
import numpy as np
import torch

RAW_SCHEMA = 'gdn_raw_v2'
TARGET_NEAR16 = 'shadow_fixed_query_loo_near16_mean_v2'
CHANNELS = {'F0_v1': 3, 'F1_v2': 6}
ARMS = {
    'L0': dict(schema='F0_v1', window=1, hidden=None),
    'L1': dict(schema='F0_v1', window=1, hidden=64),
    'L2': dict(schema='F0_v1', window=3, hidden=64),
    'L3': dict(schema='F1_v2', window=3, hidden=64),
}
STD_FLOOR = 1e-4
GROUP_ROWS = 300
ARTIFACT_VERSION = 'gdn_learned_ranker_artifact_v1'


def canonical_sha256(value):
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def feature_layers(attention_layer, window, layer_types=None):
    layers = tuple(range(attention_layer - window, attention_layer))
    if layers[0] < 0:
        raise ValueError(f'Attention layer {attention_layer} has no {window} preceding layers')
    if layer_types is not None and any(layer_types[g] != 'linear_attention' for g in layers):
        raise ValueError(f'Feature window {layers} is not all native GDN layers')
    return layers


def input_width(arm, value_heads=32):
    spec = ARMS[arm]
    return spec['window'] * value_heads * CHANNELS[spec['schema']]


class RankerHead(torch.nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.hidden = hidden
        if hidden is None:
            self.out = torch.nn.Linear(width, 1)
        else:
            self.fc = torch.nn.Linear(width, hidden)
            self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        if self.hidden is not None:
            x = torch.nn.functional.silu(self.fc(x))
        return self.out(x).squeeze(-1)


def parameter_count(arm, layers=8, value_heads=32):
    head = RankerHead(input_width(arm, value_heads), ARMS[arm]['hidden'])
    return layers * sum(p.numel() for p in head.parameters())


def _encode(array):
    array = np.ascontiguousarray(np.asarray(array, dtype='<f4'))
    return dict(shape=list(array.shape), dtype='float32_le', b64=base64.b64encode(array.tobytes()).decode())


def _decode(record):
    if record.get('dtype') != 'float32_le':
        raise ValueError('Ranker tensors are little-endian FP32')
    array = np.frombuffer(base64.b64decode(record['b64']), dtype='<f4').copy()
    shape = tuple(int(x) for x in record['shape'])
    if array.size != int(np.prod(shape)):
        raise ValueError('Ranker tensor size does not match its shape')
    return array.reshape(shape)


def build_artifact(arm, heads, normalizers, kappa, provenance, value_heads=32, target_id=TARGET_NEAR16):
    """``heads``: attention layer -> RankerHead; ``normalizers``: layer -> (mean, std)."""
    spec = ARMS[arm]
    layers = {}
    for attention in sorted(heads):
        mean, std = (np.asarray(x, dtype=np.float32) for x in normalizers[attention])
        if np.any(std < STD_FLOOR) or not (np.isfinite(mean).all() and np.isfinite(std).all()):
            raise ValueError('Normalizer std must be floored at 1e-4 and finite')
        tensors = {name: _encode(p.detach().float().cpu().numpy())
                   for name, p in heads[attention].state_dict().items()}
        layers[str(attention)] = dict(
            feature_layers=list(feature_layers(attention, spec['window'])),
            mean=_encode(mean), std=_encode(std), tensors=tensors)
    return dict(version=ARTIFACT_VERSION, ranker_id=arm, raw_feature_schema=RAW_SCHEMA,
                feature_schema=spec['schema'], window=spec['window'], hidden=spec['hidden'],
                value_heads=value_heads, channels=CHANNELS[spec['schema']], target_id=target_id,
                kappa={str(k): float(v) for k, v in kappa.items()}, layers=layers, provenance=provenance)


def validate_artifact(artifact, arm, attention_layers, layer_types=None, value_heads=32):
    """Strict structural check; returns the artifact unchanged."""
    spec = ARMS[arm]
    expected = dict(version=ARTIFACT_VERSION, ranker_id=arm, raw_feature_schema=RAW_SCHEMA,
                    feature_schema=spec['schema'], window=spec['window'], hidden=spec['hidden'],
                    value_heads=value_heads, channels=CHANNELS[spec['schema']], target_id=TARGET_NEAR16)
    for key, value in expected.items():
        if artifact.get(key) != value:
            raise ValueError(f'Ranker artifact {key}={artifact.get(key)!r}, {arm} requires {value!r}')
    if set(artifact.get('layers', {})) != {str(a) for a in attention_layers}:
        raise ValueError('Ranker artifact must cover exactly the episodic attention layers')
    width = input_width(arm, value_heads)
    for attention in attention_layers:
        record = artifact['layers'][str(attention)]
        if tuple(record['feature_layers']) != feature_layers(attention, spec['window'], layer_types):
            raise ValueError(f'Feature window of layer {attention} differs from the {arm} contract')
        mean, std = _decode(record['mean']), _decode(record['std'])
        if mean.shape != (width,) or std.shape != (width,) or np.any(std < STD_FLOOR):
            raise ValueError('Bad normalizer')
        head = RankerHead(width, spec['hidden'])
        state = {name: torch.from_numpy(_decode(t)) for name, t in record['tensors'].items()}
        head.load_state_dict(state, strict=True)              # shape/name check
        for tensor in state.values():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError('Nonfinite ranker weight')
    return artifact


class FrozenRanker:
    """FP64 frozen per-layer scorer built from a validated artifact (no autograd, no parameters)."""

    _cache = {}

    def __init__(self, artifact):
        self.arm, self.schema = artifact['ranker_id'], artifact['feature_schema']
        self.hidden, self.window = artifact['hidden'], artifact['window']
        self.layers = {}
        for attention, record in artifact['layers'].items():
            tensors = {name: torch.from_numpy(_decode(t)).double() for name, t in record['tensors'].items()}
            self.layers[int(attention)] = dict(
                feature_layers=tuple(record['feature_layers']),
                mean=torch.from_numpy(_decode(record['mean'])).double(),
                std=torch.from_numpy(_decode(record['std'])).double(), **tensors)
        self.devices = {}

    @classmethod
    def load(cls, artifact_json, sha):
        if sha not in cls._cache:
            artifact = json.loads(artifact_json)
            if canonical_sha256(artifact) != sha:
                raise ValueError('Ranker artifact SHA-256 mismatch')
            cls._cache[sha] = cls(artifact)
        return cls._cache[sha]

    def _params(self, attention, device):
        key = (attention, str(device))
        if key not in self.devices:
            self.devices[key] = {k: (v.to(device) if torch.is_tensor(v) else v)
                                 for k, v in self.layers[attention].items()}
        return self.devices[key]

    @torch.no_grad()
    def score(self, attention, features):
        """FP32 priorities [N] from features [N, width] (N a multiple of 300 rows)."""
        if features.shape[0] % GROUP_ROWS:
            raise ValueError('Ranker scores whole 300-row observation frames')
        p = self._params(attention, features.device)
        if not bool(torch.isfinite(features).all()):
            raise ValueError('Nonfinite ranker feature')
        out = []
        for chunk in features.split(GROUP_ROWS):
            z = (chunk.double() - p['mean']) / p['std']
            if self.hidden is None:
                y = z @ p['out.weight'].T + p['out.bias']
            else:
                h = torch.nn.functional.silu(z @ p['fc.weight'].T + p['fc.bias'])
                y = h @ p['out.weight'].T + p['out.bias']
            out.append(y.squeeze(-1))
        scores = torch.cat(out).float()
        if not bool(torch.isfinite(scores).all()):
            raise ValueError('Nonfinite learned priority')
        return scores


def flatten_window(encoded_by_layer, layers):
    """Concatenate per-layer [N, Hv, C] encodings in GDN-layer, head, channel order -> [N, width]."""
    return torch.cat([encoded_by_layer[g].reshape(encoded_by_layer[g].shape[0], -1) for g in layers], -1)


def huber(prediction, target, weight, delta=1.0):
    error = (prediction - target).abs()
    loss = torch.where(error <= delta, 0.5 * error ** 2, delta * (error - 0.5 * delta))
    return (loss * weight).sum() / weight.sum()


def kappa_for(raw_targets):
    """``max(median positive fit target, 1e-12)`` and a degenerate flag."""
    raw = np.asarray(raw_targets, dtype=np.float64)
    positive = raw[raw > 0]
    if positive.size == 0:
        return 1e-12, True
    return max(float(np.median(positive)), 1e-12), False


def transform_target(raw, kappa):
    return np.log1p(np.asarray(raw, dtype=np.float64) / kappa)


def init_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return torch.Generator().manual_seed(seed)


def round_half(value):
    return math.floor(value + 0.5)
