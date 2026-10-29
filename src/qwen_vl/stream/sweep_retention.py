"""Per-layer retention rule for S8N frame memory (``episodic_window_gdn_sweep_v1``, frame arms S8N/S8W).

S8N keeps, per attention layer l, the running top-F frames by the birth-fixed drift priority p_s^l of
frame s. ``RetentionTilt`` replaces that priority by a per-layer linear rule over birth-fixed features

    priority_s^l = (1 - a_l) * log p_s^l + a_l * mean_l' log p_s^l' + b_l * log(1 + s),

so a = b = 0 ranks exactly as S8N (log is monotone); a_l > 0 moves layer l towards the frames all
layers find surprising, b_l > 0 towards later frames (S8N's raw drift favours the first frames of an
episode because ||S_before|| is small there), b_l < 0 towards the episode start. Every feature is
fixed when the frame is observed, so the running top-F selection stays exact and the rule never sees
the future. Checkpoints carry the rule as ``sweep_retention.json``; without it S8N is unchanged.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SWEEP_RETENTION_FILE = 'sweep_retention.json'
SCHEMA = 's8n_tilt_v1'
PARAMETERS = ('consensus', 'birth')


class RetentionTilt:
    def __init__(self, layers, consensus, birth):
        self.layers = [int(l) for l in layers]
        self.consensus = [float(v) for v in consensus]
        self.birth = [float(v) for v in birth]
        if len(set(self.layers)) != len(self.layers) or not self.layers:
            raise ValueError('Retention tilt needs distinct attention layers')
        for name in PARAMETERS:
            values = getattr(self, name)
            if len(values) != len(self.layers) or not all(math.isfinite(v) for v in values):
                raise ValueError(f'Retention tilt {name} must be {len(self.layers)} finite values')
        self.index = {l: i for i, l in enumerate(self.layers)}

    @classmethod
    def zeros(cls, layers):
        return cls(layers, [0.0] * len(layers), [0.0] * len(layers))

    def vector(self):
        """float64 [2L]: (consensus, birth) in layer order."""
        return np.asarray(self.consensus + self.birth, dtype=np.float64)

    @classmethod
    def from_vector(cls, layers, vector):
        vector = np.asarray(vector, dtype=np.float64)
        n = len(layers)
        if vector.shape != (2 * n,):
            raise ValueError(f'Retention tilt vector must have shape ({2 * n},)')
        return cls(layers, vector[:n].tolist(), vector[n:].tolist())

    def priority(self, scores, layer, step_id):
        """Tilted priority of one frame for ``layer``; ``scores``: attention layer -> drift priority."""
        if set(scores) != set(self.layers):
            raise ValueError('Retention tilt needs the drift priority of every attention layer')
        logs = {}
        for l in self.layers:
            p = float(scores[l])
            if not (math.isfinite(p) and p > 0):
                raise ValueError(f'Retention tilt needs positive finite drift priorities, got {p}')
            logs[l] = math.log(p)
        i = self.index[layer]
        mean = math.fsum(logs[l] for l in self.layers) / len(self.layers)
        a, b = self.consensus[i], self.birth[i]
        return np.float64((1.0 - a) * logs[layer] + a * mean + b * math.log1p(int(step_id)))

    def to_dict(self):
        return dict(schema=SCHEMA, layers=self.layers, consensus=self.consensus, birth=self.birth)

    @classmethod
    def from_dict(cls, data):
        if data.get('schema') != SCHEMA:
            raise ValueError(f'Unknown sweep retention schema {data.get("schema")!r}')
        return cls(data['layers'], data['consensus'], data['birth'])

    def sha256(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


def save_sweep_retention(tilt, checkpoint):
    path = Path(checkpoint) / SWEEP_RETENTION_FILE
    path.write_text(json.dumps(tilt.to_dict(), indent=1, sort_keys=True) + '\n')
    return path


def load_sweep_retention(checkpoint):
    """The checkpoint's retention tilt (``sweep_retention.json``), or None for plain S8N."""
    path = Path(checkpoint) / SWEEP_RETENTION_FILE
    if not path.exists():
        return None
    return RetentionTilt.from_dict(json.loads(path.read_text()))
