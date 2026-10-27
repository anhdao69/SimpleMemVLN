"""GPU gate before collection: DaggerSession reproduces StreamSession bitwise (model interpreter).

    python scripts/dagger/check_session.py --checkpoint S8N_EXPORT --model-path QWEN_BASE \
        --manifest joint.jsonl [--episode-uid UID] [--steps 32] [--out check.json]

1. decode parity: ``step`` without forcing returns ``observe``'s actions and leaves a
   bitwise-identical streaming state (GDN/conv, full-attention KV, positions, episodic bank,
   recent groups, scorer states), both unarmed and armed with ``intercept_with`` as during
   collection (decisions whose first token is not STOP's must be untouched);
2. forced parity: forcing ``observe``'s own actions reproduces that state bitwise;
3. interception: treating the policy's real first token at step k as STOP's commits exactly the
   expert action (state bitwise equal to forcing it), and both sessions then decode identically.
Exit status 1 on any mismatch; the JSON report names the first differing state path.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

from qwen_vl.contracts import ACTIONS
from qwen_vl.dagger.session import DaggerSession
from qwen_vl.stream.session import StreamSession
from qwen_vl.train.vln_runtime import load_checkpoint

SKIP = {'layer', 'spec', 'model', 'mixer'}


def tensor_digest(value):
    flat = value.detach().contiguous().reshape(-1).cpu()
    raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b''
    return [str(value.dtype), list(value.shape), hashlib.sha256(raw).hexdigest()]


def array_digest(value):
    value = np.ascontiguousarray(np.asarray(value))
    return [str(value.dtype), list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]


def object_digest(obj):
    out = {}
    for name, value in sorted(vars(obj).items()):
        if name in SKIP:
            continue
        if torch.is_tensor(value):
            out[name] = tensor_digest(value)
        elif isinstance(value, np.ndarray):
            out[name] = array_digest(value)
        elif isinstance(value, (list, tuple)) and value and all(torch.is_tensor(v) for v in value):
            out[name] = [tensor_digest(v) for v in value]
        elif isinstance(value, (bool, int, float, str)) or value is None:
            out[name] = value
    return out


def state_digest(session):
    digest = dict(next_step=session.next_step, logical=session.positions.logical_token_count,
                  mrope=session.positions.mrope_cursor,
                  cache=[object_digest(layer) for layer in session.cache.layers])
    state = session.episodic
    if state is not None:
        digest['resident'] = {str(k): array_digest(v) for k, v in sorted(state.resident.items())}
        digest['running'] = {str(k): object_digest(v) for k, v in sorted(state.running.items())}
        digest['groups'] = [dict(step=g.step_id, start=g.start, end=g.end, visual=g.visual_start,
                                 scores={str(k): array_digest(v) for k, v in sorted(g.scores.items())})
                            for g in state.groups]
        scorers = {}
        for key, scorer in sorted(state.scorers.items()):
            scorers[str(key)] = object_digest(scorer)
            if getattr(scorer, 'legacy', None) is not None:
                scorers[f'{key}:legacy'] = object_digest(scorer.legacy)
        digest['scorers'] = scorers
    return digest


def first_difference(a, b, path='state'):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            found = first_difference(a.get(key, '<absent>'), b.get(key, '<absent>'), f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and not (a and isinstance(a[0], str)):
        for index, (x, y) in enumerate(zip(a, b)):
            found = first_difference(x, y, f'{path}[{index}]')
            if found:
                return found
        return None
    return None if a == b else path


def pick_row(manifest, uid, steps):
    for line in Path(manifest).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (uid and row['episode_uid'] == uid) or (not uid and row.get('dataset') == 'r2rce'
                                                   and len(row['steps']) >= steps):
            return row
    raise ValueError('No suitable manifest row')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model-path')
    p.add_argument('--manifest', required=True)
    p.add_argument('--episode-uid')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--out')
    args = p.parse_args()
    row = pick_row(args.manifest, args.episode_uid, args.steps)
    images = []
    for step in row['steps'][:args.steps]:
        with Image.open(step['rgb_path']) as image:
            images.append(image.convert('RGB'))
    model, serializer = load_checkpoint(args.checkpoint, args.model_path)
    uid, instruction = 'dagger-check', row['instruction']
    report = dict(episode_uid=row['episode_uid'], steps=len(images), checks={})

    reference = StreamSession(model, serializer)
    reference.reset(uid, instruction)
    actions, first_tokens = [], []
    for t, image in enumerate(images):
        result = reference.observe(uid, t, image)
        actions.append(result['class_id'])
        first_tokens.append(result['response_token_ids'][0])
    expected = state_digest(reference)
    report['observe_actions'] = [ACTIONS[a] for a in actions]

    decoded = DaggerSession(model, serializer)
    decoded.reset(uid, instruction)
    decoded_actions = [decoded.step(uid, t, image)['class_id'] for t, image in enumerate(images)]
    report['checks']['decode_actions'] = decoded_actions == actions
    report['checks']['decode_state'] = first_difference(state_digest(decoded), expected) or True

    # Collection always passes intercept_with under policy control: unless the policy's first token
    # starts STOP, nothing may change.  Decisions whose first token does start STOP are left plain.
    armed = DaggerSession(model, serializer)
    armed.reset(uid, instruction)
    stop_first = armed.layout.stop_first
    other = [ACTIONS.index('MOVE_FORWARD'), ACTIONS.index('TURN_LEFT')]
    armed_results = [armed.step(uid, t, image, intercept_with=None if first_tokens[t] == stop_first
                                else other[actions[t] == other[0]])
                     for t, image in enumerate(images)]
    report['checks']['armed_actions'] = [r['class_id'] for r in armed_results] == actions
    report['checks']['armed_sources'] = all(r['source'] == 'model' for r in armed_results)
    report['checks']['armed_state'] = first_difference(state_digest(armed), expected) or True

    forced = DaggerSession(model, serializer)
    forced.reset(uid, instruction)
    sources = [forced.step(uid, t, image, forced_class=a)['source'] for t, (image, a) in enumerate(zip(images, actions))]
    report['checks']['forced_sources'] = set(sources) == {'forced'}
    report['checks']['forced_state'] = first_difference(state_digest(forced), expected) or True

    k = len(images) // 2
    replacement = ACTIONS.index('MOVE_FORWARD') if actions[k] != ACTIONS.index('MOVE_FORWARD') else ACTIONS.index('TURN_LEFT')
    intercepted, twin = DaggerSession(model, serializer), DaggerSession(model, serializer)
    for session in (intercepted, twin):
        session.reset(uid, instruction)
        for t in range(k):
            session.step(uid, t, images[t], forced_class=actions[t])
    original = intercepted.layout
    intercepted.layout = copy.copy(original)
    intercepted.layout.stop_first = first_tokens[k]                 # the policy's real first token
    try:
        result = intercepted.step(uid, k, images[k], intercept_with=replacement)
    finally:
        intercepted.layout = original
    twin.step(uid, k, images[k], forced_class=replacement)
    report['checks']['intercept_source'] = (result['source'], result['class_id']) == ('stop_intercepted', replacement)
    report['checks']['intercept_state'] = first_difference(state_digest(intercepted), state_digest(twin)) or True
    tail = [(intercepted.step(uid, t, images[t])['class_id'], twin.step(uid, t, images[t])['class_id'])
            for t in range(k + 1, len(images))]
    report['checks']['intercept_continuation'] = all(a == b for a, b in tail)
    report['checks']['intercept_final_state'] = first_difference(state_digest(intercepted), state_digest(twin)) or True
    report['passed'] = all(value is True for value in report['checks'].values())
    report['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 2**30
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n')
    sys.exit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
