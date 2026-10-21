"""G-2: offline replay using independent frozen-source and diagnostic processes.

No streaming, parameter changes, fixture edits, or source monkeypatching. The
reference is the checkpoint's own frozen source; atol=.75, rtol=.03, RMS<=.15.
Bank overlap is descriptive (the specification supplies no bank pass threshold).
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def compare_logits(candidate, reference):
    if candidate.shape != reference.shape or candidate.ndim != 2 or not candidate.size:
        raise ValueError('Logits must have identical nonempty [decision, vocabulary] shapes')
    squared, maximum, outside, flips = 0., 0., 0, 0
    for start in range(0, len(reference), 16):
        a = np.asarray(candidate[start:start+16], dtype=np.float64)
        b = np.asarray(reference[start:start+16], dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite logits')
        delta = np.abs(a-b)
        squared += float(np.square(delta).sum())
        maximum = max(maximum, float(delta.max()))
        outside += int((delta > .75 + .03*np.abs(b)).sum())
        flips += int((a.argmax(-1) != b.argmax(-1)).sum())
    rms = float(np.sqrt(squared/reference.size))
    return dict(passed=outside == 0 and rms <= .15, atol=.75, rtol=.03,
                rms_limit=.15, rms=rms, max_abs=maximum, argmax_flips=flips,
                decisions=len(reference), elements=int(reference.size),
                shape=list(reference.shape), outside_tolerance=outside)


def compare_plans(candidate, reference):
    if candidate.keys() != reference.keys():
        raise ValueError('Bank plan layer sets differ')
    result, exact = {}, True
    for layer, expected in reference.items():
        actual = candidate[layer]
        if len(actual) != len(expected):
            raise ValueError('Bank plan step counts differ')
        rows = []
        empty = 0
        for step, (a, b) in enumerate(zip(actual, expected)):
            if len(a) != len(set(a)) or len(b) != len(set(b)) or len(a) != len(b):
                raise ValueError('Invalid or unequal bank cardinality')
            exact = exact and a == b
            if not b:
                empty += 1
                continue
            shared = len(set(a) & set(b))
            rows.append(dict(step=step, selected=len(b), intersection=shared,
                             selected_overlap=shared/len(b), jaccard=shared/len(set(a) | set(b))))
        result[layer] = dict(per_step=rows, nonempty_steps=len(rows), empty_steps=empty,
                             mean_selected_overlap=float(np.mean([r['selected_overlap'] for r in rows])) if rows else None,
                             min_selected_overlap=min((r['selected_overlap'] for r in rows), default=None))
    return dict(applicable=bool(reference), exact=exact, per_layer=result)


def relocate_fixture(episode, rgb_dir=None):
    relocated = copy.deepcopy(episode)
    evidence = []
    for step in relocated['steps']:
        original = step['rgb_path']
        resolved = (Path(rgb_dir)/Path(original).name if rgb_dir else Path(original)).resolve(strict=True)
        evidence.append(dict(original=original, resolved=str(resolved), sha256=sha256(resolved), bytes=resolved.stat().st_size))
        step['rgb_path'] = str(resolved)
    return relocated, evidence


def worker(args):
    # Import only after the independent process has its isolated PYTHONPATH.
    import torch
    import qwen_vl.train.vln_runtime as runtime
    source = Path(args.source).resolve()
    if not Path(runtime.__file__).resolve().is_relative_to(source/'src'):
        raise RuntimeError(f'Wrong source imported: {runtime.__file__}')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    episode, mappings = relocate_fixture(json.loads(Path(args.fixture).read_text()), args.rgb_dir)
    if len(episode['steps']) != 128:
        raise ValueError('G-2 requires the original 128-step fixture')
    torch.manual_seed(20261019)
    model, serializer = runtime.load_checkpoint(args.checkpoint, args.model_path or args.checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    data = {k: v.cuda() if torch.is_tensor(v) else v for k,v in serializer.encode_episode(episode).items()}
    diagnostics = {}
    with torch.inference_mode():
        kwargs = {'episodic_diagnostics': diagnostics} if model.episodic_spec is not None else {}
        logits = model(**data, **kwargs)['logits'].float().cpu().numpy()
    if logits.ndim != 2 or not logits.size or not np.isfinite(logits).all():
        raise ValueError('Invalid offline logits')
    np.save(output/'logits.npy', logits, allow_pickle=False)
    plans = {str(layer): [plan.group_ids(t).tolist() for t in range(128)]
             for layer, plan in diagnostics.get('plans', {}).items()}
    if model.episodic_spec is not None and not plans:
        raise RuntimeError('Episodic checkpoint produced no bank plans')
    write_json(output/'plans.json', plans)
    encoded = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            raw = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
            encoded[key] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw).hexdigest())
    write_json(output/'metadata.json', dict(source=str(source), loader=str(Path(runtime.__file__).resolve()),
        fixture=str(Path(args.fixture).resolve()), fixture_sha256=sha256(args.fixture),
        episode_uid=episode['episode_uid'], scene_id=episode.get('scene_id'),
        actions=[s['action_name'] for s in episode['steps']], mappings=mappings,
        encoded_tensors=encoded, torch_version=torch.__version__,
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        inference_mode=True, model_training=model.training, logits_sha256=sha256(output/'logits.npy')))


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve()
    weight = checkpoint/'pytorch_model.bin'
    if not weight.exists(): weight = checkpoint/'model.safetensors'
    digest = sha256(weight)
    if digest != args.weights_sha256:
        raise ValueError(f'Checkpoint SHA256 mismatch: {digest}')
    records = {}
    for label, source in [('frozen', args.frozen_source), ('diagnostic', args.diagnostic_source)]:
        source = Path(source).resolve()
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source/'src'), str(source)]),
                   PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        # Keep scripts/diag off sys.path: its statistics module must not shadow
        # Python's stdlib statistics during torch/inductor imports. The frozen
        # source root and src are already supplied explicitly via PYTHONPATH.
        command = [sys.executable, '-P', str(Path(__file__).resolve()), 'worker', '--source', str(source),
                   '--checkpoint', str(checkpoint), '--fixture', str(Path(args.fixture).resolve()),
                   '--output', str(output/label)]
        if args.rgb_dir: command += ['--rgb-dir', str(Path(args.rgb_dir).resolve())]
        if args.model_path: command += ['--model-path', str(Path(args.model_path).resolve())]
        records[label] = dict(command=command, pythonpath=env['PYTHONPATH'])
        write_json(output/'invocations.json', records)
        with (output/f'{label}.log').open('w') as log:
            subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    a, b = [json.loads((output/x/'metadata.json').read_text()) for x in ('diagnostic', 'frozen')]
    for key in ('fixture_sha256', 'episode_uid', 'scene_id', 'actions', 'mappings', 'encoded_tensors'):
        if a[key] != b[key]: raise ValueError(f'Input identity mismatch: {key}')
    result = compare_logits(np.load(output/'diagnostic/logits.npy', mmap_mode='r'),
                            np.load(output/'frozen/logits.npy', mmap_mode='r'))
    result['bank_selection'] = compare_plans(json.loads((output/'diagnostic/plans.json').read_text()),
                                              json.loads((output/'frozen/plans.json').read_text()))
    result.update(gate='G-2', model=args.model, weights_sha256=digest, checkpoint=str(checkpoint),
                  reference='frozen', fixture_sha256=a['fixture_sha256'], episode_uid=a['episode_uid'])
    write_json(output/'result.json', result)
    print(json.dumps(result, allow_nan=False))
    if not result['passed']: raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('run', 'worker'):
        command = sub.add_parser(name)
        for arg in ('checkpoint', 'fixture', 'output'): command.add_argument('--'+arg, required=True)
        command.add_argument('--rgb-dir')
        command.add_argument('--model-path', help='Verified local processor/base directory; defaults to serving checkpoint')
        if name == 'worker': command.add_argument('--source', required=True)
        else:
            for arg in ('model', 'weights-sha256', 'frozen-source', 'diagnostic-source'):
                command.add_argument('--'+arg, required=True)
    args = parser.parse_args()
    (worker if args.mode == 'worker' else run)(args)

if __name__ == '__main__': main()
