"""Task 0: pin source, native stack, checkpoint bytes and data identities on this host.

Paths in earlier reports are not evidence that assets are mounted here: every
identity is re-hashed.  The original dataset, images, instructions, actions
and checkpoints are only read.  Any expectation mismatch fails the gate; it is
recorded rather than worked around (no substitute checkpoint, no re-render).
"""
import argparse
import collections
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--image-sample', type=int, default=4000)
    args = p.parse_args()
    c = json.loads(Path(args.campaign).read_text())
    source = Path(c['source'])
    checks, report = {}, {}
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    report['source'] = dict(commit=head, clean=not dirty, scientific_baseline=c['scientific_baseline'],
                            baseline_is_ancestor=subprocess.run(
                                ['git', '-C', str(source), 'merge-base', '--is-ancestor', c['scientific_baseline'], head]
                            ).returncode == 0)
    checks['source_commit'] = head == c['source_commit'] and not dirty
    checks['baseline_ancestor'] = report['source']['baseline_is_ancestor']
    import torch, transformers, fla, flash_attn, deepspeed
    from qwen_vl.models.episodic_forward import NATIVE_FORWARD_SHA256, native_forward_fingerprint
    fingerprint = native_forward_fingerprint()
    report['environment'] = dict(python=sys.version, torch=torch.__version__, transformers=transformers.__version__,
                                 fla=getattr(fla, '__version__', None), flash_attn=flash_attn.__version__,
                                 deepspeed=deepspeed.__version__, cuda=torch.version.cuda,
                                 native_forward_sha256=fingerprint)
    checks['native_forward'] = fingerprint == NATIVE_FORWARD_SHA256
    checks['transformers_pinned'] = transformers.__version__ == '5.11.0'
    model = Path(c['model'])
    report['base_model'] = {str(f.relative_to(model)): sha(f) for f in sorted(model.rglob('*')) if f.is_file()}
    checks['base_model_revision'] = model.name == '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
    m5 = Path(c['m5_e1'])
    m5_files = {f.name: sha(f) for f in sorted(m5.iterdir()) if f.is_file()}
    published = json.loads(Path(c['m5_e1_published_sums']).read_text())
    expected = {Path(k).name: v['sha256'] for k, v in published.items() if k.startswith('M5_E1/epoch-1/')}
    report['m5_e1'] = dict(path=str(m5), files=m5_files, published_sums=c['m5_e1_published_sums'],
                           mismatched_published_files=sorted(k for k, v in expected.items() if m5_files.get(k) != v))
    checks['m5_e1_weights'] = m5_files.get('pytorch_model.bin') == c['m5_e1_weights_sha256']
    checks['m5_e1_matches_publication'] = not report['m5_e1']['mismatched_published_files']
    navigation = json.loads((m5 / 'navigation.json').read_text())
    reference = json.loads(Path(c['m5_e1_reference_config']).read_text())
    checks['m5_e1_is_plain_m5'] = (navigation['config'] == reference and 'research' not in reference
                                   and reference['memory']['episodic_score'] == 'committed_mean_l2_v1'
                                   and reference['training']['seed'] == 429)
    manifest = Path(c['manifest'])
    report['manifest'] = dict(path=str(manifest), sha256=sha(manifest))
    checks['manifest'] = report['manifest']['sha256'] == c['manifest_sha256']
    counts, physical, uids, images = collections.Counter(), collections.Counter(), set(), []
    with manifest.open() as f:
        for line in f:
            e = json.loads(line)
            counts[(e['dataset'], e['split'])] += 1
            physical[(e['dataset'], e['scene_id'], e['episode_id'])] += 1
            uids.add(e['episode_uid'])
            images += [s['rgb_path'] for s in e['steps']]
    report['manifest'].update(
        episodes={f'{d}/{s}': n for (d, s), n in sorted(counts.items())}, unique_uids=len(uids),
        physical_trajectories=len(physical), multi_instruction_trajectories=sum(n > 1 for n in physical.values()),
        frames=len(images))
    checks['manifest_counts'] = dict(counts) == {('r2rce', 'train'): 10819, ('rxrce', 'train'): 19996} and len(uids) == 30815
    rng = random.Random(429)
    sample = rng.sample(images, min(args.image_sample, len(images)))
    missing = [path for path in sample if not Path(path).is_file()]
    report['manifest']['image_sample'] = dict(sampled=len(sample), missing=missing[:20], missing_count=len(missing))
    checks['images_present_sample'] = not missing
    fixtures = Path(c['fixtures'])
    report['fixtures'] = {f.name: sha(f) for f in sorted(fixtures.glob('*.jsonl'))}
    fixture_images = [s['rgb_path'] for name in ('mixed_tail.jsonl', 'rxrce_longest.jsonl')
                      for line in (fixtures / name).read_text().splitlines() if line.strip()
                      for s in json.loads(line)['steps']]
    checks['fixture_images_present'] = all(Path(path).is_file() for path in fixture_images)
    report['random_control_identity'] = (
        'C_RANDOM uses legacy random_hash_v1 keyed by the manifest episode_uid (relative scene id); '
        'serving reproductions must preserve the evaluator UID/scene-path convention.')
    report['checks'] = checks
    report['passed'] = all(checks.values())
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(checks, indent=2), flush=True)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
