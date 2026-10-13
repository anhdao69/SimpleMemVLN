"""Sequential acceptance after timing; run once inside the four-GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    assert json.loads((root / 'experiments/online-h4-best-joint/result.json').read_text())['passed']
    options = json.loads((root / 'tuning_selection.json').read_text())['best']['options']
    train = root / 'experiments/online-h4-recovery-reference/train'
    resumed = root / 'experiments/online-h4-recovery-resumed/train'
    tasks = [
        ('mixed-extreme', ['scripts/vln/profile_vggt_g1.py', '--root', str(root),
            '--name', 'online-h4-best-mixed-extreme', '--fixture', str(root / 'fixtures/mixed_extreme.jsonl'),
            '--updates', '3', *options]),
        ('recovery-campaign', ['scripts/vln/benchmark_vggt_g1.py', '--root', str(root), '--phase', 'recovery']),
        ('recovery-compare', ['scripts/vln/check_vggt_recovery.py', '--reference', str(train),
            '--resumed', str(resumed), '--out', str(root / 'recovery-comparison.json')]),
        ('export-replay', ['scripts/vln/check_vggt_export.py', '--source', str(resumed / 'final'),
            '--output', str(root / 'serving-export'), '--manifest',
            '/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl',
            '--model-path', os.environ['VLN_MODEL_PATH']]),
        ('full-suite-final', ['-m', 'pytest', '-q', 'tests/vln']),
    ]
    records = []
    for name, command in tasks:
        print('START', name, flush=True)
        with (root / f'{name}.log').open('x') as log:
            status = subprocess.run([sys.executable, *command], stdout=log, stderr=subprocess.STDOUT).returncode
        records.append(dict(name=name, returncode=status, command=command))
        (root / 'acceptance-results.json').write_text(json.dumps(records, indent=2) + '\n')
        print('DONE', name, status, flush=True)
        if status:
            raise SystemExit(status)
        if name == 'mixed-extreme':
            result = json.loads((root / 'experiments/online-h4-best-mixed-extreme/result.json').read_text())
            assert result['passed'] and result['check']['peak_allocated_gib'] <= 72
    print('G1_ACCEPTANCE_PASS', flush=True)


if __name__ == '__main__':
    main()
