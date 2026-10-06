"""Fetch immutable R2R FullContext-B snapshots and verify publisher checksums."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download
from qwen_vl.eval.habitat_r2r import sha256, atomic_json

REPO = 'anhdao69/SimpleMemVLN-R2R-FullContext-B'
REVISION = '73d866c31e398596a67cd8be6f7dbfd25ab9241d'
BASE_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', default='artifacts/checkpoints')
    args = p.parse_args()
    root = Path(args.directory).resolve()
    checkpoint = Path(snapshot_download(REPO, revision=REVISION,
                        local_dir=root / 'fullcontext-b', max_workers=4))
    hashes = json.loads((checkpoint / 'SHA256SUMS.json').read_text())
    for epoch in (1, 2, 3):
        for required in ('pytorch_model.bin', 'navigation.json', 'config.json', 'tokenizer.json', 'processor_config.json'):
            if f'epoch-{epoch}/{required}' not in hashes:
                raise ValueError(f'Published checksums missing epoch-{epoch}/{required}')
    for name, expected in hashes.items():
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint.resolve()):
            raise ValueError(f'Unsafe checksum path: {name}')
        if sha256(path) != expected:
            raise ValueError(f'Checksum mismatch: {name}')
    atomic_json(checkpoint / 'verified_revision.json',
                dict(repo=REPO, revision=REVISION, sha256_verified=True))
    base = snapshot_download('Qwen/Qwen3.5-4B', revision=BASE_REVISION,
                             local_dir=root / 'qwen-base', max_workers=4,
                             ignore_patterns=['*.md', '*.gitattributes'])
    print(json.dumps(dict(checkpoint=str(checkpoint), base=base, verified=True)), flush=True)


if __name__ == '__main__':
    main()
