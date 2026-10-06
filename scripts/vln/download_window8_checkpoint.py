"""Pin and verify the published Window8-B epoch-1 export."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

REPOSITORY = 'anhdao69/SimpleMemVLN-R2R-Window8-B'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def verify(directory, revision):
    directory = Path(directory).resolve()
    checksums = directory / 'SHA256SUMS.json'
    if not checksums.is_file():
        raise ValueError('Published checkpoint has no SHA256SUMS.json')
    manifest = json.loads(checksums.read_text())
    required = ('epoch-1/navigation.json', 'epoch-1/config.json',
                'epoch-1/tokenizer.json', 'epoch-1/processor_config.json')
    for name in required:
        if name not in manifest:
            raise ValueError(f'Published checksum missing {name}')
    if not any(name in manifest for name in
               ('epoch-1/pytorch_model.bin', 'epoch-1/model.safetensors')):
        raise ValueError('Published epoch-1 weights are missing')
    for name, expected in manifest.items():
        file = (directory / name).resolve()
        if not file.is_relative_to(directory) or not file.is_file():
            raise ValueError(f'Missing or unsafe published path: {name}')
        if digest(file) != expected:
            raise ValueError(f'Checkpoint checksum mismatch: {name}')
    navigation = json.loads((directory/'epoch-1/navigation.json').read_text())
    config = navigation['config']
    if config['memory']['mode'] != 'window8' or config['model']['output_mode'] != 'qwen_text':
        raise ValueError('This is not the trained Window8-B policy')
    if config['memory']['kv_window_steps_including_current'] != 8:
        raise ValueError('Window8 retention contract differs')
    verification = dict(repository=REPOSITORY, revision=revision,
                        sha256_verified=True, checkpoint=str(directory/'epoch-1'))
    (directory/'verified_revision.json').write_text(json.dumps(verification, indent=2)+'\n')
    return verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='artifacts/checkpoints/window8-b')
    args = parser.parse_args()
    revision = HfApi().model_info(REPOSITORY).sha
    root = snapshot_download(REPOSITORY, revision=revision,
                             local_dir=args.directory, max_workers=4)
    print(json.dumps(verify(root, revision), indent=2))


if __name__ == '__main__':
    main()
