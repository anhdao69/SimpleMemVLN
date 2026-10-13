"""Publish one attempt's epoch export and complete training state to the Hub.

Layout in the (private) model repo, one folder per run version::

    <run_id>/<attempt>/export-e1/            serving export (strict reload contract)
    <run_id>/<attempt>/checkpoint-<step>/    full DeepSpeed/Trainer state for exact
                                             same-profile resume of epoch 2
    <run_id>/<attempt>/campaign/             launch manifest, resolved config,
                                             qualification and training logs

Files are staged by hardlink (no copy) and uploaded with the resumable large
folder uploader; remote sizes are verified before ``upload_complete.json``.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import time

ONLINE_BLOCKERS = ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')


def stage(staging, folder, checkpoint, export, attempt_dir):
    target = staging / folder
    if target.exists():
        shutil.rmtree(target)
    files = {}

    def link(source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, destination)
        files[str(destination.relative_to(staging))] = source.stat().st_size
    for source in sorted(Path(checkpoint).rglob('*')):
        if source.is_file():
            link(source, target / Path(checkpoint).name / source.relative_to(checkpoint))
    for source in sorted(Path(export).rglob('*')):
        if source.is_file():
            link(source, target / 'export-e1' / source.relative_to(export))
    attempt_dir = Path(attempt_dir)
    train = attempt_dir / 'train'
    extras = [attempt_dir / n for n in ('launch_manifest.json', 'resolved_config.yaml', 'source_hashes.json',
                                        'preflight.json', 'events.jsonl')]
    extras += sorted(train.glob('*.jsonl')) + sorted(train.glob('*.json'))
    extras += sorted(attempt_dir.glob('qualification-*/*_suite.json'))
    for source in extras:
        if source.is_file():
            link(source, target / 'campaign' / source.name)
    readme = target / 'README.md'
    readme.write_text(
        f'# {folder}\n\nEpisodic-memory campaign attempt (plans/episodic_mem.md). `export-e1/` is the '
        f'epoch-1 serving export; `{Path(checkpoint).name}/` is the complete training state (same world '
        'size/profile only) for exact epoch-2 resume with `--resume_from_checkpoint`. Epoch 1 is a '
        'mid-schedule snapshot of a two-epoch (7,704-update) schedule. No navigation result is implied.\n')
    files[str(readme.relative_to(staging))] = readme.stat().st_size
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True)
    p.add_argument('--folder', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--export', required=True)
    p.add_argument('--attempt-dir', required=True)
    p.add_argument('--private', action='store_true', default=True)
    args = p.parse_args()
    # The job wrapper exports HF_HUB_OFFLINE=1 *and* TRANSFORMERS_OFFLINE=1; huggingface_hub
    # treats either as offline mode, so both must go before it is imported.
    for name in ONLINE_BLOCKERS:
        os.environ.pop(name, None)
    from huggingface_hub import HfApi
    api = HfApi()
    staging = Path(args.attempt_dir) / 'hf_staging'
    files = stage(staging, args.folder, args.checkpoint, args.export, args.attempt_dir)
    api.create_repo(args.repo, repo_type='model', private=args.private, exist_ok=True)
    started = time.time()
    api.upload_large_folder(repo_id=args.repo, repo_type='model', folder_path=str(staging),
                            allow_patterns=[f'{args.folder}/**'], num_workers=8, print_report=False)
    remote = {}
    for info in api.list_repo_tree(args.repo, path_in_repo=args.folder, recursive=True, repo_type='model'):
        size = getattr(info, 'size', None)
        if size is not None:
            remote[info.path] = size
    missing = {k: v for k, v in files.items() if remote.get(k) != v}
    report = dict(repo=args.repo, folder=args.folder, files=len(files), bytes=sum(files.values()),
                  seconds=time.time() - started, verified=not missing, mismatched=list(missing)[:20])
    (Path(args.attempt_dir) / 'upload_complete.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if not missing else 1)


if __name__ == '__main__':
    main()
