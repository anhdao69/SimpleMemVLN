"""Download and verify the SimpleMemVLN RL assets for this cluster.

1. R2R_VLNCE_v1-3_preprocessed (official Google Drive archive) -> DATA_ROOT/datasets/
   RxR-CE 15-degree label files (DVLe/RxR_CE_15_deg_Trajectory_data) -> DATA_ROOT/datasets/rxr_15deg/
2. MP3D scenes, pinned mirror revision (same revision as the RL baseline) -> DATA_ROOT/scene_datasets/mp3d/
3. S8N epoch-1 checkpoint (anhdao69/M5_S, folder S8N) -> CKPT_ROOT/M5_S/S8N, verified against SHA256SUMS.json
The 14 train scans missing from the mirror are fetched afterwards by the project's own
scripts/rl/download_train_scenes.py (byte-identity-checked fallback mirror).
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

SCENE_REPO, SCENE_REV = 'anhdao69/mp3d-scene-datasets', 'd8c7686323e93bdc935a51c71aed8a2d662ab261'
R2R_GDRIVE_ID = '1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr'      # official VLN-CE R2R_VLNCE_v1-3_preprocessed
R2R_VAL_UNSEEN_SHA256 = '1767a407e2c8a011fbb7abece76cd64c5b39ff9fa0e9e340ebdce5a490d167c3'
RXR_REPO = 'DVLe/RxR_CE_15_deg_Trajectory_data'
RXR_LABEL_FILES = ('train/conversion_manifest.json', 'train/train_guide.json.gz', 'train/train_guide_gt.json.gz',
                   'val_unseen/conversion_manifest.json', 'val_unseen/val_unseen_guide.json.gz',
                   'val_unseen/val_unseen_guide_gt.json.gz')
RXR_AUDITED = {   # hashes recorded in the project's RxR15 evaluation contracts
    'val_unseen/val_unseen_guide.json.gz': '0bf7358953a1b59502d7876611bf5da817d19ce7a8d9922b740a53a93812b94e',
    'val_unseen/val_unseen_guide_gt.json.gz': 'ac381254bfc019a55c62d993a770ffb7a0b77c02fa0c18118a99d02e6bc6b959',
}
CKPT_REPO = 'anhdao69/M5_S'
QWEN_REPO, QWEN_REV = 'Qwen/Qwen3.5-4B', '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'   # EVALUATION_SETUP.md 10.1


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def r2r(data_root, record):
    """Official VLN-CE R2R_VLNCE_v1-3_preprocessed archive (Google Drive, user-specified source)."""
    import gdown
    downloads = data_root / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    zpath = downloads / 'R2R_VLNCE_v1-3_preprocessed.zip'
    if not zpath.is_file():
        gdown.download(id=R2R_GDRIVE_ID, output=str(zpath), quiet=False)
    digest = sha256(zpath)
    target = data_root / 'datasets'
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        bad = [n for n in names if n.startswith('/') or '..' in Path(n).parts]
        assert not bad, bad
        z.extractall(target)
    val = target / 'R2R_VLNCE_v1-3_preprocessed' / 'val_unseen' / 'val_unseen.json.gz'
    val_digest = sha256(val)
    assert val_digest == R2R_VAL_UNSEEN_SHA256, ('val_unseen hash differs from the audited file', val_digest)
    record['r2r'] = dict(source=f'gdrive:{R2R_GDRIVE_ID}', zip_sha256=digest, members=len(names),
                         roots=sorted({n.split('/')[0] for n in names}), val_unseen_sha256=val_digest)
    print('R2R extracted', record['r2r'], flush=True)


def rxr15(data_root, record):
    """RxR-CE 15-degree label files only (no trajectory tarballs, no scene archive)."""
    api = HfApi()
    info = api.dataset_info(RXR_REPO, files_metadata=True)
    meta = {s.rfilename: s for s in info.siblings}
    target = data_root / 'datasets' / 'rxr_15deg'
    files = {}
    for name in RXR_LABEL_FILES:
        path = Path(hf_hub_download(RXR_REPO, name, repo_type='dataset', revision=info.sha, local_dir=target))
        digest = sha256(path)
        if meta[name].lfs is not None:
            assert digest == meta[name].lfs.sha256, name
        files[name] = digest
    for name, expected in RXR_AUDITED.items():
        assert files[name] == expected, ('RxR15 file differs from the audited evaluation file', name)
    record['rxr15'] = dict(repo=RXR_REPO, revision=info.sha, files=files)
    print('RxR15 labels verified', len(files), flush=True)


def scenes(data_root, record):
    root = data_root / 'scene_datasets' / 'mp3d'
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(SCENE_REPO, repo_type='dataset', revision=SCENE_REV, local_dir=root, max_workers=8)
    api = HfApi()
    info = api.dataset_info(SCENE_REPO, revision=SCENE_REV, files_metadata=True)
    verified = {}
    for s in info.siblings:
        if not s.rfilename.endswith(('.glb', '.navmesh')):
            continue
        path = root / s.rfilename
        assert path.stat().st_size == s.size, s.rfilename
        if s.lfs is not None:
            digest = sha256(path)
            assert digest == s.lfs.sha256, s.rfilename
            verified[s.rfilename] = digest
    record['scenes'] = dict(repo=SCENE_REPO, revision=SCENE_REV, verified_files=len(verified),
                            scans=sorted({n.split('/')[0] for n in verified}))
    print('scenes verified', len(verified), 'files', len(record['scenes']['scans']), 'scans', flush=True)


def checkpoint(ckpt_root, record, folder='S8N'):
    api = HfApi()
    info = api.model_info(CKPT_REPO)
    snapshot_download(CKPT_REPO, revision=info.sha, allow_patterns=[f'{folder}/*'],
                      local_dir=ckpt_root / 'M5_S', max_workers=4)
    repo_root = ckpt_root / 'M5_S'
    base = repo_root / folder
    sums = json.loads((base / 'SHA256SUMS.json').read_text())
    checked = {}
    for name, entry in sums.items():   # keys are repo-relative ("S8N/config.json")
        expected = entry['sha256']
        assert (repo_root / name).stat().st_size == entry['size'], name
        digest = sha256(repo_root / name)
        assert digest == expected, (name, digest, expected)
        checked[name] = digest
    record[f'checkpoint_{folder}'] = dict(repo=CKPT_REPO, revision=info.sha, path=str(base), verified=checked)
    print('checkpoint verified', folder, len(checked), 'files', flush=True)


def qwen_base(ckpt_root, record):
    """Pinned Qwen3.5-4B base (processor/config source; SFT weights replace its parameters)."""
    target = ckpt_root / 'qwen-base'
    snapshot_download(QWEN_REPO, revision=QWEN_REV, local_dir=target, max_workers=4,
                      ignore_patterns=['*.md', '*.gitattributes'])
    info = HfApi().model_info(QWEN_REPO, revision=QWEN_REV, files_metadata=True)
    verified = {}
    for s in info.siblings:
        path = target / s.rfilename
        if not path.is_file():
            continue
        assert path.stat().st_size == s.size, s.rfilename
        if s.lfs is not None:
            digest = sha256(path)
            assert digest == s.lfs.sha256, s.rfilename
            verified[s.rfilename] = digest
    record['qwen_base'] = dict(repo=QWEN_REPO, revision=QWEN_REV, path=str(target), verified_lfs=verified)
    print('qwen base verified', len(verified), 'LFS files', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--ckpt-root', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    p.add_argument('--parts', default='r2r,rxr15,s8n,scenes')
    args = p.parse_args()
    record = json.loads(args.record.read_text()) if args.record.is_file() else {}
    parts = args.parts.split(',')
    if 'r2r' in parts:
        r2r(args.data_root, record)
        args.record.write_text(json.dumps(record, indent=2) + '\n')
    if 'rxr15' in parts:
        rxr15(args.data_root, record)
        args.record.write_text(json.dumps(record, indent=2) + '\n')
    if 's8n' in parts:
        checkpoint(args.ckpt_root, record, 'S8N')
        args.record.write_text(json.dumps(record, indent=2) + '\n')
    if 'qwen' in parts:
        qwen_base(args.ckpt_root, record)
        args.record.write_text(json.dumps(record, indent=2) + '\n')
    if 'scenes' in parts:
        scenes(args.data_root, record)
        args.record.write_text(json.dumps(record, indent=2) + '\n')
    print('done', flush=True)


if __name__ == '__main__':
    main()
