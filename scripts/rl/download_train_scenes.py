"""Download and SHA256-verify the MP3D scans used by the R2R-VLNCE train split.

Uses the same immutable scene mirror revision as the evaluation setup
(results/setup-record/scene-mirror-coverage.json); existing verified files
(the 11 val_unseen scans) are left untouched. That mirror lacks 14 train scans;
those come from a second pinned mirror that is accepted only because every
glb/navmesh the two mirrors share is byte-identical (checked here, by LFS SHA256).
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = 'anhdao69/mp3d-scene-datasets'
REVISION = 'd8c7686323e93bdc935a51c71aed8a2d662ab261'
FALLBACK_REPO = 'bingCS/mp3d_dataset'
FALLBACK_REVISION = '3fdee1c70e6010366ea8013cb0d57dc8ae0f3676'
FALLBACK_PREFIX = 'mp3d/'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-json', required=True)
    p.add_argument('--scenes-dir', required=True, help='Directory containing mp3d/')
    p.add_argument('--record', required=True)
    args = p.parse_args()
    with gzip.open(args.train_json, 'rt') as f:
        scans = sorted({Path(e['scene_id']).stem for e in json.load(f)['episodes']})
    api = HfApi()
    info = api.dataset_info(REPO, revision=REVISION, files_metadata=True)
    meta = {s.rfilename: s for s in info.siblings}
    fallback = api.dataset_info(FALLBACK_REPO, revision=FALLBACK_REVISION, files_metadata=True)
    fmeta = {s.rfilename[len(FALLBACK_PREFIX):]: s for s in fallback.siblings
             if s.rfilename.startswith(FALLBACK_PREFIX)}
    shared = [n for n in meta if n.endswith(('.glb', '.navmesh')) and n in fmeta]
    mismatched = [n for n in shared if (meta[n].size, meta[n].lfs and meta[n].lfs.sha256)
                  != (fmeta[n].size, fmeta[n].lfs and fmeta[n].lfs.sha256)]
    if not shared or mismatched:
        raise SystemExit(f'Fallback mirror is not byte-identical: {mismatched}')
    wanted = [f'{s}/{s}{ext}' for s in scans for ext in ('.glb', '.navmesh')]
    source = {w: (REPO, REVISION, w, meta[w]) if w in meta else
              (FALLBACK_REPO, FALLBACK_REVISION, FALLBACK_PREFIX + w, fmeta.get(w)) for w in wanted}
    missing = [w for w, (_, _, _, m) in source.items() if m is None]
    if missing:
        raise SystemExit(f'Both mirrors lack {missing}')
    root = Path(args.scenes_dir) / 'mp3d'
    verified = {}
    origin = {}
    for name in wanted:
        repo, revision, remote, m = source[name]
        target = root / name
        expected = m.lfs.sha256 if m.lfs is not None else None
        if not (target.is_file() and target.stat().st_size == m.size
                and (expected is None or sha256(target) == expected)):
            downloaded = Path(hf_hub_download(repo, remote, repo_type='dataset', revision=revision,
                                              local_dir=Path(args.scenes_dir) / '.fallback-download'
                                              if repo == FALLBACK_REPO else root))
            if repo == FALLBACK_REPO:
                target.parent.mkdir(parents=True, exist_ok=True)
                downloaded.replace(target)
        assert target.stat().st_size == m.size, name
        origin[name] = f'{repo}@{revision}'
        digest = sha256(target)
        if expected is not None:
            assert digest == expected, name
        verified[name] = digest
        print('verified', name, flush=True)
    Path(args.record).write_text(json.dumps(dict(
        repository=REPO, revision=REVISION, fallback_repository=FALLBACK_REPO,
        fallback_revision=FALLBACK_REVISION, shared_files_byte_identical=len(shared),
        train_scans=scans, origin=origin, verified_sha256=verified), indent=2) + '\n')
    print('Train scenes verified:', len(scans), 'scans,', len(verified), 'files', flush=True)


if __name__ == '__main__':
    main()
