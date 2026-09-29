"""Audit available English RxR guide frames and combine with audited R2R."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from qwen_vl.contracts import validate_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--r2r-manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--rxr-variant', choices=['original', '15deg'], default='original')
    args = p.parse_args()
    root, out = Path(args.data_root), Path(args.out)
    if out.exists():
        raise ValueError('Refusing to overwrite an existing manifest')
    source = root / ('RxR_CE_15_deg_Trajectory_data/train/train_guide.json.gz'
                     if args.rxr_variant == '15deg' else 'datasets/rxr/train/train_guide.json.gz')
    version = hashlib.sha256(source.read_bytes()).hexdigest()
    with gzip.open(source, 'rt') as f:
        records = json.load(f)['episodes']
    lookup = {str(r['episode_id']): r for r in records}
    if len(lookup) != len(records):
        raise ValueError('Duplicate source episode IDs')
    frames = root / ('trajectory_data/RxR_15deg/train'
                     if args.rxr_variant == '15deg' else 'trajectory_data/RxR/train')
    folders = sorted(x for x in frames.iterdir() if x.is_dir())
    rows = [json.loads(s) for s in Path(args.r2r_manifest).read_text().splitlines() if s.strip()]
    for row in rows:
        validate_episode(row)
    counts, failures = Counter(), []
    available = {x.name for x in folders}
    for number, folder in enumerate(folders):
        record = lookup[folder.name]
        instruction = record['instruction']
        if not instruction['language'].startswith('en-'):
            raise ValueError('Unexpected non-English frame dataset')
        steps = []
        for image in os.scandir(folder):
            if not image.name.endswith('.png'):
                continue
            if not image.is_file():
                raise ValueError(f'Invalid image entry {image.path}')
            match = re.fullmatch(r'step_(\d+)_(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT|STOP)\.png', image.name)
            if not match:
                raise ValueError(f'Invalid frame name {image}')
            steps.append(dict(step_id=int(match[1]), rgb_path=image.path, action_name=match[2], is_valid=True))
        steps.sort(key=lambda s: s['step_id'])
        eid, iid = folder.name, str(instruction['instruction_id'])
        row = dict(schema_version=1, episode_uid=f"rxrce:train:{record['scene_id']}:{eid}:{iid}",
            dataset='rxrce', dataset_version=version, split='train', scene_id=record['scene_id'],
            episode_id=eid, instruction_id=iid, instruction=instruction['instruction_text'].strip(),
            language=instruction['language'], trajectory_variant=args.rxr_variant,
            observation_action_alignment='observation_before_action',
            alignment_evidence='raw filename labels; collector/replay gate reported separately', steps=steps)
        try:
            validate_episode(row, check_images=False)  # DirEntry.is_file checked above.
        except (ValueError, FileNotFoundError) as exc:
            failures.append(dict(episode_uid=row['episode_uid'], reason=str(exc)))
            continue
        rows.append(row)
        counts.update(s['action_name'] for s in steps)
        if (number + 1) % 2000 == 0:
            print(f'Audited {number + 1}/{len(folders)} RxR folders', flush=True)
    if len({r['episode_uid'] for r in rows}) != len(rows):
        raise ValueError('Duplicate episode UIDs')
    report = dict(episodes=len(rows), datasets=dict(Counter(r['dataset'] for r in rows)),
        rxr_action_counts=dict(counts), rejected=failures, trajectory_variant=args.rxr_variant,
        max_steps=max(len(r['steps']) for r in rows),
        missing_english_episode_ids=sorted(str(r['episode_id']) for r in records
            if r['instruction']['language'].startswith('en-') and str(r['episode_id']) not in available),
        actions=sum(len(r['steps']) for r in rows), capture_before_action_replay_verified=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.audit.json').write_text(json.dumps(report, indent=2)+'\n')
    if failures:
        raise ValueError(f'{len(failures)} invalid episodes; no manifest published')
    with out.open('x') as f:
        for row in rows:
            f.write(json.dumps(row)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rejected','missing_english_episode_ids')}), flush=True)


if __name__ == '__main__':
    main()
