"""Audit ActiveVLN's 4K R2R RL episode list against R2R-VLNCE v1-3 (G1-R2R).

Maps every parquet ``extra_info.episode_id`` to the local ``train`` split,
records scene, instruction, start->goal geodesic distance and our 15-degree
expert length N* (train_gt actions, including STOP), expands ActiveVLN's merged
GT strings into 0.25 m / 15-degree primitives as a cross-check, and verifies
train/validation separation (no shared episode identity, trajectory or scan).
Requires pyarrow (run with the isolated tools interpreter).
"""
import argparse
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

MERGED = re.compile(r'^(move forward (25|50|75)cm|turn (left|right) (15|30|45) degrees|stop)$')


def primitives(actions):
    out = []
    for text in actions:
        m = MERGED.match(text.strip())
        if not m:
            raise ValueError(f'Unexpected ActiveVLN action string {text!r}')
        if text.startswith('move'):
            out += [1] * (int(m.group(2)) // 25)
        elif text.startswith('turn'):
            out += [2 if m.group(3) == 'left' else 3] * (int(m.group(4)) // 15)
        else:
            out.append(0)
    return out


def load(path):
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', required=True)
    p.add_argument('--r2r-root', required=True, help='R2R_VLNCE_v1-3_preprocessed directory')
    p.add_argument('--out', required=True)
    p.add_argument('--report', required=True)
    args = p.parse_args()
    root = Path(args.r2r_root)
    rows = pq.read_table(args.parquet).to_pylist()
    ids = [int(r['extra_info']['episode_id']) for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit('Duplicate episode IDs in the RL list')
    if {r['extra_info']['split'] for r in rows} != {'train'} or {r['data_source'] for r in rows} != {'r2r'}:
        raise SystemExit('RL list is not R2R train')
    train = {e['episode_id']: e for e in load(root / 'train/train.json.gz')['episodes']}
    gt = load(root / 'train/train_gt.json.gz')
    missing = sorted(set(ids) - train.keys())
    val = {}
    for split in ('val_seen', 'val_unseen'):
        episodes = load(root / f'{split}/{split}.json.gz')['episodes']
        val[split] = dict(trajectories={e['trajectory_id'] for e in episodes},
                          scans={Path(e['scene_id']).stem for e in episodes},
                          instructions={e['instruction']['instruction_text'].strip() for e in episodes})
    records, ratio, exact = [], [], 0
    for row, eid in zip(rows, ids):
        if eid not in train:
            continue
        ep = train[eid]
        ours = gt[str(eid)]['actions']
        theirs = primitives(row['extra_info']['gt_actions'])
        exact += ours == theirs
        ratio.append(len(theirs) / len(ours))
        records.append(dict(episode_id=str(eid), scene_id=ep['scene_id'], trajectory_id=ep['trajectory_id'],
                            instruction=ep['instruction']['instruction_text'].strip(),
                            geodesic_distance=ep['info']['geodesic_distance'], N_star=len(ours),
                            activevln_merged_len=len(row['extra_info']['gt_actions']),
                            activevln_primitive_len=len(theirs)))
    scans = {Path(r['scene_id']).stem for r in records}
    trajectories = {r['trajectory_id'] for r in records}
    texts = {r['instruction'] for r in records}
    separation = {split: dict(shared_scans=sorted(scans & v['scans']),
                              shared_trajectory_ids=len(trajectories & v['trajectories']),
                              shared_instruction_texts=len(texts & v['instructions']))
                  for split, v in val.items()}
    body = ''.join(json.dumps(r, sort_keys=True) + '\n' for r in records)
    Path(args.out).write_text(body)
    lengths = sorted(r['N_star'] for r in records)
    ratio.sort()
    report = dict(parquet=str(args.parquet),
                  parquet_sha256=hashlib.sha256(Path(args.parquet).read_bytes()).hexdigest(),
                  rows=len(rows), unique_ids=len(set(ids)), mapped=len(records), missing=missing,
                  pool_sha256=hashlib.sha256(body.encode()).hexdigest(),
                  sorted_ids_sha256=hashlib.sha256(','.join(sorted(r['episode_id'] for r in records)).encode()).hexdigest(),
                  scans=len(scans), N_star=dict(min=lengths[0], median=lengths[len(lengths) // 2],
                                                mean=sum(lengths) / len(lengths), max=lengths[-1]),
                  activevln_vs_ours_exact_primitive_match=exact,
                  activevln_over_ours_length_ratio=dict(p5=ratio[len(ratio) // 20], p50=ratio[len(ratio) // 2],
                                                        p95=ratio[len(ratio) * 19 // 20]),
                  activevln_magnitudes=dict(Counter(a for r in rows for a in r['extra_info']['gt_actions'])),
                  separation=separation)
    Path(args.report).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'activevln_magnitudes'}, indent=2))


if __name__ == '__main__':
    main()
