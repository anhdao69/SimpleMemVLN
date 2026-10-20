"""Write the GDN-sweep arm memory configs from the registry (never hand-edited).

Without ``--calibration`` it writes the arms that need no calibration
artifact into ``configs/memory_sweep``.  With a calibration record it writes
S6/S8-N/S8-W with the embedded, hash-verified frozen artifacts of the chosen
checkpoint role (``base`` for matched training from the original Qwen,
``m5_e1`` for frozen-checkpoint screening).
"""
import argparse
import hashlib
import json
from pathlib import Path
import yaml
from qwen_vl.research.gdn_sweep_contract import (
    ARM_REGISTRY, DRIFT_ARMS, SLOW_HEAD_ARMS, arm_memory_config, canonical_sha256)

ROOT = Path(__file__).resolve().parents[2]
CALIBRATED = SLOW_HEAD_ARMS + DRIFT_ARMS


def artifacts_for(score, calibration, role):
    record = calibration['checkpoints'][role]
    digest = hashlib.sha256(json.dumps(calibration, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if score in SLOW_HEAD_ARMS:
        mask = record['slow_head_mask']
        assert canonical_sha256(mask) == record['slow_head_mask_sha256']
        return dict(slow_head_mask=mask, slow_head_mask_sha256=record['slow_head_mask_sha256'],
                    calibration_sha256=digest)
    floors = record['drift_denominator_floor']
    assert canonical_sha256(floors) == record['drift_denominator_floor_sha256']
    return dict(drift_denominator_floor=floors,
                drift_denominator_floor_sha256=record['drift_denominator_floor_sha256'],
                calibration_sha256=digest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default=str(ROOT / 'configs/memory_sweep'))
    p.add_argument('--calibration', help='calibration.json from scripts.memory_sweep.calibrate')
    p.add_argument('--role', choices=('base', 'm5_e1'), default='base')
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.calibration:
        calibration = json.loads(Path(args.calibration).read_text())
        scores = CALIBRATED
    else:
        calibration, scores = None, [s for s in ARM_REGISTRY if s not in CALIBRATED]
    for score in scores:
        artifacts = artifacts_for(score, calibration, args.role) if calibration else None
        memory = arm_memory_config(score, artifacts=artifacts)
        path = out / f'arm_{score}.yaml'
        path.write_text(yaml.safe_dump(memory, sort_keys=False))
        print(path)


if __name__ == '__main__':
    main()
