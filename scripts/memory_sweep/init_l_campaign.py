"""Create the Phase-B L campaign record inside the label root (never reuses or overwrites one).

The source must already be a clean detached worktree at the pinned commit.
The label root holds the frozen selection and the collected labels; the arm
memories are written later by lane B's ``arm-configs`` gate from the fitted
artifacts.  Original data/checkpoints are only referenced.
"""
import argparse
import json
from pathlib import Path
import subprocess
import yaml
from scripts.memory_sweep.init_campaign import BASE, M5, RUNTIME

S_ROOT = BASE / 'm5-gdn-s-sweep-dabf4ad-20261020'
SCREENING = ['S0', 'L0', 'L1', 'L2', 'L3']


def lanes(label_step):
    return {
        'B': dict(job='4925', host_guard_gib=691.2, qualifies=True, predecessor=dict(step=label_step),
                  arms=['L3', 'L1']),
        'A': dict(job='5248', host_guard_gib=432.0, qualifies=False,
                  predecessor=dict(state=str(S_ROOT / 'lanes/A/state.json'), step='5248.27',
                                   final_status='lane-complete'),
                  arms=['L2', 'L0']),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--label-step', required=True, help='Slurm step collecting the full labels')
    args = p.parse_args()
    root, source = Path(args.root), Path(args.source).resolve()
    if (root / 'campaign.json').exists() or (root / 'configs').exists():
        raise SystemExit(f'Refusing to reuse campaign root {root}')
    if not (root / 'labels/selection.json').exists():
        raise SystemExit('The label root must already hold the frozen selection')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('Source worktree must be clean')
    (root / 'configs').mkdir(parents=True)
    (root / 'configs/runtime.yaml').write_text(
        '# M5_E1 runtime (reports/vggt_five/M5_E1/operations/runtime.yaml) with loader 4 workers/prefetch 1,\n'
        '# identical to the S campaign. Engineering only, never the objective.\n'
        + yaml.safe_dump(RUNTIME, sort_keys=False))
    campaign = dict(
        version='m5_gdn_l_sweep_campaign_v1', root=str(root), source=str(source), source_commit=commit,
        scientific_baseline='fde1cbef213e27320920400a826addea8d5e114a', base=str(BASE),
        manifest=str(BASE / 'SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl'),
        manifest_sha256='5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933',
        model=str(BASE / 'cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/'
                         '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'),
        m5_e1=str(M5 / 'publication-m5-epoch1-20261019/epoch-1'),
        m5_e1_weights_sha256='92089fa16096f5fb581a7fb85a670b539a887278d7f66bacd20b4d0424d70ea6',
        m5_e1_published_sums=str(M5 / 'publication-m5-epoch1-20261019/SHA256SUMS.json'),
        m5_e1_reference_config=str(M5 / 'production/m5_joint_e1/train/resolved_config.json'),
        fixtures=str(BASE / 'g1-omega-20261013/fixtures'), seed=429, min_free_disk_gib=220,
        recovery_arm='L3', screening_arms=SCREENING, lanes=lanes(args.label_step),
        s_campaign=str(S_ROOT / 'campaign.json'))
    (root / 'campaign.json').write_text(json.dumps(campaign, indent=2) + '\n')
    print(root / 'campaign.json')


if __name__ == '__main__':
    main()
