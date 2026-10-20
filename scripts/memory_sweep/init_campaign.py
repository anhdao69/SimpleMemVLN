"""Create a fresh GDN S-sweep campaign root (never reuses or overwrites one).

The source must already be a clean detached worktree at the pinned commit.
Non-calibrated arm memories are copied from ``configs/memory_sweep``; S6/S8
memories are written by the lane-A calibration gate.  All derived artifacts
live under the campaign root; original data/checkpoints are only referenced.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import yaml

BASE = Path('/mnt/data/vmo-ai-task/anhdh35')
M5 = BASE / 'm5-one-epoch-worker2-20261018'
RUNTIME = {
    'runtime': dict(episodic_diagnostics=False, malloc_trim_each_update=False, stage_pixels_bf16=True,
                    vision_microbatch_images=8, window_attention_batch_steps=32),
    'training': dict(activation_offload_backend='pinned_arena_v1', cpu_optimizer_impl='deepspeed_cpu_adam',
                     dataloader_num_workers=4, dataloader_persistent_workers=True, dataloader_pin_memory=False,
                     dataloader_prefetch_factor=1, gradient_accumulation_steps=2, optimizer_group_chunks=1),
}
LANES = {
    'A': dict(job='5248', host_guard_gib=432.0, qualifies=True, predecessor=None,
              arms=['S1', 'S2', 'S9', 'S4', 'S7']),
    'B': dict(job='4925', host_guard_gib=691.2, qualifies=False,
              predecessor=dict(state=str(BASE / 'dino-then-m5-seed431-worker2-20261020/queue_state.json'),
                               step='4925.147'),
              arms=['S3', 'S5', 'S6', 'S8N', 'S8W']),
}
SCREENING = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8N', 'S8W', 'S9', 'C_RECENCY', 'C_RANDOM']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--source', required=True)
    args = p.parse_args()
    root, source = Path(args.root), Path(args.source).resolve()
    if root.exists():
        raise SystemExit(f'Refusing to reuse campaign root {root}')
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise SystemExit('Source worktree must be clean')
    (root / 'configs/screening').mkdir(parents=True)
    for path in sorted((source / 'configs/memory_sweep').glob('arm_*.yaml')):
        shutil.copy2(path, root / 'configs' / path.name)
        shutil.copy2(path, root / 'configs/screening' / path.name)
    (root / 'configs/runtime.yaml').write_text(
        '# M5_E1 runtime (reports/vggt_five/M5_E1/operations/runtime.yaml) with loader 4 workers/prefetch 1:\n'
        '# qualified at 480 GiB and no measured loader starvation. Engineering only, never the objective.\n'
        + yaml.safe_dump(RUNTIME, sort_keys=False))
    campaign = dict(
        version='m5_gdn_s_sweep_campaign_v1', root=str(root), source=str(source), source_commit=commit,
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
        recovery_arm='S1', screening_arms=SCREENING, lanes=LANES)
    (root / 'campaign.json').write_text(json.dumps(campaign, indent=2) + '\n')
    print(root / 'campaign.json')


if __name__ == '__main__':
    main()
