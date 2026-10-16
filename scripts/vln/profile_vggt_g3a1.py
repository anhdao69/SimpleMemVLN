"""Owned four-H100 G3A1 diagnostics on complete episodes with online Omega (newton).

Uses the committed G3A1 runtime config; only the declared runtime knobs below may be
overridden for measurement. Never initializes production. Verifies every rank/update/
exposure, GPU limits, finite navigation+G3+A1 objectives, host guard and OOM counters.
"""
import argparse, hashlib, json, math, os, statistics, subprocess, time
from pathlib import Path
import yaml
from scripts.vln.profile_episodic import verify, _rows
from scripts.vln.profile_vggt_g1 import source_revision


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True); p.add_argument('--name', required=True)
    p.add_argument('--fixture', required=True); p.add_argument('--updates', type=int, default=3)
    p.add_argument('--workers', type=int); p.add_argument('--prefetch', type=int)
    p.add_argument('--attention-batch', type=int); p.add_argument('--vision-batch', type=int)
    p.add_argument('--host-guard-gib', type=float, default=460.8)
    p.add_argument('--max-reserved-gib', type=float, default=78.)
    p.add_argument('--save', action='store_true'); p.add_argument('--resume')
    p.add_argument('--first-update', type=int, default=1)
    p.add_argument('--config', default='configs/vggt/g3a1_newton_h4.yaml')
    a = p.parse_args()
    if not Path(a.fixture).is_file(): raise FileNotFoundError(a.fixture)
    out = Path(a.root) / 'experiments' / a.name
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    source_files = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                    for directory in ('src', 'scripts', 'configs', 'train') for f in sorted(Path(directory).rglob('*'))
                    if f.is_file() and f.suffix in ('.py', '.yaml', '.json', '.sh')}
    (out / 'source_manifest.json').write_text(json.dumps(source_files, indent=2))
    config = yaml.safe_load(Path(a.config).read_text())
    if config['research']['run_id'] != 'G3A1': raise ValueError('G3A1 profiler requires the G3A1 config')
    t, r = config['training'], config['runtime']
    if a.workers is not None:
        t.update(dataloader_num_workers=a.workers, dataloader_persistent_workers=a.workers > 0,
                 dataloader_prefetch_factor=(a.prefetch or 2) if a.workers else None)
    if a.attention_batch: r['window_attention_batch_steps'] = a.attention_batch
    if a.vision_batch: r['vision_microbatch_images'] = a.vision_batch
    config_path = out / 'runtime.yaml'; config_path.write_text(yaml.safe_dump(config))
    command = ['torchrun', '--standalone', '--nproc_per_node=4', '-m', 'qwen_vl.train.train_qwen',
               '--vln_config', 'configs/episodic/science_common.yaml', '--output_config', 'configs/vln_empty.yaml',
               '--memory_config', 'configs/vln_memory_window8.yaml', '--runtime_profile', str(config_path),
               '--deepspeed', 'configs/deepspeed_zero2_cpu_optimizer.json', '--manifest', a.fixture,
               '--model_name_or_path', os.environ['VLN_MODEL_PATH'], '--max-optimizer-updates', str(a.updates),
               '--debug-repeat-episodes', '--output_dir', str(out / 'train'), '--run-name', 'g3a1-diagnostic-' + a.name]
    command += ['--recovery-save-steps', '1'] if a.save else ['--profile-only']
    if a.resume: command += ['--resume_from_checkpoint', a.resume]
    start = time.time()
    status = subprocess.run(['python', '-m', 'scripts.vln.run_episodic_gate', '--out', str(out / 'supervisor'),
                             '--host-guard-gib', str(a.host_guard_gib), '--disk-path', str(a.root),
                             '--min-free-disk-gib', '220', '--', *command]).returncode
    resources = json.loads((out / 'supervisor/resources.json').read_text())
    check = verify(out / 'train', 4, a.updates, 8, a.max_reserved_gib, first_update=a.first_update)
    ranks = [_rows(out / 'train' / f'profile_rank{k}.jsonl') for k in range(4)]
    check['peak_allocated_gib'] = max((row['peak_allocated_gib'] for rows in ranks for row in rows), default=None)
    ends = [max(rows[i]['unix_end'] for rows in ranks) for i in range(min(map(len, ranks)))]
    intervals = [y - x for x, y in zip(ends, ends[1:])]
    exposures = [_rows(out / 'train' / f'exposures_rank{k}.jsonl') for k in range(4)]
    required = ('aux_loss_sum', 'aux_state_loss_sum', 'aux_relation_loss_sum')
    if not all(all(math.isfinite(e.get(key, float('nan'))) for key in required) and e.get('aux_pair_count', -1) >= 0
               for rank in exposures for e in rank):
        check['passed'] = False; check['problems'].append('Missing/nonfinite G3/A1 auxiliary evidence')
    supported = [e for rank in exposures for e in rank if e.get('aux_pair_count', 0) > 0]
    if supported and not all(e['aux_state_loss_sum'] > 0 and e['aux_relation_loss_sum'] > 0 for e in supported):
        check['passed'] = False; check['problems'].append('Supported pairs without positive G3 and A1 losses')
    result = dict(arm='G3A1', name=a.name, passed=status == 0 and check['passed'] and resources['status'] == 'PASS',
                  check=check, resources=resources, steady_seconds=statistics.mean(intervals) if intervals else None,
                  intervals=intervals, wall_seconds=time.time() - start, runtime=dict(training=t, runtime=r),
                  omp_threads=os.environ.get('OMP_NUM_THREADS'), fixture=a.fixture,
                  complete_episode_exposures=sum(map(len, exposures)),
                  aux_pairs=sum(e.get('aux_pair_count', 0) for rank in exposures for e in rank),
                  empty_support_exposures=sum(1 for rank in exposures for e in rank if e.get('aux_pair_count', 0) == 0),
                  aux_state_sum=sum(e.get('aux_state_loss_sum', 0) for rank in exposures for e in rank),
                  aux_relation_sum=sum(e.get('aux_relation_loss_sum', 0) for rank in exposures for e in rank),
                  cuda_toolkit=os.environ.get('CUDA_HOME'), source_commit=source_revision()[0],
                  source_status=source_revision()[1],
                  source_tree_sha256=hashlib.sha256(json.dumps(source_files, sort_keys=True).encode()).hexdigest(),
                  teacher_identity=config['research']['teacher_identity'],
                  fixture_sha256=hashlib.sha256(Path(a.fixture).read_bytes()).hexdigest())
    (out / 'result.json').write_text(json.dumps(result, indent=2)); print('RESULT', json.dumps(result), flush=True)
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
