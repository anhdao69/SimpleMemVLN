"""Queue/recipe invariants of the S-sweep campaign controller (no GPU, no Slurm)."""
import json
from pathlib import Path
import pytest
import yaml
from qwen_vl.train.recovery import mark_complete
from qwen_vl.train.vln_runtime import resolve_config
from scripts.memory_sweep import s_campaign
from scripts.memory_sweep.init_campaign import LANES, RUNTIME, SCREENING
from scripts.memory_sweep.s_campaign import (
    Lane, choose_threads, expected_config, optimizer_step_estimate, prune_rolling, ready, train_command)

ROOT = Path(__file__).resolve().parents[2]


def _resolve(memory, runtime):
    return resolve_config(ROOT / 'configs/episodic/science_common.yaml', ROOT / 'configs/vln_empty.yaml',
                          memory, ROOT / 'configs/vggt/g3_joint_one_epoch.yaml', runtime)


def test_sweep_recipe_equals_m5_e1_except_memory_and_loader(tmp_path):
    reference = _resolve(ROOT / 'configs/episodic/arm_m5.yaml', ROOT / 'reports/vggt_five/M5_E1/operations/runtime.yaml')
    assert (reference['training']['seed'], reference['training']['expected_total_steps'],
            reference['training']['warmup_steps'], reference['training']['dataloader_num_workers']) == (429, 3852, 116, 8)
    runtime = tmp_path / 'runtime.yaml'
    runtime.write_text(yaml.safe_dump(RUNTIME))
    for arm in ('S1', 'S2', 'S3', 'S4', 'S5', 'S7', 'S9'):
        path = ROOT / f'configs/memory_sweep/arm_{arm}.yaml'
        cfg = _resolve(path, runtime)
        assert cfg == expected_config(reference, yaml.safe_load(path.read_text())['memory'])
        changed = {section for section in cfg if cfg[section] != reference[section]}
        assert changed == {'memory', 'training'}
        assert {k for k in cfg['training'] if cfg['training'][k] != reference['training'][k]} == {
            'dataloader_num_workers', 'dataloader_prefetch_factor'}
    research = dict(reference, research=dict(run_id='G3_M5'))
    with pytest.raises(ValueError):
        expected_config(research, {})
    random = json.loads(json.dumps(reference)); random['memory']['episodic_score'] = 'random_hash_v1'
    with pytest.raises(ValueError):
        expected_config(random, {})


def test_lanes_split_every_screening_s_arm_once_and_calibrated_arms_wait_for_lane_b():
    arms = LANES['A']['arms'] + LANES['B']['arms']
    assert sorted(arms) == sorted(a for a in SCREENING if a.startswith('S') and a != 'S0')
    assert len(arms) == len(set(arms)) == 10 and 'S0' not in arms          # S0 is M5_E1
    assert not set(LANES['A']['arms']) & set(s_campaign.CALIBRATED)      # lane A trains before calibration exists
    assert LANES['A']['qualifies'] and not LANES['B']['qualifies']


def test_allocation_ready_rule():
    assert ready(['5248.0', '5248.extern', '5248.9'], '5248.9', '5248')
    assert not ready(['5248.0', '5248.3', '5248.9'], '5248.9', '5248')
    assert not ready(['4925.147', '4925.150'], '4925.150', '4925')


def test_thread_choice_requires_a_clear_optimizer_gain_without_regression():
    base = dict(threads=4, step_seconds=1.5, update_seconds=84.0)
    assert choose_threads(base, dict(threads=8, step_seconds=0.9, update_seconds=83.0)) == 8
    assert choose_threads(base, dict(threads=8, step_seconds=1.3, update_seconds=80.0)) == 4
    assert choose_threads(base, dict(threads=8, step_seconds=0.9, update_seconds=90.0)) == 4


def test_optimizer_step_estimate_from_repeated_episode_microsteps(tmp_path):
    for rank, step in enumerate([1.4, 1.6, 1.5, 1.7]):
        rows = [dict(update=1, backward_and_step_overhead_seconds=9.0), dict(update=1, backward_and_step_overhead_seconds=12.0),
                dict(update=2, backward_and_step_overhead_seconds=8.0),
                dict(update=2, backward_and_step_overhead_seconds=8.0 + step)]
        (tmp_path / f'microtimes_rank{rank}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (tmp_path / f'profile_rank{rank}.jsonl').write_text(json.dumps(dict(update=2, seconds=80.0 + rank)) + '\n')
    result = optimizer_step_estimate(tmp_path)
    assert result['step_seconds'] == pytest.approx(1.55) and result['update_seconds'] == 83.0


def _checkpoint(root, step, boundary):
    path = root / f'checkpoint-{step}'
    (path / 'global_step').mkdir(parents=True)
    for name in ['pytorch_model.bin', 'resume_contract.json', 'navigation.json'] + [f'rng_state_{r}.pth' for r in range(4)]:
        (path / name).write_text('x')
    (path / 'trainer_state.json').write_text(json.dumps(dict(global_step=step, epoch=1.0)))
    for r in range(4):
        (path / 'global_step' / f'rank{r}_optim_states.pt').write_text('x')
    (path / 'global_step' / 'mp_rank_00_model_states.pt').write_text('x')
    mark_complete(path, 4, boundary)
    return path


def test_pruning_keeps_epoch_checkpoint_and_export_and_refuses_early(tmp_path):
    train = tmp_path / 'train'
    for step in (3700, 3800):
        _checkpoint(train, step, False)
    with pytest.raises(Exception):
        prune_rolling(train)                                   # no epoch checkpoint yet
    _checkpoint(train, 3852, True)
    with pytest.raises(ValueError, match='export'):
        prune_rolling(train)                                   # no final export yet
    (train / 'final').mkdir()
    other = tmp_path / 'other-run/checkpoint-100'
    other.mkdir(parents=True)
    assert sorted(prune_rolling(train)) == ['checkpoint-3700', 'checkpoint-3800']
    assert (train / 'checkpoint-3852').is_dir() and (train / 'final').is_dir() and other.is_dir()


def test_production_and_smoke_commands_match_the_m5_recipe(tmp_path):
    campaign = type('C', (), {})()
    campaign.data = dict(manifest='/m.jsonl', model='/model', seed=429, min_free_disk_gib=220)
    campaign.base, campaign.configs = Path('/base'), tmp_path
    campaign.arm_memory_path = lambda arm: tmp_path / f'arm_{arm}.yaml'
    production = train_command(campaign, 'S1', tmp_path / 'run', production=True, host_guard=432.0)
    text = ' '.join(production)
    for fragment in ('--recovery-save-steps 100', '--first-recovery-save-step 10', '--resume_from_checkpoint auto',
                     '--nproc_per_node=4', 'configs/vggt/g3_joint_one_epoch.yaml',
                     'configs/deepspeed_zero2_cpu_optimizer.json', '--host-guard-gib 432.0', '--min-free-disk-gib 220'):
        assert fragment in text
    assert '--max-optimizer-updates' not in production and '--debug-repeat-episodes' not in production
    smoke = train_command(campaign, 'S1', tmp_path / 'smoke', manifest='/rxr.jsonl', updates=2)
    assert '--profile-only' in smoke and '--debug-repeat-episodes' in smoke and '--recovery-save-steps' not in smoke


def test_run_arms_continues_after_an_arm_error_and_honors_controls(tmp_path, monkeypatch):
    lane = Lane.__new__(Lane)
    lane.dir, lane.name, lane.own = tmp_path, 'A', '5248.9'
    lane.spec = dict(arms=['S1', 'S2', 'S9', 'S4'])
    calls = []

    def train(arm):
        calls.append(arm)
        if arm == 'S2':
            raise RuntimeError('boom')
        return 'complete'
    monkeypatch.setattr(lane, 'train_arm', train)
    (tmp_path / 'SKIP_S9').write_text('')
    outcomes = lane.run_arms()
    assert calls == ['S1', 'S2', 'S4'] and outcomes['S9'] == 'skipped' and outcomes['S2'].startswith('error')
    (tmp_path / 'STOP_AFTER_CURRENT').write_text('')
    calls.clear()
    assert lane.run_arms() == {} and calls == []
    history = [json.loads(line)['status'] for line in (tmp_path / 'history.jsonl').read_text().splitlines()]
    assert history == ['arm-error', 'lane-complete', 'stopped-by-request']


def _replay(arm, dataset, passed=True, rms=0.04, nll=0.3, overlap=None):
    record = dict(arm=arm, dataset=dataset, passed=passed, rms_logit_error=rms, max_logit_error=0.5,
                  argmax_agreement=1.0, decoded_actions_match=1.0, plan_comparisons=512, plan_id_mismatches=448,
                  mean_selected_overlap=0.99, min_selected_overlap=0.98, max_relative_score_error=0.1,
                  max_resident_kv=2600, kv_guard=4784, action_nll=nll)
    if arm == 'S0':
        overlap = dict(mean_overlap=1.0, min_overlap=1.0, p10_overlap=1.0, mean_jaccard=1.0, min_jaccard=1.0)
    if arm != 'S0_legacy':                     # the reference record has no self-overlap (as in real output)
        record['overlap_vs_S0_legacy'] = overlap
    return record


def test_replay_policy_requires_exact_s0_parity_and_reports_tolerance_failures():
    from scripts.memory_sweep.s_campaign import summarize_replay
    base = []
    for d in ('r2rce', 'rxrce'):
        legacy_pass = d == 'r2rce'                  # inherited allclose miss on rxrce
        base += [_replay('S0_legacy', d, legacy_pass), _replay('S0_legacy', d, legacy_pass),
                 _replay('S0', d, legacy_pass), _replay('S1', d, d == 'r2rce', overlap=dict(mean_overlap=.5))]
    summary = summarize_replay(base)
    assert summary['serving_tolerance_failed'] == ['S1'] and summary['s0_inherited_allclose_failures'] == ['rxrce']
    assert summary['rms_failed'] == [] and summary['records']['S1/r2rce']['overlap_vs_S0_legacy'] == {'mean_overlap': .5}
    drift = [dict(r) for r in base]; drift[2]['action_nll'] = 0.31                  # sweep S0 != legacy S0
    with pytest.raises(RuntimeError, match='Sweep S0 differs'):
        summarize_replay(drift)
    shard = [dict(r) for r in base]; shard[1]['rms_logit_error'] = 0.05             # GPU nondeterminism
    with pytest.raises(RuntimeError, match='deterministic'):
        summarize_replay(shard)
    bad = [dict(r) for r in base]
    for r in bad:
        if r['arm'] in ('S0', 'S0_legacy') and r['dataset'] == 'rxrce':
            r['rms_logit_error'] = 0.2
    with pytest.raises(RuntimeError, match='RMS'):
        summarize_replay(bad)
    with pytest.raises(RuntimeError, match='parity'):
        summarize_replay([r for r in base if r['arm'] != 'S0'])
