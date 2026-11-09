"""S8N R/B training ablation: arm memories, guards, gates and deadline rules (no GPU, no Slurm)."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from qwen_vl.contracts import validate_config
from qwen_vl.research.episodic_contract import QWEN35_4B_PAIRS, parse_episodic_spec
from qwen_vl.research.gdn_sweep_contract import arm_memory_config, canonical_sha256
from qwen_vl.train.vln_runtime import resolve_config
from scripts.memory_sweep import rb_ablation as rb

ROOT = Path(__file__).resolve().parents[2]
COMMON = ROOT / 'configs/episodic/science_common.yaml'
ONE_EPOCH = ROOT / 'configs/vggt/g3_joint_one_epoch.yaml'
H4 = ROOT / 'configs/episodic/profile_h4.yaml'
FLOOR = {str(layer): 1e-3 * (layer + 1) for layer, _ in QWEN35_4B_PAIRS}
ARTIFACTS = dict(drift_denominator_floor=FLOOR, drift_denominator_floor_sha256=canonical_sha256(FLOOR),
                 calibration_sha256='0' * 64)
R4B0 = dict(mode='episodic_window_v1', recent_groups_including_current=4, episodic_budget_tokens=0,
            episodic_granularity='token', episodic_score='none', episodic_selector_seed=429,
            episodic_per_source_cap=None, evict_before_new_step=True, kv_guard_tokens=3584)


def _family(tmp_path, memory):
    path = tmp_path / 'memory.yaml'
    path.write_text(yaml.safe_dump({'memory': memory}))
    return resolve_config(COMMON, ROOT / 'configs/vln_empty.yaml', path, ONE_EPOCH, H4)


def test_guards_follow_prefix_plus_recent_groups_plus_bank():
    obs = dict(max_prefix_tokens=2048, max_step_group_tokens=384)
    guards = {arm: rb.guard(obs, s['recent'], s['budget']) for arm, s in rb.ARMS.items()}
    assert guards == dict(S8N_R4B600=4184, S8N_R4B2400=5984, R8B0=5120, R12B0=6656, S8N_R2B1200=4016,
                          S8N_R8B1200=6320)
    assert rb.guard(obs, 4, 1200) == 4784 and rb.guard(obs, 4, 0) == 3584      # S8N and R4B0 themselves


def test_arm_memories_change_only_r_b_and_guard_and_parse(tmp_path):
    s8n = json.loads(json.dumps(arm_memory_config('S8N', artifacts=ARTIFACTS)['memory']))
    families = dict(S8N=_family(tmp_path, s8n), R4B0=_family(tmp_path, R4B0))
    for name, cfg in families.items():                 # each family reproduces itself unchanged
        recent, budget = (4, 1200) if name == 'S8N' else (4, 0)
        assert rb.arm_memory(cfg['memory'], cfg['observations'], recent, budget) == cfg['memory']
    for arm, spec in rb.ARMS.items():
        family = families[spec['family']]
        memory = rb.arm_memory(family['memory'], family['observations'], spec['recent'], spec['budget'])
        changed = {k for k in memory if memory[k] != family['memory'][k]}
        assert changed and changed <= set(rb.MEMORY_KEYS_CHANGED) and set(memory) == set(family['memory'])
        cfg = deepcopy(family)
        cfg['memory'] = memory
        validate_config(deepcopy(cfg), 4)
        parsed = parse_episodic_spec(cfg)
        assert (parsed.recent_groups, parsed.budget_tokens) == (spec['recent'], spec['budget'])
        if spec['family'] == 'S8N':
            assert parsed.score == 'S8N' and parsed.version == 'episodic_window_gdn_sweep_v1'
            assert parsed.granularity == 'frame300' and spec['budget'] % 300 == 0
            generated = arm_memory_config('S8N', recent=spec['recent'], budget=spec['budget'],
                                          artifacts=ARTIFACTS)['memory']
            assert json.loads(json.dumps(generated)) == memory
        else:
            assert (parsed.version, parsed.score, parsed.needs_write_scores) == ('episodic_window_v1', 'none', False)
        # the yaml written for the arm round-trips to the same memory
        assert yaml.safe_load(rb.memory_yaml(arm, memory, spec['family']))['memory'] == memory


def test_arm_memory_does_not_mutate_the_family_memory():
    memory = deepcopy(R4B0)
    obs = dict(max_prefix_tokens=2048, max_step_group_tokens=384)
    out = rb.arm_memory(memory, obs, 8, 0)
    assert out['kv_guard_tokens'] == 5120 and memory == R4B0
    assert {k for k in out if out[k] != memory[k]} == {'recent_groups_including_current', 'kv_guard_tokens'}


def test_reproduction_gate_bounds(tmp_path):
    log = {1: dict(loss=0.48368, grad_norm=50.74), 2: dict(loss=0.48632, grad_norm=40.0), 3: dict(loss=0.50726, grad_norm=30.0)}
    logs = dict(S8N=log, R4B0=log)
    plan = {0: {0: ['a', 'b'], 1: ['c', 'd'], 2: ['e', 'f']}}
    good = dict(losses=[0.48368 * (1 + 1e-4), 0.48632, 0.51269], grad_norms=[50.74 * (1 + 1e-3), 40.0, 30.5], plan=plan)
    assert rb.compare_reproduction(logs, dict(S8N_REF=good, R4B0_REF=good))['passed']
    bad = dict(good, losses=[0.4850, 0.48632, 0.50726])
    assert not rb.compare_reproduction(logs, dict(S8N_REF=bad, R4B0_REF=good))['passed']
    other = dict(good, plan={0: {0: ['a', 'x'], 1: ['c', 'd'], 2: ['e', 'f']}})
    result = rb.compare_reproduction(logs, dict(S8N_REF=good, R4B0_REF=other))
    assert not result['passed'] and 'different episodes' in result['problems'][0]


def test_deadline_watchdog_requests_then_kills():
    end = 10_000.0
    assert rb.deadline_actions(end - 3000, end, 1500, 300) == 'none'
    assert rb.deadline_actions(end - 1400, end, 1500, 300) == 'request'
    assert rb.deadline_actions(end - 200, end, 1500, 300) == 'kill'
    assert rb.deadline_actions(end, None, 1500, 300) == 'none'


def _write_epoch(train, steps, resume_at=None):
    train.mkdir(parents=True, exist_ok=True)
    for rank in range(4):
        profile, exposures = [], []
        for update in range(1, steps + 1):
            profile.append(dict(update=update, peak_reserved_gib=60.0, cuda_ooms=0))
        for update in range(steps):
            for slot in range(2):
                exposures.append(dict(update=update, episode_uid=f'r{rank}u{update}s{slot}', loss_sum=1.0))
        if resume_at:                                # a lost tail replayed after a resume
            exposures += [e for e in exposures if e['update'] >= resume_at]
        (train / f'profile_rank{rank}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in profile))
        (train / f'exposures_rank{rank}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in exposures))


def test_epoch_audit_uses_zero_based_exposures_and_collapses_replays(tmp_path):
    _write_epoch(tmp_path / 'a', 5)
    audit, plan = rb.audit_epoch(tmp_path / 'a', steps=5)
    assert audit['passed'], audit
    assert sorted(plan[0]) == [0, 1, 2, 3, 4] and plan[3][4] == ['r3u4s0', 'r3u4s1']
    _write_epoch(tmp_path / 'b', 5, resume_at=3)
    audit_b, plan_b = rb.audit_epoch(tmp_path / 'b', steps=5)
    assert audit_b['passed'] and plan_b == plan and audit_b['plan_sha256'] == audit['plan_sha256']
    _write_epoch(tmp_path / 'c', 4)
    assert not rb.audit_epoch(tmp_path / 'c', steps=5)[0]['passed']


def test_loss_correlation_detects_shared_order():
    import random
    rng = random.Random(0)
    base = {s: 0.1 + rng.random() for s in range(1, 400)}
    same = {s: v * 0.98 + 0.01 * rng.random() for s, v in base.items()}
    shuffled = dict(zip(base, rng.sample(list(base.values()), len(base))))
    assert rb.loss_correlation(base, same) > 0.95
    assert abs(rb.loss_correlation(base, shuffled)) < 0.3


def test_manifest_fingerprint_ignores_image_roots(tmp_path):
    row = dict(episode_uid='r2rce:train:x:1:1', instruction='go', steps=[
        dict(action_name='MOVE_FORWARD', rgb_path='/a/b/0.png', is_valid=True)])
    moved = deepcopy(row)
    moved['steps'][0]['rgb_path'] = '/other/root/0.png'
    (tmp_path / 'a.jsonl').write_text(json.dumps(row) + '\n')
    (tmp_path / 'b.jsonl').write_text(json.dumps(moved) + '\n')
    a, b = rb.manifest_fingerprint(tmp_path / 'a.jsonl'), rb.manifest_fingerprint(tmp_path / 'b.jsonl')
    assert a == b and a['episodes'] == 1 and a['steps'] == 1 and a['r2rce'] == 1


def test_cache_relief_walks_frames_then_trees_and_skips_links(tmp_path):
    frame = tmp_path / 'f.png'
    frame.write_bytes(b'x' * 10)
    tree = tmp_path / 'tree'
    (tree / 'sub').mkdir(parents=True)
    (tree / 'sub' / 'a.bin').write_bytes(b'y' * 5)
    (tree / 'link').symlink_to(frame)
    manifest = tmp_path / 'm.jsonl'
    manifest.write_text(json.dumps(dict(steps=[dict(rgb_path=str(frame))])) + '\n')
    paths = list(rb.cached_paths(manifest, [tree, tmp_path / 'missing']))
    assert paths == [str(frame), str(tree / 'sub' / 'a.bin')]
    assert [rb.drop_cache(p) for p in paths] == [10, 5] and rb.drop_cache(tmp_path / 'nope') == 0
    assert frame.read_bytes() == b'x' * 10
