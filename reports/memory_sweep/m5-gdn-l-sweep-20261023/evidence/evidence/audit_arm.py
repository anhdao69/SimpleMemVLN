"""Production audit of an L arm's first checkpoint: same checks as the S1 audit plus the embedded ranker."""
import glob, hashlib, json, math, sys, time
import yaml
from pathlib import Path

arm = sys.argv[1]
R = Path('/mnt/data/vmo-ai-task/anhdh35/m5-gdn-l-sweep-20261023')
M5 = Path('/mnt/data/vmo-ai-task/anhdh35/m5-one-epoch-worker2-20261018/production/m5_joint_e1/train')
S1 = Path('/mnt/data/vmo-ai-task/anhdh35/m5-gdn-s-sweep-dabf4ad-20261020/runs/S1_seed429_e1/train')
run = R / f'runs/{arm}_seed429_e1'
train = run / 'train'
ck = train / 'checkpoint-10'
state = json.loads((ck / 'trainer_state.json').read_text())
logs = [h for h in state['log_history'] if 'loss' in h][:10]
m5 = [h for h in json.loads((M5 / 'trainer_state.json').read_text())['log_history'] if 'loss' in h][:10]
s1 = [h for h in json.loads((S1 / 'trainer_state.json').read_text())['log_history'] if 'loss' in h][:10]


def exposures(root, limit):
    out = {}
    for rank in range(4):
        rows = [json.loads(l) for l in open(root / f'exposures_rank{rank}.jsonl') if l.strip()]
        out[rank] = [(r.get('update'), r.get('episode_uid')) for r in rows if r.get('update', 0) <= limit]
    return out


ours = exposures(train, 10**9)
max_update = max(u for v in ours.values() for u, _ in v)
limit = max_update - 1                       # last update may still be in flight
ours = {k: [x for x in v if x[0] <= limit] for k, v in ours.items()}
ref = exposures(M5, limit)
compared = sum(len(v) for v in ours.values())
identical = all(ours[k] == ref[k][:len(ours[k])] and len(ours[k]) == len(ref[k]) for k in ours)

nav = json.loads((train / 'navigation.json').read_text())
memory = nav['config']['memory']
fit_ledger = json.loads((R / 'rankers/fit_report.json').read_text())
artifact = memory['artifacts']['ranker']
sys.path.insert(0, str(R.parent / 'm5-gdn-l-source-022e526/src'))
from qwen_vl.research.gdn_ranker import canonical_sha256
canonical = canonical_sha256(artifact)
resolved = yaml.safe_load(open(run / 'train' / 'resolved_config.yaml')) if (run / 'train' / 'resolved_config.yaml').exists() else None
fitted = json.loads((R / f'rankers/{arm}_artifact.json').read_text())
identity = json.loads((run / 'run_identity.json').read_text())
m5_nav = json.loads((M5 / 'navigation.json').read_text())['config']
nav_m5 = {k: v for k, v in m5_nav.items() if k != 'memory'}
diff_keys = {}
for section in set(nav_m5) | set(nav['config']):
    if section == 'memory':
        continue
    a, b = nav_m5.get(section), nav['config'].get(section)
    if a != b:
        if isinstance(a, dict) and isinstance(b, dict):
            diff_keys[section] = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
        else:
            diff_keys[section] = (a, b)

micro = {}
for rank in range(4):
    for l in open(train / f'microtimes_rank{rank}.jsonl'):
        if l.strip():
            r = json.loads(l)
            micro.setdefault(r['update'], []).append(r['forward_seconds'] + r['backward_and_step_overhead_seconds'])
s1_micro = {}
for l in open(S1 / 'microtimes_rank0.jsonl'):
    if l.strip():
        r = json.loads(l)
        if r['update'] <= limit:
            s1_micro.setdefault(r['update'], []).append(r['forward_seconds'] + r['backward_and_step_overhead_seconds'])
ours_rank0 = {}
for l in open(train / 'microtimes_rank0.jsonl'):
    if l.strip():
        r = json.loads(l)
        if 2 <= r['update'] <= limit:
            ours_rank0.setdefault(r['update'], []).append(r['forward_seconds'] + r['backward_and_step_overhead_seconds'])
ours_fwd = [json.loads(l)['forward_seconds'] for l in open(train / 'microtimes_rank0.jsonl') if l.strip() and 2 <= json.loads(l)['update'] <= limit]
s1_fwd = [json.loads(l)['forward_seconds'] for l in open(S1 / 'microtimes_rank0.jsonl') if l.strip() and 2 <= json.loads(l)['update'] <= limit]

report = dict(
    utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), arm=arm,
    checkpoint_10=dict(step=state['global_step'], files=len(list(ck.iterdir()))),
    losses_1_10=[h['loss'] for h in logs], grad_norms_1_10=[h['grad_norm'] for h in logs],
    learning_rates_1_10=[h['learning_rate'] for h in logs],
    learning_rates_equal_m5_e1=[h['learning_rate'] for h in logs] == [h['learning_rate'] for h in m5],
    m5_e1_losses_1_10=[h['loss'] for h in m5], s1_losses_1_10=[h['loss'] for h in s1],
    all_finite=all(math.isfinite(h['loss']) and math.isfinite(h['grad_norm']) for h in logs),
    exposures_compared=compared, exposure_order_identical_to_m5_e1=identical, max_update_seen=max_update,
    memory_score_id=memory['score_id'], memory_backend=memory['scorer_backend'],
    embedded_ranker_sha256=memory['artifacts'].get('ranker_sha256'), embedded_ranker_canonical_sha256=canonical,
    non_memory_config_equals_m5_e1=nav_m5 == {k: v for k, v in nav['config'].items() if k != 'memory'},
    fitted_artifact_sha256=fit_ledger['arms'][arm]['artifact_sha256'],
    embedded_equals_fitted=artifact == fitted, identity_ranker_sha256=identity.get('ranker_sha256'),
    rank0_micro_seconds_mean_updates_2_to_limit=sum(sum(v) for v in ours_rank0.values()) / max(1, len(ours_rank0)),
    s1_rank0_micro_seconds_mean_same_updates=sum(sum(v) for u, v in s1_micro.items() if u >= 2) / max(1, len([u for u in s1_micro if u >= 2])),
    rank0_forward_mean=sum(ours_fwd) / len(ours_fwd), s1_rank0_forward_mean=sum(s1_fwd) / len(s1_fwd),
)
report['non_memory_config_differences_vs_m5_e1'] = diff_keys
print(json.dumps(report, indent=1, default=str))
