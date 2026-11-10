"""CPU-only publication of one completed S8N R/B ablation arm to the Hub folder ``<ARM>/`` of anhdao69/stage1_S8N.

Adapted from the M5_S S-arm/follow-up publishers. The lane declared the arm complete with a passing audit; the
bound run identity re-derives from the sealed campaign source; all-rank coverage, finite losses and the campaign's
single exposure plan are re-checked; serving files are hard links of checkpoint-3852's files (same inodes) and
the real navigation model strictly reloads on CPU. One Hub commit adds only ``<ARM>/`` on an optimistic parent
revision and never overwrites; every file is then verified by SHA-256 and size at the immutable revision.

    publish_rb.py --create-repo          one-time: create the (private) repository with its README
    publish_rb.py ARM [--check-only]     validate, stage and hash; upload unless --check-only
Outputs (staging, receipts) go to <campaign root>/publication/<ARM>/.
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ.pop(key, None)

ROOT = Path('/groups/yshang/an221229/checkpoints/m5-s8n-rb-ablation-20261109')
CAMPAIGN = ROOT / 'campaign.json'
OPS = ROOT / 'publication'
REPO = 'anhdao69/stage1_S8N'
SERVING = {'pytorch_model.bin', 'navigation.json', 'config.json', 'generation_config.json',
           'preprocessor_config.json', 'processor_config.json', 'tokenizer_config.json', 'tokenizer.json',
           'chat_template.jinja'}
REQUIRED = ('pytorch_model.bin', 'navigation.json', 'config.json', 'tokenizer.json', 'processor_config.json')

campaign_data = json.loads(CAMPAIGN.read_text())
SOURCE = Path(campaign_data['source'])
SOURCE_COMMIT = campaign_data['source_commit']
sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE)]
from scripts.memory_sweep.rb_ablation import (  # noqa: E402
    ARMS, EXPECTED_STEPS, FAMILIES, REPRODUCED_UPDATES, TRAINING_CODE_COMMIT, WORLD, Campaign, audit_epoch)
from scripts.memory_sweep.s_campaign import latest_resources, run_complete, sha, write  # noqa: E402

FAMILY_MEMORY = {
    'S8N': lambda m: f"""S8N's memory (`episodic_window_gdn_sweep_v1`, score S8N: net Gated-DeltaNet state drift
‖S_after − S_before‖_F / max(‖S_before‖_F, ε_l), whole 300-token frames selected) with S8N's frozen calibrated
drift floors ε_l, so the scorer is S8N's exactly. **R = {m['recent_groups_including_current']}** most recent
observation groups (including the current one) are always kept, plus an episodic bank of **B =
{m['episodic_budget_tokens']} tokens** ({m['episodic_budget_tokens'] // 300} frames) chosen by the S8N score.
S8N itself is R4/B1200.""",
    'R4B0': lambda m: f"""R4B0's bank-free memory (`episodic_window_v1`, no episodic score): only the **R =
{m['recent_groups_including_current']}** most recent observation groups (including the current one) are kept
in the K/V cache, **B = 0**. R4B0 itself is R4/B0.""",
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def declared_complete(arm):
    """The lane's own ``arm-complete`` record with a passing audit, or None."""
    found = None
    for line in (ROOT / 'events.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('status') == 'arm-complete' and row.get('arm') == arm and row['audit']['passed']:
            found = row
    return found


def validate_training(arm):
    from qwen_vl.train.recovery import validate_complete
    from scripts.vln.profile_episodic import verify
    campaign = Campaign(CAMPAIGN)
    run = campaign.run_dir(arm)
    identity = json.loads((run / 'run_identity.json').read_text())
    assert identity == campaign.run_identity(arm), 'bound identity differs from the sealed source'
    seal = campaign.sealed()
    assert seal and identity['qualification_sha256'] == sha(campaign.shared / 'QUALIFIED.json')
    config = identity['config']
    spec = ARMS[arm]
    memory = config['memory']
    assert (memory['recent_groups_including_current'], memory['episodic_budget_tokens']) == (spec['recent'], spec['budget'])
    assert config['training']['seed'] == 429 and config['training']['epochs'] == 1
    assert config['training']['expected_total_steps'] == EXPECTED_STEPS
    declared = declared_complete(arm)
    assert declared, f'{arm}: the lane has not declared the arm complete'
    assert run_complete(run, config)
    resources = latest_resources(run)
    assert resources['status'] == 'PASS' and resources['new_oom_events'] == resources['new_oom_kill_events'] == 0
    checks = verify(run / 'train', WORLD, EXPECTED_STEPS, 8, 78.0)
    assert checks['passed'], checks['problems']
    for key in ('losses', 'grad_norms'):
        checks.pop(key, None)
    audit, plan = audit_epoch(run / 'train')
    assert audit['passed'], audit['problems']
    flat = {str(r): {str(u): uids for u, uids in v.items()} for r, v in plan.items()}
    assert flat == json.loads((campaign.shared / 'exposure_plan.json').read_text()), 'exposure plan differs'
    reference = json.loads((campaign.shared / 'reproduction/comparison.json').read_text())['plan']
    assert {str(r): {str(u): plan[r][u] for u in range(REPRODUCED_UPDATES)} for r in plan} == reference
    checks.update(exposures_identical_to_campaign_plan=sum(len(u) for r in plan.values() for u in r.values()),
                  exposure_plan_sha256=audit['plan_sha256'], first_updates_match_s8n_r4b0_reproduction=True,
                  loss_correlation_vs=declared['audit']['loss_correlation_vs'])
    checkpoint = run / f'train/checkpoint-{EXPECTED_STEPS}'
    record = validate_complete(checkpoint)
    assert record['step'] == EXPECTED_STEPS and record['world_size'] == WORLD and record['epoch_boundary']
    trainer = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert trainer['global_step'] == trainer['max_steps'] == EXPECTED_STEPS and trainer['epoch'] == 1
    navigation = json.loads((checkpoint / 'navigation.json').read_text())
    assert navigation['config'] == config
    return campaign, run, identity, checkpoint, record, resources, checks, navigation, declared


def stage(checkpoint, navigation, root):
    """Hard-linked serving files, finite tensors and strict CPU reload of the real navigation model."""
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    from qwen_vl.data.episode_serializer import EpisodeSerializer
    from qwen_vl.models.nav_model import SimpleMemVLNForNavigation
    export = root / 'model'
    export.mkdir(parents=True, exist_ok=True)
    for path in checkpoint.iterdir():
        if path.is_file() and path.name in SERVING:
            destination = export / path.name
            if not destination.exists():
                os.link(path, destination)
            assert os.path.samefile(path, destination), destination
    assert all((export / name).exists() for name in REQUIRED)
    assert not torch.cuda.is_available()
    metadata = json.loads((export / 'navigation.json').read_text())
    assert metadata == navigation
    weights = torch.load(export / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    assert not any(k.startswith('research_aux.') for k in weights)
    assert all(torch.isfinite(t).all().item() for t in weights.values() if t.is_floating_point()), 'nonfinite tensor'
    base_config = AutoConfig.from_pretrained(campaign_data['model'], local_files_only=True)
    with torch.device('meta'):
        backbone = Qwen3_5ForConditionalGeneration(base_config)
        model = SimpleMemVLNForNavigation(backbone, metadata['config'])
    model.load_state_dict(weights, strict=True, assign=True)
    assert not hasattr(model, 'research_aux')
    assert all(p.device.type == 'cpu' for p in model.parameters())
    processor = AutoProcessor.from_pretrained(export, local_files_only=True)
    assert EpisodeSerializer(processor, metadata['config']).metadata() == metadata
    audit = dict(passed=True, cpu_only=True, strict_architecture_reload=True, exact_serving_tensors=len(weights),
                 serving_files_are_checkpoint_hard_links=True, removed_training_tensors=[],
                 serializer_metadata_exact=True, memory_metadata_exact=True, all_model_tensors_finite=True,
                 completed_model_gpu_replay=False)
    write(root / 'export_validation.json', audit)
    print('CPU_STRICT_EXPORT_VALIDATED', len(weights), flush=True)
    return export


def readme(campaign, arm, run, identity, resources, checks):
    spec = ARMS[arm]
    memory = identity['config']['memory']
    columns = [(arm, json.loads((run / 'train/reports/epoch-1.json').read_text()))]
    columns += [(f, json.loads((campaign.reference / f / 'epoch-1.json').read_text())) for f in FAMILIES]
    rows = '\n'.join(f'| {label} | ' + ' | '.join(f'{r[key]:.5f}' for _, r in columns) + ' |'
                     for label, key in (('Action-weighted epoch loss', 'loss_action_weighted'),
                                        ('First update loss', 'first_update_loss'),
                                        ('Final update loss', 'last_update_loss')))
    training = identity['config']['training']
    corr = checks['loss_correlation_vs']
    return f"""# {arm} — S8N R/B memory ablation (stage 1), completed epoch

{FAMILY_MEMORY[spec['family']](memory)}
K/V guard {memory['kv_guard_tokens']} tokens = 2048 prefix + {spec['recent']} × 384 + {spec['budget']}.

This folder is the arm's serving model; `provenance/` holds the evidence and `SHA256SUMS.json` the published hashes.

## Training (matched to S8N and R4B0)

Asserted before launch: the resolved config equals {spec['family']}'s published resolved config except the memory
keys `recent_groups_including_current`, `episodic_budget_tokens` and `kv_guard_tokens`; the runtime file is
S8N's/R4B0's byte for byte. Training code `{TRAINING_CODE_COMMIT}` (the S-sweep source), original Qwen3.5-4B
revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, joint R2R + English RxR15deg (30,815 episodes), one epoch of
3,852 optimizer updates, global batch 8 (4 H100 x 1 episode x accumulation 2), seed {training['seed']}, peak LR
{training['backbone_lr']} with 116 warmup updates and cosine decay to 0.1, Adam 0.9/0.95, weight decay
{training['weight_decay']}, clip 1, ZeRO-2 with CPU optimizer offload.

| | {' | '.join(label for label, _ in columns)} |
|---|{'---:|' * len(columns)}
{rows}

Per-update loss correlation after warmup: {corr['S8N']:.4f} vs S8N, {corr['R4B0']:.4f} vs R4B0.
Training loss is not navigation success. Wall time {resources['seconds'] / 3600:.2f} h including load and saves
(newton, 4 x H100); no GPU or host OOM.

## Verification

Before training, this campaign source reproduced S8N's and R4B0's published first updates (loss and gradient
norm) from their own configs on newton and both saw the same episodes. The lane declared this run complete; the
bound identity re-derives from the sealed source; all 3,852 updates and exposures are covered on every rank with
finite losses; every ablation arm trains on one identical exposure plan (SHA-256 `{checks['exposure_plan_sha256']}`)
whose first updates equal the S8N/R4B0 reproduction; checkpoint-3852 is a complete epoch-boundary recovery
checkpoint. Serving files are checkpoint-3852's own files and the real navigation model strictly reloads on CPU
with exact serializer and memory metadata. No completed-model GPU replay is claimed.

## Use

Code: SimpleMemVLN branch `m5_s8n_rb_ablation`, campaign source `{SOURCE_COMMIT}` (training code
`{TRAINING_CODE_COMMIT}` plus the ablation scripts only). Load with
`qwen_vl.train.vln_runtime.load_checkpoint('{arm}', base_model_path)` after downloading this folder.
These are navigation-wrapper weights for inference; optimizer/RNG state stays on the cluster.
"""


def collect(campaign, arm, run, identity, checkpoint, record, resources, checks, declared, root, export):
    write(root / 'training_validation.json', checks)
    provenance = root / 'provenance'
    provenance.mkdir(exist_ok=True)
    supervisor = sorted(run.glob('supervisor-*'))[-1]
    for source, name in [(run / 'train/reports/epoch-1.json', 'epoch-1.json'),
                         (run / 'train/resolved_config.json', 'resolved_config.json'),
                         (run / 'train/run_timing.json', 'run_timing.json'),
                         (checkpoint / 'trainer_state.json', 'trainer_state.json'),
                         (supervisor / 'resources.json', 'completion_resources.json'),
                         (run / 'completion_audit.json', 'completion_audit.json'),
                         (root / 'export_validation.json', 'publication_validation.json'),
                         (root / 'training_validation.json', 'training_validation.json'),
                         (campaign.shared / 'QUALIFIED.json', 'qualification_pass.json'),
                         (run / 'run_identity.json', 'run_identity.json'),
                         (campaign.arm_memory_path(arm), f'arm_{arm}.yaml'),
                         (campaign.configs / 'runtime.yaml', 'runtime.yaml')]:
        (provenance / name).write_bytes(source.read_bytes())
    weights = sha(export / 'pytorch_model.bin')
    if not (provenance / 'source_and_completion.json').exists():
        write(provenance / 'source_and_completion.json', dict(
            campaign=campaign_data['version'], source_commit=SOURCE_COMMIT, training_code_commit=TRAINING_CODE_COMMIT,
            source_directory=str(SOURCE), checkpoint=str(checkpoint), arm=arm, family=ARMS[arm]['family'],
            recent_groups_including_current=ARMS[arm]['recent'], episodic_budget_tokens=ARMS[arm]['budget'],
            completed_epoch=1, optimizer_updates=EXPECTED_STEPS, global_batch=8, recovery_inventory=record,
            controller_completion=declared, export_weights_sha256=weights, verified_utc=now()))
    assert json.loads((provenance / 'source_and_completion.json').read_text())['export_weights_sha256'] == weights
    (root / 'README.md').write_text(readme(campaign, arm, run, identity, resources, checks))
    files = {f'{arm}/{p.name}': p for p in export.iterdir() if p.is_file() and p.name in SERVING}
    files.update({f'{arm}/provenance/{p.name}': p for p in provenance.iterdir() if p.is_file()})
    files[f'{arm}/README.md'] = root / 'README.md'
    digests = {name: dict(sha256=weights if name.endswith('/pytorch_model.bin') else sha(path),
                          size=path.stat().st_size) for name, path in files.items()}
    write(root / 'SHA256SUMS.json', digests)
    files[f'{arm}/SHA256SUMS.json'] = root / 'SHA256SUMS.json'
    digests[f'{arm}/SHA256SUMS.json'] = dict(sha256=sha(root / 'SHA256SUMS.json'),
                                             size=(root / 'SHA256SUMS.json').stat().st_size)
    print('HASHED', len(files), flush=True)
    return files, digests


REPO_README = """# stage1_S8N — S8N recent-window / episodic-bank (R/B) ablation

SimpleMemVLN on Qwen3.5-4B, one epoch on joint R2R + English RxR15deg with exactly the recipe of the M5 GDN S-sweep
arms S8N (R4/B1200) and R4B0 (R4/B0) in `anhdao69/M5_S`: same training code, base model, data, seed and data order,
schedule and runtime file. Only R (recent observation groups kept, including the current one), B (episodic bank
tokens, S8N-scored 300-token frames) and the derived K/V guard change.

| Folder | Memory family | R | B |
|---|---|---:|---:|
| `S8N_R4B600/` | S8N | 4 | 600 |
| `S8N_R4B2400/` | S8N | 4 | 2400 |
| `R8B0/` | R4B0 (no bank) | 8 | 0 |
| `R12B0/` | R4B0 (no bank) | 12 | 0 |
| `S8N_R2B1200/` | S8N | 2 | 1200 |
| `S8N_R8B1200/` | S8N | 8 | 1200 |

References: `anhdao69/M5_S` folders `S8N/` (R4/B1200) and `R4B0/` (R4/B0). Folders appear as runs complete; each
has its own README, `provenance/` and `SHA256SUMS.json`.
"""


def create_repo():
    from huggingface_hub import HfApi
    api = HfApi()
    assert api.whoami()['name'].lower() == 'anhdao69'
    api.create_repo(REPO, repo_type='model', private=True, exist_ok=False)
    api.upload_file(path_or_fileobj=REPO_README.encode(), path_in_repo='README.md', repo_id=REPO,
                    commit_message='Describe the S8N R/B ablation repository')
    print('CREATED', REPO, flush=True)


def attributes_change(prior_revision, revision, arm, digests):
    """Return the .gitattributes lines the Hub added; they may only route this arm's new files to LFS."""
    from huggingface_hub import hf_hub_download
    before = Path(hf_hub_download(REPO, '.gitattributes', revision=prior_revision)).read_text()
    after = Path(hf_hub_download(REPO, '.gitattributes', revision=revision)).read_text()
    assert after.startswith(before), '.gitattributes rewritten'
    added = [line for line in after[len(before):].splitlines() if line.strip()]
    for line in added:
        path = line.split()[0]
        assert path in digests and path.startswith(arm + '/') and 'filter=lfs' in line, line
    return added


def publish(arm, files, digests, root):
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi()
    assert api.whoami()['name'].lower() == 'anhdao69'
    receipt = root / 'PUBLISHED.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert saved['digests'] == digests
        revision = saved['revision']
    else:
        info = api.model_info(REPO, files_metadata=True)
        assert not any(f.rfilename.startswith(arm + '/') for f in info.siblings), 'destination exists; inspect first'
        write(root / 'prior_repo_files.json', dict(revision=info.sha, files={
            f.rfilename: dict(blob_id=f.blob_id, size=f.size) for f in info.siblings}))
        commit = api.create_commit(
            repo_id=REPO, parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n, p in sorted(files.items())],
            commit_message=f'Publish {arm}: S8N R/B ablation, completed epoch (step 3852), verified serving weights')
        revision = commit.oid
        write(receipt, dict(repo=REPO, revision=revision, digests=digests))
        print('PUBLISHED', revision, flush=True)
    info = api.model_info(REPO, revision=revision, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    prior = json.loads((root / 'prior_repo_files.json').read_text())
    attributes = []
    for name, old in prior['files'].items():
        if name == '.gitattributes' and remote[name].blob_id != old['blob_id']:
            attributes = attributes_change(prior['revision'], revision, arm, digests)
        else:
            assert remote[name].blob_id == old['blob_id'] and remote[name].size == old['size'], name
    assert set(remote) == set(prior['files']) | set(digests), 'unexpected files in the commit'
    verified = []
    for name, expected in sorted(digests.items()):
        f = remote[name]
        assert f.size == expected['size'], name
        if f.lfs:
            assert f.lfs.sha256 == expected['sha256'], name
            method = 'Hub LFS SHA256 and size'
        else:
            assert sha(hf_hub_download(REPO, name, revision=revision)) == expected['sha256'], name
            method = 'Downloaded SHA256 and size'
        verified.append(dict(path=name, **expected, method=method))
    write(root / 'UPLOAD_VERIFIED.json', dict(
        passed=True, repo=REPO, revision=revision, folder=arm, files=verified,
        prior_files_unchanged=len(prior['files']) - int(bool(attributes)), automatic_metadata_change=attributes,
        verified_utc=now()))
    print('UPLOAD_VERIFIED', revision, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', nargs='?', choices=tuple(ARMS))
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--create-repo', action='store_true')
    args = parser.parse_args()
    if args.create_repo:
        return create_repo()
    root = OPS / args.arm
    root.mkdir(parents=True, exist_ok=True)
    campaign, run, identity, checkpoint, record, resources, checks, navigation, declared = validate_training(args.arm)
    print('COMPLETED_ARM_VERIFIED', args.arm, flush=True)
    export = stage(checkpoint, navigation, root)
    files, digests = collect(campaign, args.arm, run, identity, checkpoint, record, resources, checks, declared,
                             root, export)
    if args.check_only:
        print('CHECK_ONLY_PASSED', args.arm, len(files), flush=True)
        return
    publish(args.arm, files, digests, root)


if __name__ == '__main__':
    main()
