"""CPU-only publication of one completed S arm to the Hub folder ``<ARM>/`` with an integrity audit.

Adapted from the M5_E1 and seed-431 publishers. The lane controller must have declared the arm complete; the
bound run identity is re-derived from the sealed source; all-rank coverage, finite losses and the M5_E1 exposure
order are checked; serving files are copied from checkpoint-3852 (bitwise checked) and the real navigation model
strictly reloads on CPU. One Hub commit adds only ``<ARM>/`` on an optimistic parent revision and never
overwrites; every file is then verified by SHA-256 and size at the immutable revision.

Runs from an operations directory outside the sealed source (which must stay clean):
    publish_arm.py ARM [--check-only]       validate, stage and hash; upload unless --check-only
    publish_arm.py --create-repo            one-time: create the public repository with its README
"""
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ.pop(key, None)

BASE = Path('/mnt/data/vmo-ai-task/anhdh35')
CAMPAIGN = BASE / 'm5-gdn-s-sweep-dabf4ad-20261020/campaign.json'
M5_E1_RUN = BASE / 'm5-one-epoch-worker2-20261018/production/m5_joint_e1'
OPS = Path(__file__).resolve().parent
REPO = 'anhdao69/M5_S'
SOURCE_COMMIT = 'dabf4ad085d20512640fe0ba3bda72ad4724dc76'
M5_E1_LOSS = 0.12657747317027537           # action-weighted epoch loss from the M5_E1 publication
ARMS = ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8N', 'S8W', 'S9')
DESCRIPTION = {
    'S1': 'mean over heads of the unscaled delta-rule residual ‖e‖, e = v − S̄ᵀk (token)',
    'S2': 'mean over heads of the gated value norm β‖v‖, no state scan (token)',
    'S3': 'mean over heads of the raw value norm ‖v‖, no state scan (token)',
    'S4': 'minimum over heads of the HAM-local cosine error between v and S̄ᵀk (token)',
    'S5': 'mean over heads of the HAM-local cosine error between v and S̄ᵀk (token)',
    'S6': 'mean of β‖e‖ over 8 frozen slow heads per layer, calibrated on base Qwen (token)',
    'S7': 'arithmetic mean of the 300 M5 (S0) token scores of a frame; whole frames kept (frame)',
    'S8N': 'net state drift ‖S_after − S_before‖_F / max(‖S_before‖_F, ε_l), ε_l calibrated (frame)',
    'S8W': 'decay-transported write ‖W‖_F / max(‖S_before‖_F, ε_l), ε_l calibrated (frame)',
    'S9': 'mean over heads of the gate complement residual (1 − β)‖e‖ (token)',
}
SERVING = {'pytorch_model.bin', 'navigation.json', 'config.json', 'generation_config.json',
           'preprocessor_config.json', 'processor_config.json', 'tokenizer_config.json', 'tokenizer.json',
           'chat_template.jinja'}
REQUIRED = ('pytorch_model.bin', 'navigation.json', 'config.json', 'tokenizer.json', 'processor_config.json')

campaign_data = json.loads(CAMPAIGN.read_text())
SOURCE = Path(campaign_data['source'])
sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE)]
from scripts.memory_sweep.s_campaign import (  # noqa: E402
    EXPECTED_STEPS, WORLD, Campaign, latest_resources, run_complete, sha, write)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def declared_complete(arm):
    """The lane controller's own ``arm-complete`` record, or None."""
    for history in sorted((Path(campaign_data['root']) / 'lanes').glob('*/history.jsonl')):
        for line in history.read_text().splitlines():
            row = json.loads(line)
            if row.get('status') == 'arm-complete' and row.get('arm') == arm:
                return row
    return None


def exposure_plan(run):
    """rank -> update -> ordered episode uids; rows replayed after a resume collapse onto the originals."""
    plan = {}
    for rank in range(WORLD):
        rows = {}
        for line in (Path(run) / f'train/exposures_rank{rank}.jsonl').read_text().splitlines():
            row = json.loads(line)
            rows.setdefault(row['update'], {})[row['episode_uid']] = None
        plan[rank] = {update: list(uids) for update, uids in rows.items()}
    return plan


def validate_training(arm):
    from qwen_vl.train.recovery import validate_complete
    from scripts.vln.profile_episodic import verify
    campaign = Campaign(CAMPAIGN)
    assert campaign.data['source_commit'] == SOURCE_COMMIT
    run = campaign.run_dir(arm)
    identity = json.loads((run / 'run_identity.json').read_text())
    assert identity == campaign.run_identity(arm), 'bound identity differs from the sealed source'
    seal = json.loads((campaign.shared / 'QUALIFIED.json').read_text())
    assert seal['passed'] and seal['source_commit'] == SOURCE_COMMIT
    config = identity['config']
    assert config['memory']['score_id'] == arm and 'research' not in config
    assert config['training']['seed'] == 429 and config['training']['epochs'] == 1
    assert config['training']['expected_total_steps'] == EXPECTED_STEPS
    assert declared_complete(arm), f'{arm}: lane controller has not declared the arm complete'
    assert run_complete(run, config)
    resources = latest_resources(run)
    assert resources['status'] == 'PASS' and resources['new_oom_events'] == resources['new_oom_kill_events'] == 0
    checks = verify(run / 'train', WORLD, EXPECTED_STEPS, 8, 78.0)
    assert checks['passed'], checks['problems']
    for key in ('losses', 'grad_norms'):
        checks.pop(key, None)
    exposures = exposure_plan(run)
    assert exposures == exposure_plan(M5_E1_RUN), f'{arm}: training exposures differ from M5_E1'
    checkpoint = run / f'train/checkpoint-{EXPECTED_STEPS}'
    record = validate_complete(checkpoint)
    assert record['step'] == EXPECTED_STEPS and record['world_size'] == WORLD and record['epoch_boundary']
    trainer = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert trainer['global_step'] == trainer['max_steps'] == EXPECTED_STEPS and trainer['epoch'] == 1
    navigation = json.loads((checkpoint / 'navigation.json').read_text())
    assert navigation['config'] == config
    checks['exposures_identical_to_m5_e1'] = sum(len(u) for r in exposures.values() for u in r.values())
    return campaign, run, identity, checkpoint, record, resources, checks, navigation


def stage(arm, checkpoint, navigation, root):
    """Exact serving copy, bitwise tensor check and strict CPU reload of the real navigation model."""
    import torch
    from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    from qwen_vl.data.episode_serializer import EpisodeSerializer
    from qwen_vl.models.nav_model import SimpleMemVLNForNavigation
    export = root / 'model'
    export.mkdir(parents=True, exist_ok=True)
    for path in checkpoint.iterdir():
        if path.is_file() and path.name in SERVING:
            destination = export / path.name
            if not destination.exists() or destination.stat().st_size != path.stat().st_size:
                partial = export / (path.name + '.partial')
                shutil.copy2(path, partial)
                partial.replace(destination)
    assert all((export / name).exists() for name in REQUIRED)
    assert not torch.cuda.is_available()
    metadata = json.loads((export / 'navigation.json').read_text())
    assert metadata == navigation
    before = torch.load(checkpoint / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    after = torch.load(export / 'pytorch_model.bin', map_location='cpu', weights_only=True, mmap=True)
    assert not any(k.startswith('research_aux.') for k in before)
    assert all(torch.isfinite(t).all().item() for t in before.values() if t.is_floating_point()), 'nonfinite tensor'
    assert set(after) == set(before) and all(torch.equal(after[k], before[k]) for k in before)
    print('ALL_SERVING_TENSORS_BITWISE_EQUAL', len(before), flush=True)
    # Real architecture on meta, then strict assignment of the mmap-backed CPU tensors (no second model copy).
    base_config = AutoConfig.from_pretrained(campaign_data['model'], local_files_only=True)
    with torch.device('meta'):
        backbone = Qwen3_5ForConditionalGeneration(base_config)
        model = SimpleMemVLNForNavigation(backbone, metadata['config'])
    model.load_state_dict(after, strict=True, assign=True)
    assert not hasattr(model, 'research_aux')
    assert all(p.device.type == 'cpu' for p in model.parameters())
    processor = AutoProcessor.from_pretrained(export, local_files_only=True)
    assert EpisodeSerializer(processor, metadata['config']).metadata() == metadata
    audit = dict(passed=True, cpu_only=True, strict_architecture_reload=True, exact_serving_tensors=len(before),
                 removed_training_tensors=[], serializer_metadata_exact=True, memory_metadata_exact=True,
                 all_model_tensors_finite=True, completed_model_gpu_replay=False)
    write(root / 'export_validation.json', audit)
    print('CPU_STRICT_EXPORT_VALIDATED', flush=True)
    return export


def readme(arm, run, identity, resources):
    epoch = json.loads((run / 'train/reports/epoch-1.json').read_text())
    memory = identity['config']['memory']
    hours = resources['seconds'] / 3600
    return f"""# {arm} — M5 GDN-only S sweep, completed epoch 1

Memory priority: **{DESCRIPTION[arm]}**.
Score `{memory['score_id']}`, unit `{memory['selection_unit']}`, head reduction `{memory['head_reduction']}`,
backend `{memory['scorer_backend']}`. Everything else is M5 (`episodic_window_v1` at R{memory['recent_groups_including_current']}/B{memory['episodic_budget_tokens']}):
{memory['recent_groups_including_current']} recent observation groups including the current one plus a
{memory['episodic_budget_tokens']}-token episodic budget, native K/V rows kept at their original positions,
canonical text action history and full-episode BPTT.

This folder is the arm's serving model; `provenance/` holds the evidence and `SHA256SUMS.json` the published hashes.

## Training (matched to M5_E1)

The recipe is M5_E1's apart from the memory priority and two loader-only settings (asserted before launch):
original Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, joint R2R + English RxR15deg
(30,815 episodes), one epoch of 3,852 optimizer updates, global batch 8 (4 H100 x 1 episode x accumulation 2),
seed 429, backbone LR 5e-6 with 116 warmup updates and cosine decay to 0.1, Adam 0.9/0.95, weight decay 0.01,
clip 1, ZeRO-2 with CPU Adam. Every rank saw exactly M5_E1's episodes in M5_E1's order.

| | {arm} | M5_E1 |
|---|---:|---:|
| Action-weighted epoch loss | {epoch['loss_action_weighted']:.5f} | {M5_E1_LOSS:.5f} |
| Final update loss | {epoch['last_update_loss']:.5f} | |

Training loss is not navigation success. Wall time {hours:.2f} h including load and saves; no GPU or host OOM.

## Verification

The lane controller declared the run complete; the bound identity re-derives from the sealed source; all
3,852 updates and exposures are covered on every rank with finite losses; checkpoint-3852 is a complete
epoch-boundary recovery checkpoint. Serving tensors are bitwise identical to it and the real navigation model
strictly reloads on CPU with exact serializer and memory metadata. No completed-model GPU replay is claimed.

## Use

Code: SimpleMemVLN branch `m5_gdn_s_sweep`, source `{SOURCE_COMMIT}` (memory mode
`episodic_window_gdn_sweep_v1`). Load with
`qwen_vl.train.vln_runtime.load_checkpoint('{arm}', base_model_path)` after downloading this folder.
These are navigation-wrapper weights for inference; optimizer/RNG state stays on the cluster.
"""


def collect(campaign, arm, run, identity, checkpoint, record, resources, checks, root, export):
    write(root / 'training_validation.json', checks)
    provenance = root / 'provenance'
    provenance.mkdir(exist_ok=True)
    supervisor = sorted(run.glob('supervisor-*'))[-1]
    for source, name in [(run / 'train/reports/epoch-1.json', 'epoch-1.json'),
                         (run / 'train/resolved_config.json', 'resolved_config.json'),
                         (run / 'train/run_timing.json', 'run_timing.json'),
                         (checkpoint / 'trainer_state.json', 'trainer_state.json'),
                         (supervisor / 'resources.json', 'completion_resources.json'),
                         (root / 'export_validation.json', 'publication_validation.json'),
                         (root / 'training_validation.json', 'training_validation.json'),
                         (campaign.shared / 'QUALIFIED.json', 'qualification_pass.json'),
                         (run / 'run_identity.json', 'run_identity.json'),
                         (campaign.arm_memory_path(arm), f'arm_{arm}.yaml'),
                         (campaign.configs / 'runtime.yaml', 'runtime.yaml')]:
        shutil.copy2(source, provenance / name)
    if not (provenance / 'source_and_completion.json').exists():
        write(provenance / 'source_and_completion.json', dict(
            source_commit=SOURCE_COMMIT, source_directory=str(SOURCE), checkpoint=str(checkpoint), arm=arm,
            completed_epoch=1, optimizer_updates=EXPECTED_STEPS, global_batch=8, recovery_inventory=record,
            controller_completion=declared_complete(arm),
            export_source_weights_sha256=sha(checkpoint / 'pytorch_model.bin'),
            export_weights_sha256=sha(export / 'pytorch_model.bin'), verified_utc=now()))
    (root / 'README.md').write_text(readme(arm, run, identity, resources))
    files = {f'{arm}/{p.name}': p for p in export.iterdir() if p.is_file() and p.name in SERVING}
    files.update({f'{arm}/provenance/{p.name}': p for p in provenance.iterdir() if p.is_file()})
    files[f'{arm}/README.md'] = root / 'README.md'
    digests = {name: dict(sha256=sha(path), size=path.stat().st_size) for name, path in files.items()}
    write(root / 'SHA256SUMS.json', digests)
    files[f'{arm}/SHA256SUMS.json'] = root / 'SHA256SUMS.json'
    digests[f'{arm}/SHA256SUMS.json'] = dict(sha256=sha(root / 'SHA256SUMS.json'),
                                             size=(root / 'SHA256SUMS.json').stat().st_size)
    print('HASHED', len(files), flush=True)
    return files, digests


def attributes_change(api, prior_revision, revision, arm, digests):
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
        assert not info.private
        assert not any(f.rfilename.startswith(arm + '/') for f in info.siblings), 'destination exists; inspect first'
        write(root / 'prior_repo_files.json', dict(revision=info.sha, files={
            f.rfilename: dict(blob_id=f.blob_id, size=f.size) for f in info.siblings}))
        commit = api.create_commit(
            repo_id=REPO, parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n, p in sorted(files.items())],
            commit_message=f'Publish {arm}: M5 GDN S sweep, completed epoch 1 (step 3852), verified serving weights')
        revision = commit.oid
        write(receipt, dict(repo=REPO, revision=revision, digests=digests))
        print('PUBLISHED', revision, flush=True)
    info = api.model_info(REPO, revision=revision, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    prior = json.loads((root / 'prior_repo_files.json').read_text())
    attributes = []
    for name, old in prior['files'].items():
        if name == '.gitattributes' and remote[name].blob_id != old['blob_id']:
            attributes = attributes_change(api, prior['revision'], revision, arm, digests)
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


ROOT_README = """# M5_S — GDN-only memory-priority sweep on M5 (SimpleMemVLN)

Each top-level folder is one completed navigation model (`S1/`, `S2/`, …). The arms change only how M5's
episodic memory ranks old visual K/V rows for its fixed R4/B1200 budget; the priority comes from the
Gated DeltaNet layer feeding each full-attention layer. Everything else matches the M5_E1 reference
(`anhdao69/Streaming_VGGT`, folder `M5_E1`), which is the S0 arm and is not repeated here.

Each folder holds the serving files, `README.md` (arm definition, training summary, loading),
`provenance/` (identity, qualification, timing and validation evidence) and `SHA256SUMS.json`.
Folders are added one at a time as training finishes; earlier folders are never modified.

Code: SimpleMemVLN branch `m5_gdn_s_sweep`, source `dabf4ad085d20512640fe0ba3bda72ad4724dc76`.
Training loss is reported per arm; navigation success has not been evaluated in these folders.
"""


def create_repo():
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    assert api.whoami()['name'].lower() == 'anhdao69'
    assert not api.repo_exists(REPO), f'{REPO} already exists'
    api.create_repo(REPO, repo_type='model', private=False, exist_ok=False)
    info = api.model_info(REPO)
    path = OPS / 'ROOT_README.md'
    path.write_text(ROOT_README)
    commit = api.create_commit(repo_id=REPO, parent_commit=info.sha,
                               operations=[CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=str(path))],
                               commit_message='Describe the M5 GDN S sweep repository layout')
    write(OPS / 'REPO_CREATED.json', dict(repo=REPO, private=False, revision=commit.oid, utc=now()))
    print('REPO_CREATED', REPO, commit.oid, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', nargs='?', choices=ARMS)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--create-repo', action='store_true')
    args = parser.parse_args()
    if args.create_repo:
        return create_repo()
    root = OPS / args.arm
    root.mkdir(exist_ok=True)
    campaign, run, identity, checkpoint, record, resources, checks, navigation = validate_training(args.arm)
    print('COMPLETED_ARM_VERIFIED', args.arm, flush=True)
    export = stage(args.arm, checkpoint, navigation, root)
    files, digests = collect(campaign, args.arm, run, identity, checkpoint, record, resources, checks, root, export)
    if args.check_only:
        print('CHECK_ONLY_PASSED', args.arm, len(files), flush=True)
        return
    publish(args.arm, files, digests, root)


if __name__ == '__main__':
    main()
