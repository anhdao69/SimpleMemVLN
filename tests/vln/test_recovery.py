import json
from types import SimpleNamespace
import pytest
import torch


def test_episode_loader_keeps_accumulation_inputs_on_cpu(tmp_path):
    from qwen_vl.train.train_episode import EpisodeTrainer
    from transformers import TrainingArguments
    trainer = EpisodeTrainer(
        model=torch.nn.Linear(1, 1), serializer=None, data_contract={},
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True,
            per_device_train_batch_size=1, gradient_accumulation_steps=2,
            remove_unused_columns=False, report_to='none'),
        train_dataset=[{'pixel_values': torch.ones(2, 2), 'num_actions': 2}] * 4,
        data_collator=lambda rows: rows[0],
    )
    loader = trainer.get_train_dataloader()
    assert loader.device is None  # Accelerate must not stage either GAS slot.
    batches, actions = trainer.get_batch_samples(iter(loader), 2, torch.device('cpu'))
    assert actions == 4 and len(batches) == 2
    assert all(b['pixel_values'].device.type == 'cpu' for b in batches)


def checkpoint_fixture(root, step, epoch):
    p = root / f'checkpoint-{step}'
    p.mkdir()
    (p/'pytorch_model.bin').write_bytes(b'weights')
    (p/'trainer_state.json').write_text(json.dumps(dict(global_step=step, epoch=epoch)))
    (p/'resume_contract.json').write_text('{}')
    (p/'navigation.json').write_text('{}')
    (p/'rng_state.pth').write_bytes(b'rng')
    (p/'optimizer.pt').write_bytes(b'optimizer')
    (p/'scheduler.pt').write_bytes(b'scheduler')
    return p


def test_recovery_ignores_incomplete_and_detects_truncated_files(tmp_path):
    from qwen_vl.train.recovery import mark_complete, latest_checkpoint
    first = checkpoint_fixture(tmp_path, 100, .25)
    mark_complete(first, world_size=1, epoch_boundary=False)
    second = checkpoint_fixture(tmp_path, 200, .5)
    assert latest_checkpoint(tmp_path) == first
    mark_complete(second, world_size=1, epoch_boundary=False)
    assert latest_checkpoint(tmp_path) == second
    (second/'optimizer.pt').write_bytes(b'')
    assert latest_checkpoint(tmp_path) == first


def test_retention_keeps_epochs_and_two_latest_recovery_points(tmp_path):
    from qwen_vl.train.recovery import mark_complete, prune_recovery
    for step, epoch in [(100,.25),(200,.5),(300,.75),(400,1),(500,1.25)]:
        mark_complete(checkpoint_fixture(tmp_path,step,epoch), world_size=1,
                      epoch_boundary=epoch==1)
    unknown = checkpoint_fixture(tmp_path,600,1.5)
    prune_recovery(tmp_path, keep=2)
    assert {p.name for p in tmp_path.iterdir()} == {
        'checkpoint-300','checkpoint-400','checkpoint-500','checkpoint-600'}
    assert unknown.exists()


def test_recovery_requires_optimizer_and_all_rank_rng(tmp_path):
    from qwen_vl.train.recovery import mark_complete
    p = checkpoint_fixture(tmp_path,100,.25)
    (p/'optimizer.pt').unlink()
    with pytest.raises(ValueError):
        mark_complete(p, world_size=1, epoch_boundary=False)


def test_periodic_save_does_not_trigger_epoch_report():
    from qwen_vl.train.campaign import CampaignReports
    CampaignReports().on_save(SimpleNamespace(num_train_epochs=2),
        SimpleNamespace(epoch=100/3852,global_step=100,max_steps=7704),None)


def test_resume_rewinds_metrics_but_preserves_original_history(tmp_path):
    from qwen_vl.train.recovery import rewind_reports
    profile=tmp_path/'profile_rank0.jsonl'
    exposure=tmp_path/'exposures_rank0.jsonl'
    profile.write_text('\n'.join(json.dumps({'update':n}) for n in [99,100,101])+'\n')
    exposure.write_text('\n'.join(json.dumps({'update':n}) for n in [99,100])+'\n')
    rewind_reports(tmp_path,100)
    assert [json.loads(s)['update'] for s in profile.read_text().splitlines()]==[99,100]
    assert [json.loads(s)['update'] for s in exposure.read_text().splitlines()]==[99]
    archived=list((tmp_path/'recovery-history').glob('*/profile_rank0.jsonl'))
    assert len(archived)==1 and len(archived[0].read_text().splitlines())==3
