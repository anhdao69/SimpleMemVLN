import importlib.util
import json
from pathlib import Path
import pytest


@pytest.mark.parametrize('world_size', [1, 2, 4])
def test_epoch_report_uses_actual_schedule(tmp_path, world_size):
    from types import SimpleNamespace
    from qwen_vl.train.campaign import CampaignReports
    for rank in range(world_size):
        (tmp_path / f'profile_rank{rank}.jsonl').write_text('\n'.join(json.dumps(dict(
            update=i, seconds=1, peak_reserved_gib=50, peak_allocated_gib=40,
            unix_start=i, unix_end=i+1)) for i in range(1, 7))+'\n')
        (tmp_path / f'exposures_rank{rank}.jsonl').write_text('\n'.join(json.dumps(dict(
            update=i-1, actions=1, loss_sum=2)) for i in range(1, 7))+'\n')
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0, num_train_epochs=2, world_size=world_size)
    state = SimpleNamespace(epoch=2.0, global_step=6, max_steps=6)
    CampaignReports().on_save(args, state, None)
    report = json.loads((tmp_path/'reports/epoch-2.json').read_text())
    assert report['completed_updates'] == 3
    assert report['start_update'] == 4
    assert report['end_update'] == 6


def test_report_uses_global_action_loss_and_slowest_rank_time(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts/vln/report_campaign.py"
    assert script.exists(), "Campaign reporting is missing"
    spec = importlib.util.spec_from_file_location("reporting", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for rank in range(4):
        (tmp_path / f"profile_rank{rank}.jsonl").write_text(
            json.dumps(
                {
                    "update": 1,
                    "seconds": rank + 1,
                    "peak_reserved_gib": 50 + rank,
                    "peak_allocated_gib": 40 + rank,
                    "unix_start": 0,
                    "unix_end": 4,
                }
            )
            + "\n"
        )
        (tmp_path / f"exposures_rank{rank}.jsonl").write_text(
            json.dumps({"update": 0, "actions": rank + 1, "loss_sum": 2 * (rank + 1)})
            + "\n"
        )
    report = module.build_report(tmp_path, 1, 1, world_size=4)
    assert report["loss_action_weighted"] == 2
    assert report["mean_update_compute_seconds"] == 4
    assert report["peak_reserved_gib"] == 53
    assert report["gpu_utilization_mean_pct"] is None


@pytest.mark.parametrize('world_size', [1, 2, 4])
@pytest.mark.parametrize('fault', [None, 'missing_rank', 'duplicate_update', 'missing_exposure'])
def test_first50_callback_validates_actual_world_size(tmp_path, world_size, fault):
    from types import SimpleNamespace
    from qwen_vl.train.campaign import CampaignReports
    for rank in range(world_size):
        rows = [dict(update=i, seconds=rank+1, peak_reserved_gib=50,
                     peak_allocated_gib=40, unix_start=i*10, unix_end=i*10+rank+1)
                for i in range(1, 51)]
        if fault == 'missing_rank' and rank == world_size-1:
            continue
        if fault == 'duplicate_update' and rank == 0:
            rows.append(rows[-1])
        (tmp_path/f'profile_rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        exposures = [dict(update=i, actions=2, loss_sum=3) for i in range(50)]
        if fault == 'missing_exposure' and rank == world_size-1:
            exposures.pop()
        (tmp_path/f'exposures_rank{rank}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in exposures))
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0, world_size=world_size)
    callback = CampaignReports()
    if fault:
        with pytest.raises(ValueError):
            callback.on_step_end(args, SimpleNamespace(global_step=50), None)
        assert not (tmp_path/'reports/first-50-updates.json').exists()
    else:
        callback.on_step_end(args, SimpleNamespace(global_step=50), None)
        report = json.loads((tmp_path/'reports/first-50-updates.json').read_text())
        assert report['completed_updates'] == 50
        assert report['world_size'] == world_size
        assert report['loss_action_weighted'] == 1.5
        assert report['mean_update_compute_seconds'] == world_size
