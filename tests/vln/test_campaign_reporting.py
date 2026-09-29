import importlib.util
import json
from pathlib import Path


def test_epoch_report_uses_actual_schedule(tmp_path):
    from types import SimpleNamespace
    from qwen_vl.train.campaign import CampaignReports
    for rank in range(4):
        (tmp_path / f'profile_rank{rank}.jsonl').write_text('\n'.join(json.dumps(dict(
            update=i, seconds=1, peak_reserved_gib=50, peak_allocated_gib=40,
            unix_start=i, unix_end=i+1)) for i in range(1, 7))+'\n')
        (tmp_path / f'exposures_rank{rank}.jsonl').write_text('\n'.join(json.dumps(dict(
            update=i-1, actions=1, loss_sum=2)) for i in range(1, 7))+'\n')
    args = SimpleNamespace(output_dir=str(tmp_path), process_index=0, num_train_epochs=2)
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
    report = module.build_report(tmp_path, 1, 1)
    assert report["loss_action_weighted"] == 2
    assert report["mean_update_compute_seconds"] == 4
    assert report["peak_reserved_gib"] == 53
    assert report["gpu_utilization_mean_pct"] is None
