import json
from types import SimpleNamespace
import torch


def test_manifest_counts_match_selected_episodes(tmp_path):
    from qwen_vl.models.action_loss import resolve_class_balance
    from test_candidate_contract import config

    path = tmp_path / "m.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                dict(episode_uid=str(i), steps=[dict(action_name=a) for a in actions])
            )
            for i, actions in enumerate(
                [["MOVE_FORWARD", "STOP"], ["TURN_LEFT", "TURN_RIGHT", "STOP"]]
            )
        )
    )
    cfg = config()
    resolve_class_balance(cfg, path, 1, "shortest")
    assert cfg["training"]["action_class_counts"] == [1, 0, 0, 1]
    assert cfg["training"]["action_class_weights"] == [1.0, 1.0, 1.0, 1.0]


def test_callback_accumulates_microsteps_and_resets(tmp_path):
    from qwen_vl.train.action_reporting import ActionMetricsCallback

    c = ActionMetricsCallback(tmp_path)
    c.add(
        torch.tensor([0, 3]), torch.tensor([0, 0]), torch.tensor(4.0), torch.tensor(3.0)
    )
    c.add(torch.tensor([1]), torch.tensor([1]), torch.tensor(2.0), torch.tensor(1.0))
    c.on_step_end(
        SimpleNamespace(process_index=0), SimpleNamespace(global_step=7), None
    )
    report = json.loads((tmp_path / "action_metrics.jsonl").read_text())
    assert report["weighted_ce"] == 2 and report["unweighted_ce"] == 4 / 3
    assert report["per_class"]["STOP"]["recall"] == 0
    assert c.counts is None


def test_recovery_rewinds_new_action_metrics(tmp_path):
    from qwen_vl.train.recovery import rewind_reports

    p = tmp_path / "action_metrics.jsonl"
    p.write_text('{"update": 100}\n{"update": 101}\n')
    rewind_reports(tmp_path, 100)
    assert p.read_text() == '{"update": 100}\n'


def test_component_profile_opt_in(monkeypatch):
    from test_session import make_session

    s = make_session(monkeypatch, [])
    s.serializer.mode = "candidate_logits"
    s.serializer.feedback_ids = lambda c: [32, 9, 8]
    s.model.action_logits = lambda h: torch.tensor([1.0, 0.0, 0.0, 0.0])
    s.cfg["runtime"]["profile_components"] = True
    r = s.observe("e", 0, None)
    assert set(r["component_seconds"]) >= {
        "image_preprocessing",
        "action_logit_projection",
        "action_history_append",
    }
    assert all(v >= 0 for v in r["component_seconds"].values())
    assert s.model._stage_timer is None
