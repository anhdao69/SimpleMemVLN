def test_closed_loop_summary_does_not_invent_supervised_recall():
    from qwen_vl.eval.metrics import summarize

    records = [
        dict(
            failed=False,
            metrics={"success": 1, "spl": 0.8},
            forced_stop=False,
            actions=[
                dict(predicted=1, executed=1, model_seconds=0.1),
                dict(predicted=0, executed=0, model_seconds=0.2),
            ],
        ),
        dict(
            failed=True,
            failure_reason="ValueError: Invalid navigation response: 'MOVE_RIGHT'",
            metrics={},
            actions=[],
            forced_stop=False,
        ),
    ]
    r = summarize(records)
    assert r["mean_episode_steps"] == 1
    assert r["invalid_responses"] == 1
    assert abs(r["model_latency_mean_seconds"] - 0.15) < 1e-10
    assert r["predicted_stops"] == 1
    assert "stop_recall" not in r
