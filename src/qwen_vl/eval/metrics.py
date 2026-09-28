"""Failure-adjusted metrics retain the entire scheduled denominator."""
import math
from collections import Counter


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def summarize(records):
    if not records:
        raise ValueError("No evaluated episodes")
    count = len(records)
    result = dict(
        episodes=count,
        failures=sum(bool(r["failed"]) for r in records),
        sr=sum(0 if r["failed"] else r["metrics"].get("success", 0) for r in records)
        / count,
        spl=sum(0 if r["failed"] else r["metrics"].get("spl", 0) for r in records)
        / count,
    )
    for name, source in [
        ("navigation_error", "distance_to_goal"),
        ("trajectory_length", "trajectory_length"),
        ("oracle_success", "oracle_success"),
        ("ndtw", "ndtw"),
    ]:
        values = [
            float(r["metrics"][source])
            for r in records
            if source in r["metrics"] and math.isfinite(float(r["metrics"][source]))
        ]
        result[name] = sum(values) / len(values) if values else None
        result[name + "_coverage"] = len(values)
        result[name + "_undefined"] = count - len(values)
    result["forced_stops"] = sum(bool(r.get("forced_stop")) for r in records)
    actions = [a for r in records for a in r.get("actions", [])]
    result["predicted_action_counts"] = dict(
        Counter(str(a["predicted"]) for a in actions)
    )
    result["executed_action_counts"] = dict(
        Counter(str(a["executed"]) for a in actions)
    )
    result["predicted_stops"] = sum(a["predicted"] == 0 for a in actions)
    result["failure_reasons"] = dict(
        Counter(r.get("failure_reason", "unspecified") for r in records if r["failed"])
    )
    for key in ["model_seconds", "transport_and_model_seconds", "environment_seconds"]:
        values = [float(a[key]) for a in actions if key in a]
        prefix = (
            "model_latency" if key == "model_seconds" else key.removesuffix("_seconds")
        )
        result[prefix + "_p50_seconds"] = quantile(values, 0.5)
        result[prefix + "_p95_seconds"] = quantile(values, 0.95)
    result["peak_kv_tokens"] = max(
        (a.get("retained_kv_tokens", 0) for a in actions), default=0
    )
    result["peak_allocated_gib"] = max(
        (a.get("peak_allocated_gib", 0) for a in actions), default=0
    )
    result[
        "environment_timing_note"
    ] = "Habitat env.step includes simulation and rendering; not separated by this adapter"
    return result
