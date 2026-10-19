"""Recompute completed local reports without changing evaluation artifacts."""
import hashlib
import json
import math
from pathlib import Path
import runpy
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
RUNS = {
    "G3": "g3_epoch1_20261018",
    "M5 (older two-epoch schedule)": "m5_epoch1_20261018",
    "G3_M5 (L24)": "g3_m5_epoch1_20261017",
    "G3_M5_L8": "g3_m5_l8_epoch1_20261018",
    "G3_M5_L8_L16_L24": "g3_m5_l8_l16_l24_epoch1_20261018",
    "M5_E1": "m5_e1_epoch1_20261019",
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def journal(path):
    with path.open("rb") as stream:
        result = []
        for line in stream:
            assert line.endswith(b"\n"), path
            result.append(json.loads(line))
    assert not any(row.get("fatal") for row in result), path
    return result


def compare(computed, saved):
    for key, value in computed.items():
        other = saved[key]
        assert (math.isclose(value, other, rel_tol=1e-12, abs_tol=1e-12)
                if isinstance(value, float) else value == other), (key, value, other)


def main():
    evidence = {"verified_utc": datetime.now(timezone.utc).isoformat(),
                "metric_tolerance": 1e-12, "runs": {}, "comparisons": []}
    paired = {}
    common_protocol = None
    for label, dirname in RUNS.items():
        root = REPO / "evaluation" / dirname
        out = root / ("full_unqualified" if label == "G3_M5_L8" else "full")
        summarize = runpy.run_path(str(root / "source/src/qwen_vl/eval/metrics.py"))["summarize"]
        wanted = read(root / "episode_ids.json")
        assert len(wanted) == len(set(wanted)) == 1839
        rows = journal(out / "episodes.jsonl")
        assert [r["episode_id"] for r in rows] == wanted
        summary = read(out / "summary.json")
        assert summary["complete"] and summary["scheduled_episodes"] == 1839
        compare(summarize(rows), summary)
        contract = read(out / "evaluation_contract.json")
        runtime = read(out / "runtime.json")
        assert contract["episode_ids"] == wanted
        protocol_keys = ("dataset_sha256", "habitat_config_sha256", "simulator_version",
                         "split", "seed", "scenes_dir", "protocol", "max_steps", "failure_policy",
                         "base_assets_sha256")
        protocol = {k: contract[k] for k in protocol_keys}
        if common_protocol is None:
            common_protocol = protocol
        assert protocol == common_protocol
        contracts, shard_rows, counts = [], [], []
        for shard in sorted(out.glob("shard-*")):
            panel = read(root / ("panel-" + shard.name.split("-")[1] + ".json"))
            sr = journal(shard / "episodes.jsonl")
            assert [r["episode_id"] for r in sr] == panel
            saved = read(shard / "summary.json")
            assert saved["complete"] and saved["scheduled_episodes"] == len(panel)
            compare(summarize(sr), saved)
            c = read(shard / "evaluation_contract.json")
            assert c.pop("episode_ids") == panel
            contracts.append(c)
            assert read(shard / "runtime.json") == runtime
            counts.append(len(sr))
            shard_rows.extend(sr)
        assert shard_rows == rows
        assert all(c == contracts[0] for c in contracts)
        for key, value in contracts[0].items():
            assert contract[key] == value
        for relative, sha in contract["source_sha256"].items():
            assert digest(root / "source/src/qwen_vl" / relative) == sha
        status = read(root / "status.json")
        validation = read(out / "validation.json")
        assert status["status"] == "complete" and validation["passed"]
        assert all(w["exit_code"] == 0 for w in status["workers"])
        audit = read(root / "serving_audit.json")
        assert audit["passed"] == (label != "G3_M5_L8")
        if label == "G3_M5_L8":
            assert summary["serving_qualification_passed"] is False
        success = sum(0 if r["failed"] else r["metrics"].get("success", 0) for r in rows)
        assert success == int(success)
        paired[label] = {r["episode_id"]: {
            "identity": [r["episode_uid"], r["scene_id"], r["instruction"]],
            "success": 0 if r["failed"] else r["metrics"].get("success", 0),
            "spl": 0 if r["failed"] else r["metrics"].get("spl", 0),
        } for r in rows}
        source = read(root / "source_verification.json")
        download = read(root / "download_verified.json")
        artifacts = [out / x for x in ("summary.json", "validation.json", "evaluation_contract.json",
                                      "runtime.json", "episodes.jsonl")]
        artifacts += [root / x for x in ("status.json", "serving_audit.json", "source_verification.json",
                                        "download_verified.json", "review_record.json", "episode_ids.json")]
        artifacts += list(out.glob("shard-*/episodes.jsonl"))
        evidence["runs"][label] = {
            "root": str(root.relative_to(REPO)), "result_directory": out.name,
            "source_commit": source.get("source_commit", source.get("commit")),
            "branch": source["branch"], "hub_revision": download["revision"],
            "weights_sha256": contract["weights_sha256"], "summary": summary,
            "successes": int(success), "completed_utc": status["updated_utc"],
            "serving_qualification_passed": audit["passed"], "final_validation": validation,
            "shard_counts": counts, "worker_exit_codes": [w["exit_code"] for w in status["workers"]],
            "fresh_checks": {k: True for k in ("exact_unique_ordered_coverage", "merged_equals_shards",
                              "all_summary_fields_recomputed", "common_shard_contracts_and_runtimes",
                              "contract_source_hashes", "common_local_protocol")},
            "artifact_sha256": {str(p.relative_to(REPO)): digest(p) for p in artifacts},
        }
        print(label, int(success), f"SR={100*summary['sr']:.6f}%", f"SPL={100*summary['spl']:.6f}%", flush=True)
        del rows, shard_rows, sr
    a = paired["M5_E1"]
    for label, b in paired.items():
        if label == "M5_E1":
            continue
        assert a.keys() == b.keys()
        assert all(a[k]["identity"] == b[k]["identity"] for k in a)
        evidence["comparisons"].append({
            "a": "M5_E1", "b": label, "shared_episodes": len(a),
            "episode_uids_scenes_instructions_equal": True,
            "delta_sr_pp": 100 * sum(a[k]["success"] - b[k]["success"] for k in a) / len(a),
            "delta_spl_pp": 100 * sum(a[k]["spl"] - b[k]["spl"] for k in a) / len(a),
            "a_only_successes": sum(a[k]["success"] == 1 and b[k]["success"] == 0 for k in a),
            "b_only_successes": sum(b[k]["success"] == 1 and a[k]["success"] == 0 for k in a),
            "descriptive_only": True,
        })
    root = REPO / "evaluation/m5_e1_epoch1_20261019"
    resume = read(root / "latest_resume.json")
    for entry in resume["journals"]:
        with Path(entry["path"]).open("rb") as stream:
            prefix = stream.read(entry["bytes"])
        assert hashlib.sha256(prefix).hexdigest() == entry["sha256"]
        assert len(prefix.splitlines()) == entry["episodes"]
    evidence["m5_e1_resume"] = {"record": resume, "original_1099_journal_prefixes_reverified": True,
                               "remaining_740_completed": True}
    evidence["common_local_protocol"] = common_protocol
    evidence["verifier_sha256"] = digest(Path(__file__))
    target = Path(__file__).with_name("LOCAL_R2R_FINAL_20261019.json")
    target.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print("PASS", target)
    print(json.dumps(evidence["comparisons"], indent=2))


if __name__ == "__main__":
    main()
