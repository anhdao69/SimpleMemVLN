"""Submit only after a successful preflight; preserve partial submission IDs."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--verified-preflight", action="store_true", required=True)
    args = p.parse_args()
    root, campaign = Path(args.root).resolve(), Path(args.campaign).resolve()
    if (
        json.loads((campaign / "preflight_pass.json").read_text()).get("passed")
        is not True
    ):
        raise ValueError("Preflight has not passed")
    # Exclusive creation prevents accidentally duplicating a partially submitted run.
    with (campaign / "submission.jsonl").open("x") as journal:

        def submit(kind, variant, dependency=None):
            command = ["sbatch", "--parsable", "--export=NONE"]
            if dependency:
                command += [
                    f"--dependency=afterok:{dependency}",
                    "--kill-on-invalid-dep=yes",
                ]
            command += [
                str(root / f"train/slurm/{kind}.slurm"),
                str(root),
                str(campaign),
                variant,
            ]
            result = subprocess.check_output(command, text=True).strip()
            job = result.split(";")[0]
            if not job.isdigit():
                raise ValueError(f"Unexpected sbatch response: {result}")
            row = dict(kind=kind, variant=variant, job_id=job, dependency=dependency)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            print(json.dumps(row), flush=True)
            return job

        full = submit("train_b", "fullcontext")
        window = submit("train_b", "window8", full)
        submit("upload_b", "fullcontext", full)
        submit("upload_b", "window8", window)


if __name__ == "__main__":
    main()
