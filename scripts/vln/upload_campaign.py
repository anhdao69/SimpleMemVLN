"""Publish only model artifacts; verify every uploaded file's downloaded SHA256."""
import argparse
import json
from pathlib import Path
import tempfile


def publication_files(run):
    run = Path(run)
    allowed = {
        "pytorch_model.bin",
        "config.json",
        "navigation.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "trainer_state.json",
    }
    files = {}
    for epoch, step in enumerate((1353, 2706, 4059), 1):
        checkpoint = run / f"checkpoint-{step}"
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        if state["epoch"] != epoch or state["global_step"] != step:
            raise ValueError("Epoch snapshot does not match the approved schedule")
        for required in (
            "pytorch_model.bin",
            "navigation.json",
            "config.json",
            "tokenizer.json",
        ):
            if not (checkpoint / required).is_file():
                raise ValueError(f"Missing model artifact: {checkpoint / required}")
        for path in checkpoint.iterdir():
            if path.is_file() and path.name in allowed:
                files[f"epoch-{epoch}/{path.name}"] = path
    for name in (
        "resolved_config.json",
        "run_timing.json",
        "smoke_summary.json",
        "train_results.json",
    ):
        if (run / name).is_file():
            files[name] = run / name
    return files


def main():
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download, get_token
    from qwen_vl.train.campaign import sha256_file, verify_sha256

    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--variant", choices=["fullcontext", "window8"], required=True)
    args = p.parse_args()
    campaign = Path(args.campaign)
    run = campaign / args.variant
    if (
        json.loads((run / "TRAINING_COMPLETE.json").read_text()).get("passed")
        is not True
    ):
        raise ValueError("Training completion gate has not passed")
    provenance = json.loads((campaign / "provenance.json").read_text())
    repo = "anhdao69/SimpleMemVLN-R2R-" + (
        "FullContext-B" if args.variant == "fullcontext" else "Window8-B"
    )
    token = get_token()
    if not token:
        raise ValueError("Missing protected upload credential")
    api = HfApi(token=token)
    if api.whoami()["name"].lower() != "anhdao69":
        raise ValueError("Unexpected Hub identity")
    files = publication_files(run)
    card = run / "MODEL_CARD.md"
    card.write_text(
        f"""---
base_model: Qwen/Qwen3.5-4B
library_name: transformers
tags:
- vision-language-navigation
- r2r
- simplememvln
---
# SimpleMemVLN R2R {args.variant} — option B

**Not yet evaluated in Habitat.** Training loss is not a navigation success metric.
The observation-before-action alignment is declared but not independently verified
from collector source or Habitat replay.

## Recipe
Independently initialized from Qwen/Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, not from the other variant.
Code commit: `{provenance["commit"]}`.
Serializer: `vln_append_only_chat_v1`. Native LLM text-action head (option B),
gold action history, per-action mean token CE including assistant terminator.
Frozen vision/merger; trainable text backbone and LLM head. BF16, ZeRO-2,
nonreentrant full-episode gradient checkpointing, no truncation or TBPTT.
Four H100 GPUs, one episode/rank, GAS 2: global batch 8 episodes.
R2R train: 10,819 unique episodes, 631,244 actions; distributed repetition
adds one episode per epoch. Three epochs, 4,059 updates, 122 warmup updates.
Peak LR 5e-6, cosine decay to 10% of peak, weight decay 0.01, seed 429.
Memory mode: `{args.variant}`. Window8 keeps prefix plus eight complete steps
for full-attention KV; persistent GDN state and gradients are not reset.

## Snapshots
`epoch-1`, `epoch-2`, `epoch-3` correspond to updates 1353, 2706, 4059.
**Epochs 1–2 are mid-schedule snapshots of one 3-epoch cosine run**, not
independently trained one-/two-epoch models. All optimizer states remain local.

## Loading and integrity
These weights use the SimpleMemVLN navigation wrapper, not an unmodified
`AutoModel` state dictionary. Download an epoch directory and load it with
`qwen_vl.train.vln_runtime.load_checkpoint(path, model_path)` using the pinned
base snapshot and the recorded environment. Navigation metadata contains the
complete serializer and memory contract.
`SHA256SUMS.json` records model/configuration/metric hashes. The uploader
downloads every uploaded file at the immutable commit revision and verifies
SHA256 before marking publication complete locally. Dataset images, credentials
and optimizer states are not published.
"""
    )
    files["README.md"] = card
    digests = {name: sha256_file(path) for name, path in files.items()}
    checksum_file = run / "SHA256SUMS.json"
    checksum_file.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")
    files["SHA256SUMS.json"] = checksum_file
    digests["SHA256SUMS.json"] = sha256_file(checksum_file)
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    if api.model_info(repo).private:
        raise ValueError("Target repository is not public")
    commit = api.create_commit(
        repo_id=repo,
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
            for name, path in sorted(files.items())
        ],
        commit_message=f"Publish three verified R2R {args.variant} epoch snapshots",
    )
    verified = []
    for name, expected in sorted(digests.items()):
        # One-file temporary cache bounds verification disk usage. Never modify
        # the base-model cache or remove local training checkpoints.
        with tempfile.TemporaryDirectory(prefix="verify-hf-", dir=run) as cache:
            path = hf_hub_download(
                repo,
                name,
                revision=commit.oid,
                token=token,
                cache_dir=cache,
                force_download=True,
            )
            verify_sha256(path, expected)
        verified.append({"path": name, "sha256": expected})
        print("SHA256_VERIFIED", name, expected, flush=True)
    result = {"passed": True, "repo": repo, "revision": commit.oid, "files": verified}
    (run / "UPLOAD_VERIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    print("UPLOAD_VERIFIED", repo, commit.oid, flush=True)


if __name__ == "__main__":
    main()
