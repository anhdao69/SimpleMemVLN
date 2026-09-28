"""Real-image offline/streaming parity under each policy's own visibility."""
import argparse
import json
from pathlib import Path
import torch
from qwen_vl.train.vln_runtime import resolve_config, load_model
from qwen_vl.stream.cache import (
    make_stream_cache_fp32,
    evict_kv,
    kv_length,
    assert_state_dtypes,
)
from qwen_vl.stream.positions import PositionLedger
from qwen_vl.contracts import select_episode_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-path")
    p.add_argument(
        "--output-mode",
        default="classification",
        choices=["classification", "qwen_text"],
    )
    p.add_argument(
        "--memory-mode", default="full_context", choices=["full_context", "window8"]
    )
    p.add_argument("--episode-list")
    p.add_argument("--steps", type=int)
    p.add_argument("--forced-actions", action="store_true")
    p.add_argument("--out", default="artifacts/parity.json")
    p.add_argument("--atol", type=float, default=0.15)
    p.add_argument("--rtol", type=float, default=0.03)
    p.add_argument("--max-rms", type=float, default=0.15)
    args = p.parse_args()
    cfg = resolve_config(
        "configs/vln_r2r_v0_base.yaml",
        f"configs/vln_r2r_v0_{args.output_mode}.yaml",
        "configs/vln_memory_window8.yaml" if args.memory_mode == "window8" else None,
    )
    torch.manual_seed(429)
    model, serializer = load_model(cfg, args.model_path)
    model.eval()
    episodes = [json.loads(l) for l in Path(args.manifest).read_text().splitlines()]
    episodes.sort(key=lambda e: len(e["steps"]))
    if args.steps:
        episodes = [e for e in episodes if len(e["steps"]) >= args.steps]
        if not episodes:
            raise ValueError("No complete episode meets requested minimum step count")
    if args.episode_list:
        episodes = select_episode_ids(
            episodes, json.loads(Path(args.episode_list).read_text())
        )
    else:
        # Three complete real trajectories, with different lengths, economical first gate.
        distinct = sorted(set(len(e["steps"]) for e in episodes))
        episodes = [
            next(e for e in episodes if len(e["steps"]) == n)
            for n in [
                distinct[0],
                distinct[min(4, len(distinct) - 1)],
                distinct[min(10, len(distinct) - 1)],
            ]
        ]
    if not episodes:
        raise ValueError("Parity requires at least one matching complete episode")
    report = []
    with torch.inference_mode():
        for episode in episodes:
            encoded = serializer.encode_episode(episode)
            tensors = {
                k: v.cuda() if torch.is_tensor(v) else v for k, v in encoded.items()
            }
            lm = model.backbone.model.language_model
            pos = PositionLedger().append(
                model.backbone.model,
                tensors["input_ids"],
                tensors["mm_token_type_ids"],
                tensors["image_grid_thw"],
            )
            embeddings = model.embed(
                tensors["input_ids"], tensors["pixel_values"], tensors["image_grid_thw"]
            )
            reference = lm(
                inputs_embeds=embeddings,
                position_ids=pos,
                use_cache=False,
                step_plan=encoded["step_plan"],
            ).last_hidden_state
            cache = make_stream_cache_fp32(model.config.text_config)
            prefix, spans = encoded["step_plan"]
            ledger = PositionLedger()
            chunks = []

            def append(s, e, grid=None):
                block_pos = ledger.append(
                    model.backbone.model,
                    tensors["input_ids"][:, s:e],
                    tensors["mm_token_type_ids"][:, s:e],
                    grid,
                )
                assert torch.equal(block_pos, pos[..., s:e])
                result = lm(
                    inputs_embeds=embeddings[:, s:e],
                    position_ids=block_pos,
                    use_cache=True,
                    past_key_values=cache,
                    stream_append=True,
                ).last_hidden_state
                chunks.append(result)
                assert_state_dtypes(cache)

            append(0, prefix)
            peak = kv_length(cache)
            for t, (s, e) in enumerate(spans):
                if args.memory_mode == "window8" and t >= 8:
                    old_s, old_e = spans[t - 8]
                    evict_kv(cache, prefix, old_e - old_s)
                if args.output_mode == "classification":
                    append(s, e, tensors["image_grid_thw"][t : t + 1])
                else:
                    targets = tensors["response_target_positions"][
                        tensors["response_action_index"] == t
                    ]
                    first = int(targets[0])
                    eos = int(targets[-1])
                    append(s, first, tensors["image_grid_thw"][t : t + 1])
                    for j in range(first, eos):
                        append(j, j + 1)
                    append(eos, e)
                peak = max(peak, kv_length(cache))
            streamed = torch.cat(chunks, 1)
            positions = (
                tensors["read_positions"]
                if args.output_mode == "classification"
                else tensors["response_target_positions"] - 1
            )
            head = (
                model.classifier
                if args.output_mode == "classification"
                else model.backbone.lm_head
            )
            ref = head(reference[0, positions]).float()
            actual = head(streamed[0, positions]).float()
            error = (ref - actual).abs()
            margins = ref.topk(2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
            flips = ref.argmax(-1) != actual.argmax(-1)
            unexpected = flips & (margins > 2 * error.max(-1).values)
            item = dict(
                episode_uid=episode["episode_uid"],
                steps=len(spans),
                tokens=pos.shape[-1],
                max_logit_error=float(error.max()),
                rms_logit_error=float(error.square().mean().sqrt()),
                argmax_agreement=float((~flips).float().mean()),
                high_margin_flips=int(unexpected.sum()),
                per_read_max_error=error.max(-1).values.cpu().tolist(),
                margins=margins.cpu().tolist(),
                peak_kv_tokens=peak,
            )
            # Sequence-shape control: recompute uncached prefixes at early and
            # late decisions. This distinguishes carry error from BF16 kernels.
            if args.output_mode == "classification":
                controls = []
                for step in [0, 3, len(spans) - 1]:
                    end = spans[step][1]
                    h = lm(
                        inputs_embeds=embeddings[:, :end],
                        position_ids=pos[..., :end],
                        use_cache=False,
                        step_plan=(prefix, spans[: step + 1]),
                    ).last_hidden_state
                    logits = head(h[0, -1]).float()
                    controls.append(
                        dict(
                            step=step,
                            uncached_prefix_vs_full=float(
                                (logits - ref[step]).abs().max()
                            ),
                            uncached_prefix_vs_stream=float(
                                (logits - actual[step]).abs().max()
                            ),
                        )
                    )
                item["shape_controls"] = controls
            report.append(item)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps(
                    dict(
                        memory_mode=args.memory_mode,
                        output_mode=args.output_mode,
                        atol=args.atol,
                        rtol=args.rtol,
                        episodes=report,
                    ),
                    indent=2,
                )
                + "\n"
            )
            print(
                json.dumps(
                    {
                        k: v
                        for k, v in item.items()
                        if k not in ("per_read_max_error", "margins")
                    }
                ),
                flush=True,
            )
            assert not unexpected.any()
            try:
                torch.testing.assert_close(actual, ref, atol=args.atol, rtol=args.rtol)
                assert item["rms_logit_error"] <= args.max_rms
                item["numerical_gate_passed"] = True
            except AssertionError as exc:
                item["numerical_gate_passed"] = False
                item["numerical_failure"] = str(exc)
            if args.memory_mode == "window8":
                assert (
                    peak
                    <= cfg["observations"]["max_prefix_tokens"]
                    + 8 * cfg["observations"]["max_step_group_tokens"]
                )
        Path(args.out).write_text(
            json.dumps(
                dict(
                    memory_mode=args.memory_mode,
                    output_mode=args.output_mode,
                    atol=args.atol,
                    rtol=args.rtol,
                    max_rms=args.max_rms,
                    episodes=report,
                    passed=all(e["numerical_gate_passed"] for e in report),
                ),
                indent=2,
            )
            + "\n"
        )
        if not all(e["numerical_gate_passed"] for e in report):
            raise AssertionError("Real-image numerical gate failed; diagnostics saved")


if __name__ == "__main__":
    main()
