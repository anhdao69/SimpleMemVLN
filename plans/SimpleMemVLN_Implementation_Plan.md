# SimpleMemVLN: Streaming Qwen3.5 for VLN

## Implementation plan v2 — actual SimpleMemVLN base, FullContext first, then Window8

**Revision date:** 2026-09-28. **Language:** English.

**Status:** Source-audited design and implementation plan. The navigation model, kernels, training loop, and GPU benchmarks described below have **not** been implemented or run as part of this document update. Configuration values are proposed starting points; acceptance gates determine whether they work on the target machines.

**Goal:** Extend the actual **SimpleMemVLN** repository at `fb039150254af66a97dd2e8591d10a35a2a9d055`, initialize its backbone from **pretrained Qwen3.5-4B**, supervise every valid navigation action, and run inference one new RGB observation at a time. First verify persistent GDN state with FullContext; then add Window8 to bound full-attention KV. SimpleMemVLA is the methodological reference, and StageVLN-v2 supplies selected VLN data/evaluator utilities.

**Recommendation after technical review:** Verify native cached continuation with **FullContext (no KV eviction)** first. Bring up **Option A: four-class action head** as the smallest end-to-end debugging path, then implement **Window8** as the bounded-memory target. Keep **Option B: Qwen text output** fully supported, with its original LM head and explicit action-history contract. A-first reduces decoding and feedback variables; it is an engineering order, not evidence that B is invalid.

Both output options use **FlashAttention from the initial implementation**. FullContext uses native causal FlashAttention-2. Window8 uses a registered text-only attention callback that gathers per-step K/V and calls direct FlashAttention-2; it does not require a copied text-model forward. SDPA-Flash remains an optional reference, not a second production backend that must be implemented.

**What changed in this revision:** Rebased integration on the verified Uniform8 `src/qwen_vl` repository; retained its HF Trainer and existing runtime stack as the starting candidate; changed row-based loss to a global action mean; specified native Accelerate tail repetition and partial-update accounting; separated training admission limits from serving context limits; removed A's textual frame number; and corrected overstatements about FLA precision, token caps, and claimed CPU test coverage. FullContext + A remains the initial debugging path, Window8 remains the bounded-memory target, and B remains fully specified. A complete FullContext training campaign and a window-size sweep are not prerequisites for Window8. Section 19 records the review dispositions.

This document is the implementation specification. Code examples describe interfaces and algorithms to build; proposed file paths and commands do not imply those files already exist.

## 1. Decisions at a glance

| Decision | Option A: action classification head | Option B: existing Qwen output |
|---|---|---|
| Backbone initialization | Pretrained Qwen3.5-4B | Same pretrained checkpoint |
| Output module | New `Linear(2560, 4)` | Existing pretrained Qwen LM head |
| Decision representation | Final-normalized hidden state at the end of a fixed post-image `Action:` cue | Hidden state immediately before each response token |
| Prediction | One four-way classification | Autoregressive canonical action string, then assistant turn terminator |
| Training target | One class label per navigation step | Action-body tokens plus assistant `<\|im_end\|>` |
| Loss shift | No token shift | Exactly one causal next-token shift |
| Action history in persistent state | No action tokens | Previous action responses are committed |
| Input action history in training | None | Expert actions, through causal teacher forcing |
| Input action history during rollout | None | Generated actions that were actually executed |
| Full-attention retention | FullContext for initial bring-up; prefix + 8 groups in Window8 | Same selectable memory modes; groups also include response and turn markers |
| Proposed maximum tokens per step | 320 | 384 |
| Proposed Window8 maximum retained KV tokens | 3,072 | 3,584 |
| New-head learning rate | `5e-5` | No new output head; use backbone LR for the pretrained LM head |
| Decoding calls | One observation append | Observation append, token decoding, final turn commitment |

The head is a choice of output interface. **Streaming GDN does not require adding an action head.** Option B keeps language-model output while changing how the model receives observations and preserves its state.

Memory mode and output mode are independent:

| Memory mode | Full-attention behavior | Purpose | Limitation |
|---|---|---|---|
| `full_context` | Keep all previously appended K/V; standard causal attention | Native continuation reference, initial overfit/pilot if resource gates pass | KV grows with episode length; full attention can solve memory tasks without proving GDN-specific use |
| `window8` | Pin instruction prefix and retain 8 complete step groups including current | Main bounded-streaming experiment | Must train with the same window; navigation memory quality remains unproven |

Run the no-eviction correctness gate before Window8. A complete FullContext training campaign is optional, not a prerequisite if its resource cost is excessive. Train each reported memory-mode comparison under its own declared visibility rule; switching a full-attention checkpoint to Window8 is an explicit transfer experiment, not exact inference for the trained policy.

### Shared v0 constraints

- One complete, unpadded episode per microbatch per rank; shuffle episodes, preserve order inside each episode.
- Independent still-image observations; the image at step `t` is captured before action `a_t`.
- Loss at **every valid action**, including the final expert STOP.
- Full-episode gradients through the selected computation graph. No detach between steps.
- Training uses `use_cache=False` and `past_key_values=None`; inference uses persistent native hybrid state.
- GDN and convolution history persist throughout an episode in both modes. Window8 evicts only full-attention KV; FullContext performs no eviction.
- Freeze the complete visual encoder and visual merger initially; fine-tune the text backbone.
- In Window8, prefix retention and FIFO eviction are deterministic. No learned pruning, keyframe retrieval, distillation, or Uniform8 warm-start is required.
- Reset the entire stream between episodes. No episode packing or batched asynchronous serving in v0.

These are initial engineering choices, not claims of an optimal VLN architecture or a CVPR-level contribution.

## 2. Source audit and reproducibility boundary

### 2.1 Pinned sources and their roles

| Component | Audited revision / candidate version | Primary source |
|---|---|---|
| **Actual implementation base** | SimpleMemVLN `fb039150254af66a97dd2e8591d10a35a2a9d055` | [Repository at audited commit](https://github.com/anhdao69/SimpleMemVLN/tree/fb039150254af66a97dd2e8591d10a35a2a9d055) |
| VLN data/evaluator port source | StageVLN-v2 `f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403` | [Repository at audited commit](https://github.com/anhdao69/StageVLN-v2/tree/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403) |
| Methodological reference | SimpleMemVLA `c564c17d276d7294200122b286c21901a3bfb99f`; paper v1 | [Code reference](https://github.com/OpenBMB/SimpleMemVLA/tree/c564c17d276d7294200122b286c21901a3bfb99f), [paper](https://arxiv.org/html/2609.05533v1) |
| Pretrained backbone and processor | `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | [Model configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json) |
| Transformers | `5.11.0`, source `e7b5b964e6f64923f2770208178f3ed367978895` | [Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py) |
| Flash Linear Attention | `flash-linear-attention==0.5.2`, `fla-core==0.5.2`; source `9c8e42e762fce087c27b673af4922795d9edb85e` | [Gated-delta chunk implementation](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/gated_delta_rule/chunk.py) |
| FlashAttention-2 | Preserve the repository's `2.8.3` Torch 2.10/CUDA 12.9 wheel; source `060c9188beec3a8b62b33a3bfa6d5d2d44975fab` | [Direct attention API](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/flash_attn/flash_attn_interface.py) |
| Accelerate / DeepSpeed | `1.13.0` / `0.16.4` | [Accelerate integration](https://github.com/huggingface/accelerate/blob/v1.13.0/src/accelerate/utils/deepspeed.py), [DeepSpeed engine](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.4/deepspeed/runtime/engine.py) |

The repository/distribution is already named `simplememvln`; keep its existing `qwen_vl` import package and training entry point. Name the new navigation wrapper/configuration `SimpleMemVLN` without renaming internal pretrained identifiers `qwen3_5` and `qwen3_5_text`. Preserve licenses and attribution. The earlier plan targeted a different base; that integration assumption is superseded here, not silently attributed to this repository.

### 2.2 What the actual SimpleMemVLN repository implements

The audited path is `train.sh → qwen_vl.train.train_qwen → R2RSFTDataset/DataCollatorForSFT → Qwen3_5ForConditionalGeneration → QwenSFTTrainer`. It is a compact Uniform8 supervised-fine-tuning implementation, not an existing native streaming policy. [README](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/README.md), [training entry point](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/src/qwen_vl/train/train_qwen.py)

| Existing behavior at `fb03915` | Required episode-stream adaptation |
|---|---|
| One preselected Uniform8 decision per record; at most eight historical images plus current; exactly one human prompt and one assistant action | Add an ordered episode manifest/dataset and immutable A/B serializer; one full episode per row, all actions supervised |
| `transformers==5.3.0`; no FLA distributions in the lock | Pin exact 5.11.0, add pinned FLA, and verify real model-serving/training kernels |
| Sparse LM logits, dummy leading label/trailing logit, one causal shift | Reuse sparse projection for B with explicit action boundaries; add direct cue-end classifier A |
| Token mean per row; `_get_num_items_in_batch` counts rows | Count actions across the full optimizer update; B averages tokens inside each action, then averages actions globally |
| `model_accepts_loss_kwargs=True`; complete custom `compute_loss` with world-size compensation | Keep this Trainer contract; no separate manual DeepSpeed backward loop |
| Optimizer splits merger versus all other trainable parameters | Freeze visual encoder and merger initially; add an explicit A-head group, retain exact parameter-coverage checks |
| `model_max_length=12800`; collator refuses overlength; history-image cap | Replace with audited episode admission limits; preserve refusal rather than silent truncation |
| ZeRO-1; warmup defaults to one step; checkpointing default is off | Change actual DeepSpeed JSON to ZeRO-2, map scheduler/warmup explicitly, enable non-reentrant decoder checkpointing |
| Length sampler uses an estimate of 252 tokens per image | Use actual encoded episode lengths and one global sharding path |
| Fixed non-thinking chat template; no streaming session/evaluator | Reuse the stable template for B, version the serializer, add native cache ownership, and port the Habitat loop selectively |

Sources: [data and collator](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/src/qwen_vl/data/data_qwen.py), [Trainer/optimizer](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/src/qwen_vl/train/trainer.py), [sampler](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/src/qwen_vl/train/sampler.py), [launcher](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/train.sh), [dependencies](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/pyproject.toml), [DeepSpeed JSON](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/deepspeed.json).

These are migration tasks, not proof that the existing Uniform8 objective was wrong for its original one-action-per-row data. Merely changing the repository/model name or turning on `use_cache` does not perform the migration. Absence of FLA in a lock establishes the declared dependencies, not the exact installed packages or kernels of historical jobs.

The received second-review attachment contains a checklist and reported CPU results, but not the stated `test_qwen35_streaming_contract.py` source or execution logs. The quoted errors `4.8e-7` and `1.6e-2` remain reviewer-reported. This audit inspected source and independently exercised the upstream sampler rule on synthetic indices; it did not reproduce the reviewer's Qwen tests, run the pretrained model, install the candidate stack, or run distributed GPU training. Implement the specified tests without marking them passed in advance.

### 2.3 What to reuse from StageVLN and SimpleMemVLA

Port only relevant StageVLN episode-manifest, action-codec, checkpoint-metadata and Habitat-evaluation utilities with their source revisions. Its old generation helper creates a fresh reader cache per decision and can return after sampling EOS before committing that EOS. That cache ownership must be replaced. Do not import its learned-memory writer or old reader lifecycle into this native streaming policy. [Manifest](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/src/qwen_vl/data/episode_manifest.py), [old generation session](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/src/qwen_vl/eval/session.py), [Habitat evaluator](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/src/qwen_vl/eval/habitat_r2r.py)

SimpleMemVLA main is now a methodological/code reference, not the files to edit. Its released sampled-history, subtask-conditioned DiT training path is not full-episode VLN gradient propagation, and its released prefix pipeline rebuilds a sampled finite history rather than retaining every observation forever. [Released trainer](https://github.com/OpenBMB/SimpleMemVLA/blob/c564c17d276d7294200122b286c21901a3bfb99f/simplememvla/training/sft_trainer.py), [prefix pipeline](https://github.com/OpenBMB/SimpleMemVLA/blob/c564c17d276d7294200122b286c21901a3bfb99f/rmbench_sim/pipelined_policy.py)

Appendix E explicitly describes retaining native recurrent state while windowing only softmax attention, with 60 video units, a 5,888-token budget, and 91.0 versus 94.0 on RMBench. Those values are available in the HTML text; they need not remain labelled unconfirmed. Its eight anchor branches are not an eight-navigation-step window. The paper's 20.6 versus 88.3 RoboMME comparison instead uses a fixed 16-token recurrent historical interface trained through four decisions; it does not refute native GDN plus a window. Neither result establishes the best VLN window or exact released full-episode backward code. [Appendix E](https://arxiv.org/html/2609.05533v1#A5), [memory-interface ablation](https://arxiv.org/html/2609.05533v1#S4.SS3)

### 2.4 Confirmed multi-token cache regression: Transformers 5.3.0

In 5.3.0, the GDN state-reuse branch requires an existing state **and `seq_len == 1`**. A multi-token image append follows the chunk branch with `initial_state=None` and constructs convolution history from that new chunk. Thus this version does not implement the proposed native image-block continuation correctly. In the audited 5.11.0 source, the chunk path reads prior recurrent state and carries the convolution history forward. The single-token route alone is not an adequate compatibility test. [5.3.0 GDN implementation at commit aad13b8](https://github.com/huggingface/transformers/blob/aad13b87ed59f2afcfaebc985f403301887a35fc/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L524), [5.11.0 GDN implementation](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L449)

Pin **exactly 5.11.0** for the initial train and model-serving environments, not an open-ended `>=5.11` range. Later versions require the same regression gates. Check the interpreter that actually runs the model, including an evaluator subprocess or server; a simulator-only environment need not import Transformers. Repository launcher defaults or environment names do not prove which package is installed on the cluster.

This bug resets the explicit GDN/conv carry in that route; it does not erase full-attention KV. It also does not, by itself, explain the old StageVLN-v3 results, whose learned-memory mechanism and fresh reader calls differ from the proposed native continuation.

## 3. What streaming GDN means here

Qwen3.5-4B already contains a hybrid decoder: 24 GDN/linear-attention layers and 8 full-attention layers. We are adapting its existing recurrent computation to navigation. The backbone is pretrained; only A's small classifier is newly initialized. [Pinned architecture](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)

During inference, each new observation updates the GDN recurrent matrices and convolution buffers left by earlier observations. Full-attention layers retain detailed keys and values for either all prior steps (FullContext) or the instruction and recent steps (Window8). The next decision reuses this state instead of re-encoding all earlier images.

During training, the full chronological episode is supplied in one differentiable forward. Native GDN chunk kernels compute the recurrence over that sequence. **`use_cache=False` does not disable recurrence**; it avoids a mutable serving-cache object between training calls. Gradient checkpointing recomputes activations and preserves the intended gradient graph. It does not truncate history. The FLA chunk implementation has backward support; its fused recurrent decoding path is not the training primitive to assume. [Chunk implementation](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/gated_delta_rule/chunk.py), [fused recurrent implementation](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/fla/ops/gated_delta_rule/fused_recurrent.py)

The engineering change from ordinary independent-prompt fine-tuning is the **episode data unit, chronological token stream, local full-attention rule, and persistent inference state**. There is no need to rebuild the pretrained GDN architecture or train Qwen from random initialization.

The three relevant horizons are independent:

| Quantity | v0 value | Meaning |
|---|---|---|
| Direct full-attention history | All prior groups in FullContext; `W = 8` in Window8 | Which past K/V entries remain directly visible |
| Supervision coverage | `T` actions | Every valid expert decision has a loss |
| Gradient horizon | Whole episode | Later losses can backpropagate through earlier recurrent computation |

Recent full-attention K/V are contextual representations and can relay older information. Therefore, do not claim that every influence older than eight frames exists only in GDN. An isolated test is needed to attribute a memory benefit specifically to GDN.

## 4. Option A — four-class action head

### 4.1 Observation-only stream

Construct an instruction prefix once, then append one image block for each step:

\[
X_A=[P_A(I),B_A(o_0,0),\ldots,B_A(o_{T-1},T-1)].
\]

The prefix describes the four navigation actions and provides the instruction. A concrete format is a system message followed by an open user observation stream. Each step has native image delimiters and the fixed suffix `\nAction:` using an actual newline. Omit the textual `Frame t:` header; the step index remains metadata for grouping, positions and logging. Version this deliberate serialization change as `vln_observation_stream_v3`. Removing an unnecessary numeric feature simplifies the interface; it does not prove that larger step numbers would otherwise fail or that long-horizon generalization is solved.

`B_A` ends at the last token of that fixed `Action:` cue. Let its absolute index be `r_t`. The cue carries no target information and belongs to the same step group as its image. Read the hidden state **after the decoder's final normalization**, then compute:

\[
z_t=W_A h_{r_t}+b_A,\qquad
\ell^A_t=\operatorname{CE}(z_t,a_t).
\]

Use `Linear(hidden_size, 4, bias=True)`, weight initialization `Normal(0, 0.02)`, bias zero. Read `hidden_size` from the pinned configuration and assert the expected value, 2560. Labels align directly with the fixed cue-end read positions: **do not shift them by a token**. The earlier plan read `vision_end`; this revision makes the cue-end change explicit in the serializer/readout version and token-budget checks.

Initializing the classifier from LM embedding rows is an optional verbalizer-initialization experiment, not a required correction. A single token for a word such as `left` does not establish that its output row is a calibrated navigation classifier at this hidden position. If tested later, verify exact tokenization in the intended context, class mapping, row norms/logit scale, bias, and checkpoint provenance, and compare against the specified random initialization. Do not claim it is automatically better or secretly alter B's tied LM embeddings.

### 4.2 Inference

Append the new image block, obtain four logits, choose `argmax`, and map the class to a Habitat command. Do not append the action name or an assistant response to A's state. The stream contains instruction and observations only.

The LM head is not called. If its weights are tied to input embeddings, leaving the LM projection unused must not accidentally freeze the trainable input embeddings. A's new head receives the higher proposed LR; text-backbone parameters retain the lower LR.

### 4.3 Strengths and limits

A removes response parsing, EOS handling, and autoregressive action decoding. It is a useful minimal readout and speed comparison. Its read position and prompt interface differ from pretrained assistant generation, and it does not receive explicit previous action tokens. Good performance still needs to be learned; it is not guaranteed by the small output space.

## 5. Option B — preserve Qwen's action text output

### 5.1 Exact output contract

Use the existing pretrained LM head and these literal action strings:

`MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`.

Generate one action body followed by the assistant turn terminator `<|im_end|>`. Use non-thinking output: the empty thinking wrapper is fixed prompt scaffolding, not a generated reasoning trace. The default decoder is greedy over the normal vocabulary, with a maximum of **16 generated tokens including the terminator**.

This option retains Qwen generation. It does not introduce four trainable action tokens, replace the vocabulary projection, score only four class logits, or require a new output head.

### 5.2 Immutable per-turn serialization

Use the following append structure; bracketed image content denotes native processor-expanded image tokens:

```text
<|im_start|>system
You control a navigation agent. Follow the instruction using the current RGB view.
Reply with exactly one action: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, or STOP.
Instruction: [episode instruction]<|im_end|>
<|im_start|>user
<|vision_start|>[image tokens for observation 0]<|vision_end|><|im_end|>
<|im_start|>assistant
<think>

</think>

MOVE_FORWARD<|im_end|>
<|im_start|>user
<|vision_start|>[image tokens for observation 1]<|vision_end|><|im_end|>
<|im_start|>assistant
<think>

</think>

TURN_LEFT<|im_end|>
```

Define the stream precisely as:

\[
X_B=[P_B(I),U(o_0),R,Y(a_0),N,U(o_1),R,Y(a_1),N,\ldots].
\]

Here `U` is one user image turn, `R` is the fixed assistant-generation boundary, `Y` is the action body plus assistant terminator, and `N` is a deterministic trailing newline. **Only `Y` receives LM supervision.** User terminators, role markers, the fixed empty thinking wrapper, and `N` are prompt-only tokens.

The actual SimpleMemVLN Uniform8 labeling also supervises the trailing newline. This plan deliberately masks that deterministic separator while retaining supervision on the assistant terminator. Record this difference; do not claim byte-for-byte reproduction of the old loss. The second review's newline decision is broken across table lines, so this specification retains the explicit choice above. [Actual conversation labeling](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/src/qwen_vl/data/data_qwen.py)

**Use the repository's fixed `QWEN3_5_NON_THINKING_CHAT_TEMPLATE`.** It renders every assistant turn with the same empty thinking wrapper, so the native template's historical-turn rewriting problem is not present in that custom template. Preserve its exact bytes/hash, derive fixed fragments once, and keep the append-only invariant. If the native template is substituted later, its most-recent-versus-historical rendering difference must be checked explicitly. [Actual fixed template](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/src/qwen_vl/data/data_qwen.py#L21), [native Qwen template](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/chat_template.jinja)

Implement an append-only renderer:

1. Pin tokenizer, processor, and the repository's fixed non-thinking template. Use `add_generation_prompt=True` for both fragment-derivation calls; the custom template itself fixes the empty thinking wrapper. Use no tools, supplied reasoning content or added vision identifiers. Without the generation-prompt flag, rendering omits `R`.
2. Derive and validate the system prefix and the single-user-image-plus-assistant-generation fragment once. For example, render `[system, user-image]` and `[user-image]` with identical options, require exact suffix equality, and extract the system prefix. Use actual newline characters; inspect escaped strings in the inherited custom template.
3. Encode each approved fragment with `add_special_tokens=False`; the rendered template already contains structural tokens. Expand the image placeholder using the native processor.
4. Append action-body IDs, the assistant terminator ID, and fixed newline IDs as explicit segments. Store the four canonical action token sequences in the checkpoint's action codec.
5. Offline training concatenates these exact fragments. Inference appends the same fragments. Previously committed IDs, image assignments, and positions never change.

Independent fragment tokenization need not equal a fresh BPE encoding of the entire concatenated transcript. The required invariant is exact agreement between the training and inference serializers. Never decode and re-encode generated history to normalize its spelling or whitespace.

### 5.3 Special tokens must be resolved explicitly

| Role | Token | ID in the pinned tokenizer |
|---|---|---:|
| Assistant turn terminator | `<\|im_end\|>` | 248046 |
| Message start | `<\|im_start\|>` | 248045 |
| Padding / end-of-text token | `<\|endoftext\|>` | 248044 |
| Image start | `<\|vision_start\|>` | 248053 |
| Image end | `<\|vision_end\|>` | 248054 |

Resolve IDs from the tokenizer and assert the expected strings. Explicitly stop generation on the assistant `<|im_end|>` token. The checkpoint's text configuration contains an end-of-text ID that differs from the tokenizer's assistant-turn EOS; blindly inheriting the wrong stop ID would violate this contract. Do not copy Qwen2 token IDs. [Tokenizer configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer_config.json), [model configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)

### 5.4 Sparse causal LM loss, with one shift

For action `t`, let `J_t` contain the absolute input positions of its action-body tokens and assistant terminator. Their prediction positions are `J_t - 1`:

\[
\ell^B_t=\frac{1}{|J_t|}\sum_{j\in J_t}
\operatorname{CE}(W_{LM}h_{j-1},x_j).
\]

The first action token is predicted from the final fixed assistant-boundary token; each later token is predicted from its preceding context. The target at position `j` cannot influence `h[j-1]` under the causal computation. Ground-truth earlier action tokens are valid teacher-forcing inputs; they are not current-target leakage.

Gather hidden states at prediction positions **before** applying the LM head. Compute vocabulary logits only for supervised response positions. If using the native `logits_to_keep` interface, pass the prediction indices, set `labels=None`, and compute the aligned CE manually. Do not pass full-length labels against sparse logits or shift the sparse targets again. Both the current StageVLN adapter and the pinned Qwen implementation support the underlying sparse-projection approach. [StageVLN adapter](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/src/qwen_vl/models/qwen_adapter.py), [Qwen LM forward](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

Average token CE **within each action first**, then average action losses globally as specified in Section 11. A global mean over all response tokens is a different objective because action strings have different token lengths.

### 5.5 Decode and commit each token exactly once

A sampled token is not present in GDN/KV state until it has actually been forwarded through the model. In particular, many generation loops return immediately after sampling EOS. Returning text does not prove that the returned cache includes that final token. [Existing generation loop](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/src/qwen_vl/eval/session.py)

Use a small explicit loop:

```text
begin_step(t):
    validate request and reserve the complete step token budget
    if memory_mode == window8: evict full-attention groups older than t - 7
    append U(current_rgb) + R once
    next_logits = LM(last_hidden)
    response_ids = []

for i in range(max_response_tokens_including_eos):
    token = argmax(next_logits)
    if token == assistant_turn_end:
        append [token] + fixed_newline_ids once
        parse response_ids as exactly one canonical action
        complete request and return action
    reject forbidden structural tokens
    response_ids.append(token)
    next_logits = LM(append_token(token).last_hidden)

mark generation failure: response limit reached without terminator
```

Each append advances the position ledger and all applicable native states. Action tokens, EOS, and the newline belong to the **same step group** as their observation. They do not advance the navigation step or trigger another eviction. The final EOS and newline may be committed together in a small causal forward whose output logits are ignored.

Decode the action body with `skip_special_tokens=False` and `clean_up_tokenization_spaces=False`, so unexpected structural tokens cannot disappear during parsing. Trim surrounding whitespace only and require exactly one canonical action string with no explanatory text. Preserve the actual generated IDs in state even if they use a different token segmentation from the canonical training sequence. After a valid response, execute the parsed action; it must agree with the action recorded in persistent history.

On invalid text, an unexpected structural token, missing EOS, or response-budget overflow, **terminate that evaluation episode as an inference failure**, record the reason, and keep it in the denominator. It contributes zero SR and SPL under the failure-reporting contract in Section 13. Do not silently rewrite the response as STOP or roll back only the attention cache. If simulator cleanup requires a STOP command, record it as cleanup after failure, not a successful policy decision. A finite canonical-trie decoder can be added later as a separately reported decoding policy; it is not a v0 prerequisite.

### 5.6 What changes between training and rollout

Training commits expert actions. Rollout commits the model's own valid actions. This is standard teacher-forcing exposure mismatch in addition to navigation's changed observations after a wrong action. It does not require a new dataset for the initial behavioral-cloning baseline.

Efficient-VLN v2 provides evidence that historical expert-action access can create a harmful shortcut in packed training. Its action-isolating mask removes earlier action chunks, matching an inference interface without those chunks; it still uses causal teacher forcing within the current chunk. B deliberately commits executed action history at inference, so its shifted objective is causally valid but remains vulnerable to copying and exposure bias. Do not equate these risks with feeding the current target into its own predictor. [Efficient-VLN v2 §3.3](https://arxiv.org/html/2512.10310v2#S3.SS3)

Before training B, measure the corpus's previous-action-copy accuracy, per-class run lengths, and action transition matrix. During validation, report accuracy separately at action changes and action repeats, plus model/expert repetition rates. Class frequency alone does not determine copying accuracy. Removing historical action tokens would require changing both B's training stream and inference-state contract; masking only their full-attention keys would still allow information to flow through GDN.

Parity tests must provide **identical observations and identical committed action token IDs** to offline and streaming execution. Comparing teacher-forced history with generated history after the first disagreement is not a cache-parity test.

## 6. Dataset contract and action alignment

Use existing trajectories if they contain an instruction, ordered RGB observations, and aligned expert actions. Build an episode adapter and manifest; no subtask annotations, paired histories, distillation teacher, or counterfactual dataset is required for v0.

```json
{
  "schema_version": 1,
  "episode_uid": "r2rce:train:scene_id:episode_id:instruction_id",
  "dataset": "r2rce",
  "dataset_version": "source_version",
  "split": "train",
  "scene_id": "source_scene_id",
  "episode_id": "source_episode_id",
  "instruction_id": "source_instruction_id",
  "instruction": "Walk ...",
  "observation_action_alignment": "observation_before_action",
  "steps": [
    {
      "step_id": 0,
      "rgb_path": "relative/path/frame000000.png",
      "action_name": "MOVE_FORWARD",
      "is_valid": true
    }
  ]
}
```

The `source_*` fields illustrate required metadata, not fallback values. Preserve scene and instruction identity to avoid collisions between multiple instructions on a path or future R2R/RxR datasets. The audited exporter writes scene metadata while the older loader loses some of it and hardcodes a dataset name; port and correct that boundary. [Existing manifest implementation](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/src/qwen_vl/data/episode_manifest.py)

### Shared action codec

| Model class ID | Canonical action string | Habitat action ID |
|---:|---|---:|
| 0 | MOVE_FORWARD | 1 |
| 1 | TURN_LEFT | 2 |
| 2 | TURN_RIGHT | 3 |
| 3 | STOP | 0 |

Both options return an action through this codec. Never send A's raw argmax index directly to `env.step`. B parses text into the same model action enum before environment mapping. Save both mappings in each checkpoint. [Model class order](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/src/qwen_vl/data/episode_manifest.py), [Habitat action IDs](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/src/qwen_vl/eval/habitat_r2r.py#L32)

### Preflight requirements

- Verify that `o_t` is the observation **before** `a_t`, using collector evidence and replay of the first step, a turn, and the terminal observation. Matching filenames alone is insufficient.
- Require consecutive step IDs, existing images, valid labels, and one final expert STOP under the chosen dataset contract. Preserve the image used to decide STOP; do not invent STOP labels or repeat STOP for padding.
- Preserve official train/val_seen/val_unseen and scene membership. Do not split individual frames or instruction variants across partitions.
- Audit episode lengths, action counts/distribution, instruction tokens, RGB sizes, encoded tokens, and per-step token maxima for each output option.
- Reject missing/invalid within-episode targets during v0 preflight with an explicit report. Do not silently remove their observations. Supporting missing labels later requires a versioned chronology-preserving contract; B would also need an explicit action-history policy for such steps.
- Determine optimizer steps from the actual selected manifest. An estimate of 18–19k episodes is not a scheduling constant or a guarantee of generalization. Pilot and validation results determine whether the data cover the task well enough.

The initial loader tail follows Accelerate's native repetition policy described in Section 11.3. Log repeated episodes/actions separately from unique corpus coverage. A later exact-once variant may use `loss_weight=0` for whole duplicate fill episodes; that is distinct from missing targets inside a real episode.

### Audited R2R snapshot and limits of that evidence

The tracked preparation log records **10,819 episodes, 631,244 action labels**, and a maximum training length of **183 steps**. Counts are MOVE_FORWARD 404,912; TURN_LEFT 111,177; TURN_RIGHT 104,336; STOP 10,819. This is approximately 64.15% forward and 1.714% STOP. These are the audited snapshot's counts, not universal R2R constants. [Preparation log](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/implementations/data_preparation.log)

The counts and filename/label checks do not independently prove capture-before-action timing or exactly one STOP at the final position of every current episode. Preserve the preflight checks above. The maximum training length is not the rollout horizon: evaluation can run to 500 steps. A later 18–19k-episode manifest must be audited separately rather than silently replaced with this snapshot's schedule.

## 7. Visual preprocessing and absolute positions

### 7.1 Still-image token contract

Use the processor from the same pinned checkpoint. The initial camera contract is RGB `uint8`, shape `[480, 640, 3]`, with no augmentation. Unexpected dimensions require a versioned preprocessing decision before training; do not silently stretch or crop images.

With patch size 16 and spatial merge size 2, the expected image grid is `[1, 30, 40]`, giving 300 merged visual tokens. Assert the actual processor output and image-token expansion; configuration arithmetic is not a substitute for running this check. Temporal patch size 2 does not require future observations: use the single-image path, including native replication of the same image if needed. Never combine `o_t` with `o_(t+1)` before predicting `a_t`. [Pinned visual configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)

The full prefix cap is 512 decoder tokens, including its instructions and delimiters. A's full image-step cap is 320. B's **entire completed turn** cap is 384, including the user turn, assistant boundary, response, EOS, and separator. Reserve the maximum response/closure space before appending a new B step. Verify that all four tokenized canonical responses plus EOS fit the 16-token response cap.

These are upper bounds; **do not pad blocks to the caps**. Padding tokens can update GDN and convolution state. An oversized instruction or step is a preflight error with episode ID and measured counts, not permission for silent truncation.

During offline training, encode images on demand using frozen visual microbatches, initially four images per visual call. A single call containing all episode images is causally valid if visual attention remains per image, but `no_grad()` does not guarantee that its transient activation peak fits; profile before increasing the microbatch. Only visual feature extraction belongs under `no_grad()`: text `embed_tokens` and image-feature insertion must preserve the trainable text-embedding graph. Keep features in chronological order and preserve native per-image boundaries; batching images must not introduce cross-image visual attention. Frozen visual evaluation can precede the text forward without giving text queries access to future images. During single-stream rollout, encode only the one currently available RGB observation; never wait for future observations to fill a visual microbatch. Do not cache the entire feature dataset by default; its storage cost can be substantial.

### 7.2 Two position counters

Maintain a position ledger with:

1. `logical_token_count`: number of appended decoder tokens since episode start.
2. `mrope_cursor`: next coordinate for the temporal/height/width axes.

Use native `get_rope_index` with the fragment's `mm_token_type_ids` and `image_grid_thw` to build local three-axis positions. Offset them by `mrope_cursor`, then advance that cursor to the maximum assigned coordinate plus one. Build an independent logical text axis from `logical_token_count`. The adapter supplies `position_ids` with shape `[4, B, L]`: text axis followed by the three MRoPE axes. Native `get_rope_index` returns three axes; the adapter adds the fourth. [Pinned position implementation](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

Gate: blockwise-concatenated three-axis positions must match native full-sequence `get_rope_index` on the **same immutable token sequence**; axis zero must match the independent logical arange. Cover multiple images and variable text lengths. B's action tokens, EOS, and newline advance both counters as text.

In the pinned batch-one FlashAttention wrapper, a non-unit gap or repeat in the **text** position axis can trigger packed-sequence routing. Supply `offset, offset+1, ..., offset+L-1` for each append. The three MRoPE axes may repeat or jump as part of the native image geometry; do not flatten or renumber those axes to satisfy a text-only check. Instrument FullContext's native Flash route to confirm ordinary single-sequence execution. This is separate from the GDN continuation regression. [Pinned Flash wrapper](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/modeling_flash_attention_utils.py)

Eviction does not decrease either counter. Keep rotated keys at their original positions; never renumber or rotate them again. Always pass explicit positions. Do not derive global positions from physical retained KV length, cache sequence length, or an uncorrected model-level `rope_deltas` value.


### 7.3 Training admission limits are separate from serving context

Replace the actual launcher's 12,800-token cap for the episode path. That cap prevents full-corpus episode training, but the review's statement that it rejects **every** episode is not supported by an average above the cap; short episodes can fit. Audit and report exact token lengths and the number exceeding each candidate threshold before launching.

For the audited maximum of 183 actions, the declared prefix/step bounds give at most `512 + 183×320 = 59,072` A tokens and `512 + 183×384 = 70,784` B tokens. Use **65,536 for A** and **73,728 for B** as initial training admission candidates under those bounds. B may use 65,536 only after actual serialization proves every selected episode fits; the review's approximate 59k B maximum is not an independently measured bound. These are admission limits, not padding targets or GPU-fit guarantees.

Set the episode collator and relevant HF tokenizer/configuration cap consistently; reject and report overlength rather than dropping/truncating it. A different manifest, larger images, or changed prompt requires a new audit. Do not propagate this training cap into FullContext serving: 500 steps can require up to 160,512 A or 192,512 B tokens under the same conservative bounds. Serving has its own model-context ceiling and resource profile in Section 8.

## 8. FullContext reference, Window8 target, and KV budgets

### 8.1 FullContext: initial continuation reference

In `full_context`, keep all appended keys and values and use normal causal FlashAttention-2 over the complete episode for training. At inference, append the current observation/cue or response tokens to the existing native cache. No FIFO or custom attention mask is needed. Explicit multimodal positions, FP32 recurrent storage, fresh reset, and B's token-commit rules still apply.

This mode reduces implementation variables and preserves all explicit history. It is still streaming because prior images are not re-encoded. It is **not bounded-cost in episode length**: retained KV grows linearly and each new query can attend to a growing history. Full-episode training still has quadratic softmax-attention arithmetic even with FlashAttention's memory-efficient implementation.

Use the model's 262,144-token structural context limit as a hard ceiling, not a promise of GPU feasibility. Enforce a separately measured run limit if resources require it; do not silently evict, truncate, or switch policies at that boundary.

### 8.2 Window8: retain a prefix and complete navigation steps

In `window8`, for either output option, pin the full instruction prefix and retain the eight most recent step groups **including the current step**. A group is an observation plus fixed decision cue in A; it is the complete observation/assistant/action/closure turn in B.

The existing **Uniform8** dataset has up to **eight sampled historical images plus the current image**, or nine images. **Window8** here has up to **seven recent historical observations plus current**, with persistent recurrent history from earlier steps. The shared numeral does not mean matching selected observations, explicit image count, or memory state. Keep those differences visible in comparisons; a matching-count control would need its own declared window and budget.

For a query in step `t`, a key is visible if it is causally earlier or equal and either belongs to the prefix or has a step ID in `[max(0,t-7), t]`. Prefix queries only see their causal prefix. All keys must belong to the same episode.

Before the first token of step `t`, evict all full-attention K/V for groups older than `t-7`. Do this at every full-attention layer. Do not evict after the new observation has already attended to expired keys. Do not evict individual action tokens or image patches within a group.

During training, enforce the same visibility through the attention computation while retaining the differentiable graph. During inference, enforce it by physically selecting the resident cache entries. **Never crop GDN matrices or convolution buffers when cropping full-attention KV.**

### 8.3 Memory arithmetic

The pinned checkpoint uses 8 full-attention layers, 4 KV heads per layer, and head dimension 256. BF16 K and V therefore cost, across all full-attention layers:

\[
2\times8\times4\times256\times2=32768\text{ bytes/token}=32\text{ KiB/token}.
\]

The 24 recurrent matrices, with 32 value heads and 128-by-128 state per head, total 48 MiB in FP32. The native convolution buffers add approximately 1.5 MiB in BF16 under this configuration. These estimates follow the pinned architecture and native buffer shapes. [Model configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json), [native GDN implementation](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

| Window8 per-stream planning bound | Option A | Option B |
|---|---:|---:|
| Prefix cap | 512 tokens | 512 tokens |
| Step-group cap | 320 tokens | 384 tokens |
| Retained step groups | 8 | 8 |
| Maximum retained KV length | `512 + 8×320 = 3,072` | `512 + 8×384 = 3,584` |
| Full-attention K/V | 96 MiB | 112 MiB |
| FP32 GDN state | 48 MiB | 48 MiB |
| BF16 convolution buffers | ~1.5 MiB | ~1.5 MiB |
| Principal persistent tensors | ~145.5 MiB | ~161.5 MiB |
| Absolute KV error guard | 4,096 tokens | 4,096 tokens |

The guard is an assertion, not permission to keep extra steps. These numbers exclude model weights, image features, temporary gather/copy buffers, kernel workspaces, allocator overhead, and training activations. Actual block lengths can be smaller. They assume the FP32 recurrent-cache fix in Section 10.

For FullContext, using those same per-step upper bounds gives:

| Horizon | A maximum tokens / BF16 KV | B maximum tokens / BF16 KV |
|---|---:|---:|
| 183 steps, maximum in the audited training snapshot | 59,072 / 1.803 GiB | 70,784 / 2.160 GiB |
| 500 steps, evaluation limit | 160,512 / 4.898 GiB | 192,512 / 5.875 GiB |

Add recurrent/conv state and other runtime allocations separately. The review's approximately 1.9 GB estimate for roughly 60k tokens is reasonable **KV-only arithmetic**; it does not prove the longest training episode fits or that extra attention takes only a few milliseconds. The 183-step training maximum is not an inference memory cap. Profile both 183-step training and a representative 500-step continuation before making a deployment claim.

FIFO by whole step is the Window8 choice because it is deterministic and easy to match between training and inference. Eight observations including the current one span seven action intervals; under seven successful 0.25 m forward moves that is 1.75 m, and turns/collisions reduce translation. A window is defined in observations/tokens, not a universal travel distance. A later `W={8,16,32}` comparison is reasonable, but is not required before the first working Window8 baseline. Attention-score pruning, patch selection, a keyframe bank, and changing `W` belong in later controlled experiments. If `W` or preprocessing changes, recompute every bound and rerun parity; do not silently preserve the old guard. FullContext and Window8 outputs are not expected to match after the first eviction: each is compared with its own offline reference using the same attention rule.

## 9. FlashAttention from v0

### 9.1 FullContext native path and Window8 registration

Use native `flash_attention_2` for FullContext. For Window8, register a unique backend such as `simplememvln_step_flash` with `AttentionInterface.register` and select it **only on the text configuration**. Keep vision on its separately selected native backend.

In the pinned Qwen source, this callback receives Q/K after normalization and RoPE, and K/V after native cache update. It returns attention output before Qwen applies its output gate/projection. Immutable `step_plan` metadata can pass through the existing `**kwargs` path, including checkpoint recomputation; a text-forward copy is unnecessary for the current unpadded, batch-one contract. [Native attention callback site](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L672), [checkpoint kwargs](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/modeling_layers.py)

With a custom name absent from the mask registry, the pinned mask builder returns `None` early. Use `attention_mask=None` for the unpadded episode, pass no prebuilt 4D mask, and let the callback implement all causal/window restrictions. A pre-existing 4D mask takes an earlier return path; do not construct one upstream. Never claim an arbitrary future padding/packing scheme is supported by this shortcut. [Mask builder](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/masking_utils.py#L839)

The callback must return `[B, L, Hq, D]` and the expected auxiliary attention value, normally `None`. It must not update the cache, reapply RoPE, or apply the output gate a second time. Eviction happens in the session before native append; native attention then performs its normal single cache update.

### 9.2 Window8: gather the intended keys, then apply causal attention

Do not create an episode-sized dense attention mask in production. At each full-attention layer, first compute native projections, Q/K normalization, and RoPE. For each query step, gather K/V in logical order:

`[prefix, retained previous step groups, current step group]`.

The current query block is the suffix of these selected keys. If it contains `L` query tokens and the gathered keys have length `S`, the valid rectangle is:

\[
\mathrm{allow}(q,k)\iff k\le S-L+q.
\]

This is **lower-right causal alignment**. The gap from evicted history does not matter because the remaining keys retain their order and original rotary positions.

In training, process the prefix separately and each complete step group as a causal rectangle. For B, the group may include teacher-forced future response tokens; the rectangle prevents those tokens from influencing earlier queries. At inference, a group is consumed in smaller chunks: observation prefill, one-token action continuations, and closure. Each new chunk remains the suffix of its currently available keys.

### 9.3 Window8 direct FlashAttention-2 call

Use `flash-attn==2.8.3` and its verified bottom-right causal semantics for unequal Q/K lengths. Preserve GQA: 16 query heads and 4 KV heads. Do not expand KV to 16 heads before calling the direct API.

```python
# Adapter inputs: q [B, Hq, L, D], k/v [B, Hkv, S, D].
# k/v have already been selected in logical order and include current tokens.
from flash_attn import flash_attn_func

out = flash_attn_func(
    q.transpose(1, 2).contiguous(),
    k.transpose(1, 2).contiguous(),
    v.transpose(1, 2).contiguous(),
    dropout_p=0.0,
    softmax_scale=native_scaling,
    causal=True,
    window_size=(-1, -1),
)  # [B, L, Hq, D]
```

The FIFO selection already implements the step window. `window_size=(-1,-1)` means unrestricted attention **within that selected causal rectangle**, preserving the prefix. A single native token window over the whole episode does not implement a persistent prefix plus eight variable-length step groups.

Retain the native attention scaling, output gate, head reshape, and output projection. Do not double-update K/V, double-apply RoPE, or route an already-gathered query through a second native cache update. The direct API has backward support. `flash_attn_with_kvcache` explicitly lacks backward and is not the training primitive for this plan. [Direct function and cache API](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/flash_attn/flash_attn_interface.py), [causal alignment and supported shapes](https://github.com/Dao-AILab/flash-attention/blob/060c9188beec3a8b62b33a3bfa6d5d2d44975fab/README.md)

### 9.4 Optional SDPA-Flash reference; not required for initial delivery

The review suggested dropping this alternative. That is accepted as a scope reduction: direct/native FA2 is the initial implementation path. It is not a finding that PyTorch lacks GQA support. The pinned native Flash implementation supports grouped query heads and their backward reductions; actual project shapes/builds still need numerical tests if this alternative is enabled. [Native Flash GQA checks and backward](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp)

`scaled_dot_product_attention` is a dispatcher; calling it does not by itself prove that a Flash kernel ran. For these rectangular queries, do not use plain `is_causal=True`: PyTorch's documented non-square causal convention is upper-left. If implementing this optional reference, use an explicit lower-right causal bias and require the Flash backend:

```python
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

bias = causal_lower_right(q.shape[-2], k.shape[-2])
with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=bias,
        dropout_p=0.0,
        is_causal=False,
        scale=native_scaling,
        enable_gqa=True,
    )  # [B, Hq, L, D]
out = out.transpose(1, 2).contiguous()  # [B, L, Hq, D], as in direct FA2.
```

Both backends must return the same `[B, L, Hq, D]` layout to the shared native gate/reshape/projection. This alternative must pass the same forward/backward tests and report actual dispatch on the target stack. A short dense/math implementation is a correctness oracle, not an unreported production fallback. [SDPA API](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html), [lower-right bias](https://docs.pytorch.org/docs/2.10/generated/torch.nn.attention.bias.causal_lower_right.html), [pinned dispatch implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/attention/bias.py)

### 9.5 FlashAttention and FLA serve different layers

FlashAttention accelerates Qwen's softmax full-attention layers. Flash Linear Attention supplies GDN recurrence kernels. They are separate packages and separate numerical paths; both need compatibility tests.

The direct per-step loop costs multiple launches and may retain repeated gathered K/V tensors for backward. It is a correctness-first implementation that still uses fused attention. Batched variable-length rectangles or FlexAttention can optimize it later without changing the declared visibility rule.

## 10. Native Qwen adapter and streaming cache

### 10.1 Thin native bridge and separate attention routing

The pinned Qwen text forward uses the attention-mask input for both its causal attention mask and its linear-attention mask; passing an arbitrary dictionary as `attention_mask` is not supported. Use its existing `**kwargs` path for immutable step metadata and the registered Window8 callback from Section 9. The thin adapter handles pretrained modules, embedding/image insertion, positions, readout, and cache lifecycle; it does not duplicate the text-model forward. Route:

- GDN: `None` or the native valid-token mask; v0 episodes are unpadded.
- Full attention: native causal FA2 in FullContext; prefix/step span metadata through the registered gather-and-Flash callback in Window8.
- Vision: native image attention inputs, independent of text-step metadata.

For A, call `backbone.model.language_model(...).last_hidden_state` and gather only the cue positions before the four-class projection. Keep `output_hidden_states=False`; do not request all layer outputs to access the final representation. B similarly gathers predictor states before vocabulary projection. Frozen visual extraction must not detach the trainable text embeddings.

Preserve native weights and module ownership with source-version assertions. In checkpointed training, pass immutable span/position metadata to recomputation; never mutate a serving cache or external step cursor. Any later reason to fork text forward must be demonstrated by a missing native interface, not assumed in advance. [Pinned Qwen text forward](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py)

### 10.2 Cache state and FP32 recurrent storage

`StreamState` owns the native mixed cache, prefix length, resident step spans, two position cursors, episode/request IDs, and the last completed result. It contains no autograd graph. Serving runs under `model.eval()` and `torch.inference_mode()`.

Initialize a fresh `DynamicCache(config=text_config)` per episode. Before prefix prefill, replace only the linear-attention cache layers with a small subclass that preserves recurrent state in FP32 while keeping native convolution storage in BF16.

Reason: the audited Transformers 5.11.0 linear-cache initialization can allocate recurrent storage using the convolution state's dtype. An incoming FP32 FLA state can then be copied into BF16 storage. The configuration's `mamba_ssm_dtype=float32` alone does not prove that this particular cache allocation stays FP32. This is a source-level precision risk; numerical drift has not been measured here. [Linear cache implementation](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/cache_utils.py)

Override `update_recurrent_state` narrowly: allocate FP32 storage on first use from the incoming shape/device, set the required initialization flags correctly, and copy subsequent states into that FP32 destination. Preserve convolution methods, native state shape, and `has_previous_state` semantics. Assert dtype after prefix prefill, multi-token image updates, and B's one-token decode updates.

FP32 cache storage and internal arithmetic are separate guarantees. With autocast disabled, both audited PyTorch GDN fallbacks promote their working tensors to FP32 before converting `initial_state`; they do not automatically undo the cache fix merely because the model weights are BF16. However, the chunk fallback's intervening `value = attn @ v_beta` can produce lower-precision output under an enclosing autocast context, so `initial_state.to(value)` can then round the incoming state. This is a source-level risk, not a measured drift result. Record the autocast policy, disable autocast around any fallback precision reference, and verify the production FLA bindings and continuation numerics; checking the destination cache dtype alone does not establish internal precision. [Fallback implementations](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L246-L368), [PyTorch autocast semantics](https://docs.pytorch.org/docs/2.10/amp.html#autocasting)

For Window8 FIFO, modify only full-attention K/V along the sequence dimension before the new append. An `index_select`/concatenation implementation is sufficient initially. FullContext does not invoke this operation. Do not call a generic hybrid-cache crop/rollback and assume it preserves recurrent history.

### 10.3 Reset, retries, and failure handling

Create a new cache object at reset; do not rely on zeroing an existing cache. The pinned generic cache reset can retain tensor sequence dimensions. Reset model-level rotary bookkeeping, the resident deque, both position cursors, and request counters. Assert zero KV length and no prior recurrent state before prefix prefill. [Cache lifecycle source](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/cache_utils.py)

Accept one episode and one in-flight request per session in v0. A retry of the last completed `(episode_uid, step_id)` returns its saved result without another forward. Reject out-of-order or mismatched requests.

A failure partway through a mutating append invalidates the session: abort the episode and count the failure. Do not continue from partially updated state. Recovery through reset-and-replay requires caller-supplied observations in a later explicit recovery path; the policy must not maintain an unbounded RGB archive merely to enable recovery.

In Window8, resident metadata must also stay bounded: retain prefix and current-window spans, scalar counters, and the last result. FullContext necessarily retains growing K/V, but need not keep an RGB archive or duplicate every position tensor. Send growing diagnostics to external logs; do not accumulate all images or logits inside the serving policy.

## 11. Full-episode training for both options

### 11.1 One differentiable episode forward

For each microbatch:

1. Load one ordered episode and construct A or B tokens with the shared versioned processor.
2. Encode images under `no_grad()` with the frozen visual module in evaluation mode.
3. Run the text backbone on the full sequence with `use_cache=False`, `past_key_values=None`, native differentiable GDN chunk execution, and the selected memory mode: native full causal FA2 or registered Window8 FA2.
4. A: gather fixed cue-end hidden states and compute four-class CE. B: gather shifted response prediction states and compute per-action token-mean CE.
5. Sum valid action losses locally, apply the distributed scale below, and backpropagate through the complete episode graph.
6. Start the next episode with a new sequence; no training cache is carried between samples.

Checkpoint decoder layers with non-reentrant checkpointing. Retain the original parameter graph, including recurrent influences from earlier steps. Window8 does not detach the GDN graph. Full-episode gradients permit distant credit assignment, but do not force distant-history use if local observations already predict the action well; measure that separately.

### 11.2 Freeze and optimizer policy

Freeze **all visual encoder and visual merger parameters**. Override wrapper `train(mode)` so that `visual.eval()` remains true after the trainer calls `model.train()`. The text embedding, GDN, full attention, MLP, and normalization parameters remain trainable.

Use module-based optimizer groups and deduplicate tied parameters by identity:

- Text backbone: proposed LR `5e-6`.
- A's newly initialized classifier: proposed LR `5e-5`.
- B's existing LM head: backbone LR, including tied input/output embeddings. Do not copy A's new-head LR multiplier onto this pretrained vocabulary matrix.
- Bias and normalization parameters: no weight decay; other trainable weights use proposed decay `0.01`.

Log parameter counts and LR for each group. Verify nonzero gradients reach the intended modules in a smoke run; the list of `requires_grad=True` parameters alone is not enough.

### 11.3 Global mean over actions

Retain and adapt the actual repository's `QwenSFTTrainer`, using Transformers 5.11.0, Accelerate 1.13.0, and DeepSpeed 0.16.4. Its existing `compute_loss` fully overrides the base Trainer implementation; it already enables loss-kwargs normalization and compensates data-parallel gradient averaging. Replace its row-based item count and row-based token mean with action-based equivalents. Do not add a separate handwritten DeepSpeed backward loop. [Base repository trainer](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/src/qwen_vl/train/trainer.py#L13)

For an optimizer update, let `E` be the multiset of episode exposures actually consumed across all data-parallel ranks and accumulation microsteps. Define:

\[
\mathcal L=\frac{\sum_{e\in E}\sum_{t\in\mathrm{valid}(e)}\ell_{e,t}}
{N},\qquad
N=\sum_{e\in E}|\mathrm{valid}(e)|.
\]

For A, `ell` is one four-class CE. For B, first average CE over the supervised body and assistant-terminator tokens **within each action**, then sum those action means. The denominator counts actions, not response tokens, episodes, retained KV tokens, or accumulation microsteps. A repeated episode at the loader tail is a real additional exposure under the initial policy below and contributes its actions to both numerator and denominator.

Use one complete episode per microbatch per rank. The initial four-GPU recipe uses GAS two: eight episode exposures in a full update. With eight GPUs and GAS one, the nominal update size is also eight, but its tail accounting must be computed for that configuration rather than copied from the four-GPU example.

The custom Trainer contract is:

1. Preserve the existing explicit `self.model_accepts_loss_kwargs = True` after `super().__init__`. Do not rely on the new wrapper's forward signature to select this behavior.
2. Override `_get_num_items_in_batch(batch_samples, device)` to count valid actions across **all microbatches prefetched for the current update**, then globally sum that count once. Use validated per-episode action metadata, including the final STOP. Do not reuse `labels.shape[0]` or HF's default nonignored-token counter.
3. Keep the complete `compute_loss` override. For each local microbatch, compute `local_sum_of_per_action_losses` and return:

   ```python
   loss = (
       local_sum_of_per_action_losses
       * self.accelerator.num_processes
       / num_items_in_batch
   )
   ```

   Here `num_items_in_batch` is the same global action count `N` for every microbatch in that update. In the declared pure-data-parallel v0, `num_processes` equals the DP world size `D`. Require a positive, supplied denominator during training; do not silently fall back to an episode mean.
4. Let Trainer call `accelerator.backward`. Do not divide by GAS again, call an additional `engine.backward`, or add another world-size multiplier. Preserve ordinary Trainer/Accelerate control of optimizer boundaries.

This arrangement matters because the pinned Trainer skips its own GAS division only when the loss/count contract is active; it sets `scale_wrt_gas=False` for DeepSpeed. Accelerate then forwards that argument without performing its own DeepSpeed GAS division. ZeRO's DP gradient average cancels the single factor `D`, producing the objective above. In contrast, **base** `Trainer.compute_loss` can apply world-size compensation itself; calling it after retaining the subclass's multiplier would risk applying the compensation twice. [Trainer scaling](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/trainer.py#L1933), [base `compute_loss` compensation](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/trainer.py#L2022), [Accelerate backward](https://github.com/huggingface/accelerate/blob/v1.13.0/src/accelerate/accelerator.py#L2826), [DeepSpeed GAS scaling](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.4/deepspeed/runtime/engine.py#L2095)

For B, gather sparse predictor states at `J-1`, compute CE against targets at `J`, scatter token-loss sums and counts by action index, and average within each action before summing. Do not preserve the old per-row token mean when a row becomes a whole episode. Adapt the existing sparse dummy-label/logit convention or replace it with this direct indexing contract; apply the causal shift exactly once. Validate the full objective against a single-process reference using two ranks, GAS greater than one, unequal action counts, unequal B response lengths, and the short final update. Compare gradients or optimizer updates, not only displayed scalar loss. [Existing sparse-shift and row reduction](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/src/qwen_vl/train/trainer.py#L28)

Bucket by the selected output mode's **encoded token length**, not frame count or the old image-token estimate. Build one deterministic global sampler order: shuffle episodes into pools of 64, sort within each pool by encoded length, form complete groups of eight, shuffle complete-group order, and balance/randomize their rank assignments. Keep complete episodes intact. Preserve the final corpus remainder as a tail; if a pool size changes, use a multiple of eight or carry its remainder into the next pool. **Do not independently fill partial pools or pre-pad groups in addition to Accelerate's final loader padding.** Shard the resulting global order once through the Trainer/Accelerate loader. Persist the order/seed/cursor for resume, and log useful-token throughput and rank idle time.

The initial tail policy accepts Accelerate's native repeated sample with `even_batches=True`, `split_batches=False`, `dispatch_batches=False`, and `dataloader_drop_last=False`. For the audited **10,819 unique episodes**, four ranks, local batch one, and GAS two, this produces:

- **10,820 episode exposures:** every unique episode plus one repeated sampler item.
- **2,705 local microbatches per rank.**
- **1,353 optimizer updates:** 1,352 updates of eight exposures and a final update of four exposures, with one microbatch per rank.

Trainer and Accelerate explicitly mark that final partial update as a DeepSpeed accumulation boundary; the action denominator uses its actual consumed actions. Record the repeated episode ID and its action count. Log unique episode/action coverage separately from exposed episode/action counts and optimizer-update counts. Do not report a fixed corpus label count as the number of training actions actually consumed: the repeated episode adds its own variable number of actions. DeepSpeed's nominal `global_samples` counter is also unsuitable for this accounting; it can report 10,824 here because it increments by nominal batch eight even on the short final update. [Loader tail behavior](https://github.com/huggingface/accelerate/blob/v1.13.0/src/accelerate/data_loader.py#L218), [Trainer remainder handling](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/trainer.py#L1699), [explicit DeepSpeed boundary](https://github.com/huggingface/accelerate/blob/v1.13.0/src/accelerate/utils/deepspeed.py#L264), [nominal sample counter](https://github.com/deepspeedai/DeepSpeed/blob/v0.16.4/deepspeed/runtime/engine.py#L2278)

Exact-once weighted coverage with zero-weight fill slots is an optional later sampler policy. It requires explicit weights in both numerator and denominator, compatible collectives, and rerunning the loss/tail/resume gates; it is not the initial implementation. Never silently drop real tail episodes.

Save and resume at optimizer boundaries: model/readout, optimizer, scheduler, RNG, episode order/cursor, unique/exposure counters, configuration, source revisions, action codec, and serializer version. Training checkpoints need no persistent inference cache because every microbatch starts from an independent full episode.

At checkpoint load or resume, validate the saved output mode, serializer, action codec, assistant EOS/separator, preprocessing, position convention, sampler/tail policy, and cache policy against the resolved configuration. Reject conflicting overrides. Loading a B checkpoint as A, or vice versa, with a newly initialized readout is a separate initialization experiment, not an ordinary resume or deployment of the saved policy. A deliberate memory-window or preprocessing experiment must declare its changed contract and pass the relevant gates before use.

### 11.4 Shared starting configuration

```yaml
model:
  name: SimpleMemVLN
  backbone: Qwen/Qwen3.5-4B
  revision: 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  initialize_navigation_checkpoint: false
  output_mode: classification  # Use qwen_text for Option B.
  freeze_visual_encoder: true
  freeze_visual_merger: true
  dtype: bfloat16
  recurrent_cache_dtype: float32

observations:
  modality: independent_rgb_images
  expected_rgb_shape: [480, 640, 3]
  expected_visual_tokens: 300
  max_prefix_tokens: 512
  augmentation: none

memory:
  mode: full_context  # Native continuation reference; Window8 override below.
  kv_window_steps_including_current: null
  evict_before_new_step: false
  kv_guard_tokens: 262144  # Structural ceiling; resource gate may lower it.
  preserve_absolute_positions: true

training:
  gradient_mode: full_episode
  use_cache: false
  microbatch_episodes_per_rank: 1
  trainer: QwenSFTTrainer
  model_max_length: 65536  # A admission limit; B override below.
  nominal_episodes_per_update: 8  # Final update may be smaller.
  gradient_accumulation_steps: 2  # Four data-parallel ranks.
  remove_unused_columns: false
  average_tokens_across_devices: true
  dataloader_drop_last: false
  tail_policy: accelerate_repeat
  pad_each_pool: false
  accelerator_config:
    even_batches: true
    split_batches: false
    dispatch_batches: false
  length_bucket_pool_episodes: 64
  length_bucket_key: encoded_tokens
  shuffle_global_update_groups: true
  all_valid_actions: true
  loss_normalization: global_valid_actions
  epochs: 1
  evaluate_at_epoch_end: true
  backbone_lr: 0.000005
  optimizer: adamw
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps: 0.03  # HF 5.11: fraction when < 1; do not also pass warmup_ratio.
  lr_scheduler_type: cosine_with_min_lr
  lr_scheduler_kwargs:
    min_lr_rate: 0.1
  max_grad_norm: 1.0
  checkpoint_decoder_layers: true
  checkpoint_use_reentrant: false
  zero_stage: 2
  seed: 429

runtime:
  text_attention: flash_attention_2
  attention_dropout: 0.0
  require_verified_flash_backend: true
  vision_attention: flash_attention_2
  vision_microbatch_images: 4
  max_logical_context_tokens: 262144
  allow_automatic_detach: false
  allow_automatic_episode_drop: false
```

Option-specific overrides:

```yaml
# configs/vln_r2r_v0_classification.yaml
model:
  output_mode: classification
  readout: cue_end_linear_4
observations:
  serializer_version: vln_observation_stream_v3
  fixed_decision_cue: "\nAction:"
  max_step_group_tokens: 320
  append_action_tokens: false
training:
  classifier_lr: 0.00005
  action_loss: categorical_cross_entropy
```

```yaml
# configs/vln_r2r_v0_qwen_text.yaml
model:
  output_mode: qwen_text
  readout: pretrained_lm_head
observations:
  serializer_version: vln_append_only_chat_v1
  max_step_group_tokens: 384
  append_action_tokens: true
training:
  model_max_length: 73728  # Conservative B training admission candidate.
  action_loss: mean_response_token_ce_per_action
  supervise_assistant_terminator: true
  supervise_fixed_separator: false
  sparse_lm_projection: true
generation:
  do_sample: false
  max_response_tokens_including_eos: 16
  assistant_end_token: "<|im_end|>"
  commit_final_token: true
  fixed_separator: "\n"
  invalid_output: terminate_episode_as_failure
```

Window8 memory override, independent of the output option:

```yaml
# configs/vln_memory_window8.yaml
memory:
  mode: window8
  kv_window_steps_including_current: 8
  evict_before_new_step: true
  kv_guard_tokens: 4096
runtime:
  text_attention: simplememvln_step_flash
```

These YAMLs specify proposed project recipes; they are not already native HF CLI flags. Add the thin resolver described in Section 14 and map them explicitly to the existing dataclasses/Trainer and actual DeepSpeed JSON. For example, `warmup_steps=0.03` is the HF 5.11 fraction interface (41 steps for 1,353 updates); `cosine_with_min_lr` consumes `lr_scheduler_kwargs={"min_lr_rate":0.1}`. Change the launcher's old warmup default and the JSON's ZeRO stage, rather than only adding disconnected recipe fields. Keep the explicit loss-kwargs flag in Trainer code. [Warmup semantics](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/training_args.py#L2084), [minimum-LR scheduler](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/optimization.py#L338)

Use a recursive/deep merge in the order **base → output option → optional memory override** and log the resolved result. A shallow replacement of nested dictionaries would erase required fields. Reject incompatible combinations: A with appended response tokens, B without an EOS target, a step cap that cannot hold its contents, FullContext with automatic FIFO, or Window8 with an unmodified full-context backend. In FullContext, the large structural guard does not certify that all lengths fit GPU memory.

Assert that the resolved nominal episode batch equals actual data-parallel ranks × episodes per rank × GAS. The initial commands explicitly use four ranks and GAS two; an eight-rank run uses GAS one to preserve nominal batch eight and recomputes its tail counts. Do not inherit the launcher's old batch-two/GAS-eight defaults or infer the number of usable process-local GPUs from a physical-device listing alone.

One epoch is the first full-data budget, with mandatory epoch-1 validation before extending training. Longer runs require an explicit decision based on validation and compute, not an assumption that three epochs is better. Log val_seen and val_unseen and reserve the test set. Starting directly from pretrained Qwen avoids an additional Uniform8 checkpoint/interface transition; a Uniform8 or FullContext warm-start for Window8 is a separately labeled experiment.

## 12. Compatibility, memory, and gradient stability

### 12.1 Candidate environment

Start from the actual SimpleMemVLN `fb039150254af66a97dd2e8591d10a35a2a9d055` environment. Preserve its Python, Torch/CUDA, FlashAttention, Accelerate and DeepSpeed choices initially; upgrade Transformers and add pinned FLA. Regenerate and inspect the complete lock before installation. These are proposed package changes, not a claim that the resulting stack has passed runtime tests. [Repository dependency pins and package indexes](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/pyproject.toml#L10-L50)

| Package / runtime | Initial candidate |
|---|---|
| Python | `3.12` |
| PyTorch / CUDA runtime | Preserve locked `2.10.0+cu129` from the PyTorch CUDA 12.9 index |
| torchvision / Triton | Preserve locked `0.25.0+cu129` / `3.6.0` |
| Transformers | Upgrade `5.3.0` → exact `5.11.0`; audited source `e7b5b964e6f64923f2770208178f3ed367978895` |
| FlashAttention | Preserve the Astral CUDA 12.9 wheel: `2.8.3+cu.12.9.torch.2.10`, CPython 3.12 |
| Accelerate / DeepSpeed | Preserve `1.13.0` / `0.16.4` |
| flash-linear-attention / fla-core | Add exact `0.5.2` / `0.5.2` |
| causal-conv1d | Optional; initially permit native PyTorch convolution with FLA GDN |
| Other dependencies | Resolve and record the complete lock; review transitive changes rather than assuming they are unchanged |

Initially use native `flash_attention_2` for both the text and vision configs in FullContext. Window8 changes only the text config to the registered step-attention backend. CPU reference tests may use SDPA and the Torch GDN fallback, but must assert those bindings in an explicitly CPU-only environment; they do not validate GPU FA2 or FLA execution.

The repository already records the appropriate Torch/CUDA-specific FlashAttention wheel; changing Transformers or adding FLA does not by itself require rebuilding it. Preserve the existing package indexes and verify the selected artifact's hash and runtime import. For Linux x86_64, the audited lock records FlashAttention wheel SHA256 `fa278650341f2171a1e4b85643ae5b664f321fda9bf04b2ec8be850084e5218c`. This identifies the intended artifact, not a successful kernel execution. [Locked FlashAttention artifact](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/uv.lock#L146-L156), [Torch, torchvision and Triton lock entries](https://github.com/anhdao69/SimpleMemVLN/blob/fb039150254af66a97dd2e8591d10a35a2a9d055/uv.lock#L824-L911)

FLA 0.5.2's CUDA dependency declarations require Torch at least 2.7 and Triton at least 3.3; this candidate satisfies those lower bounds. The imported GDN and gated-normalization interfaces also match the audited Transformers source. Neither finding certifies binary compatibility, Triton compilation, numerical accuracy, backward correctness or distributed integration. Require import, real H100 forward/backward, continuation, and distributed loss/gradient gates before accepting the stack. Record the actual interpreter/import paths, package and artifact hashes, driver/toolkit/runtime versions, autocast policy and complete resolved lock. [FLA dependency declarations](https://github.com/fla-org/flash-linear-attention/blob/9c8e42e762fce087c27b673af4922795d9edb85e/pyproject.toml#L17-L35), [Transformers FLA imports](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L66-L76)

Set the intended process-local CUDA device **before `from_pretrained`**. In the audited Qwen implementation, FLA's gated RMS normalization is constructed with `device=torch.cuda.current_device()`. Under ordinary torchrun with all local GPUs visible, use `LOCAL_RANK`; if the launcher exposes one GPU per process, use that process's visible index 0. Log the visible-device mapping and selected device. [Native constructor](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L406-L415)

Assert that **every GDN layer** binds its chunk and recurrent functions to the intended FLA implementations; record the convolution functions and normalization class as well. The native code chooses these independently: missing causal-conv1d leaves FLA GDN active while convolution uses PyTorch. Its aggregate fast-path warning can therefore appear in this deliberately mixed configuration. Accept this configuration only after its multi-token and single-token continuation, backward and performance gates pass; add a pinned causal-conv1d build later if profiling justifies it. A missing FLA binding is a production-gate failure because the Torch chunk fallback executes Python loops across 64-token chunks. [Kernel selection](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L420-L430), [mixed convolution and GDN execution](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L469-L550), [Torch chunk fallback](https://github.com/huggingface/transformers/blob/e7b5b964e6f64923f2770208178f3ed367978895/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L246-L324)

Run the environment and continuation gates in both the actual training interpreter and the interpreter that executes the model during evaluation. If Habitat cannot share this validated stack, keep the simulator in a separate process/environment and exchange lossless RGB plus explicit episode/step IDs; the model service owns Qwen and its kernels. Re-time Uniform8/SW4 and the new model with recorded, comparable runtime settings. Absence of FLA in an old repository lock does not prove which kernels historical runs used.

### 12.2 First runtime gate: native cached continuation without eviction

Before custom attention, a classifier pilot, or full training, compare native full-sequence and cached multi-token execution. Use the actual production FP32 cache factory, not a cache repaired after the first prefill. Resolve the real training and serving interpreter versions independently and save those runtime reports.

```python
# Proposed acceptance-test skeleton, not a claim that this test has been run.
# E: identical prepared embeddings; P: explicit [4, 1, N] positions.
# spans: complete contiguous coverage: prefix, image/cue blocks, and B token chunks.
model.eval()
lm = model.model.language_model
with torch.inference_mode():
    reference = lm(
        inputs_embeds=E,
        position_ids=P,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
    ).last_hidden_state
    cache = make_stream_cache_fp32(model.config.text_config)
    chunks = []
    for start, stop in spans:
        result = lm(
            inputs_embeds=E[:, start:stop],
            position_ids=P[..., start:stop],
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
        )
        chunks.append(result.last_hidden_state)
    streamed = torch.cat(chunks, dim=1)
    torch.testing.assert_close(streamed, reference, atol=atol, rtol=rtol)
```

Preconditions: batch one; no padding/packing; no eviction in either branch; all relevant dropout disabled; identical embeddings and positions; declared tolerances. Include multi-token continuation lengths greater than one and mixed single-token/multi-token B continuations. Log max/RMS error, decision/logit agreement, and cache state dtypes; run on three real episodes of differing lengths after a small synthetic check. Also test a native GDN block independently so full-attention history cannot hide a recurrent-state reset. A deliberately reset-state control should differ on a constructed state-sensitive input.

The second review reports four CPU tests, but their Python file and logs were not present in the received attachment. Implement or obtain those tests and inspect their actual backend/configuration before citing them as passed. The 5.11 and 5.3 cache classes differ, so a cross-version test must select each version's API deliberately. CPU fallback continuation does not certify FLA/FA2 BF16 on H100; no GPU execution of this gate is claimed here.

Use a declared numerical error budget rather than demanding exactly constant error as more blocks are appended. Report per-step maximum/RMS logit differences and top-two margins. If the reference top-two margin exceeds twice the maximum absolute logit error, argmax should agree; investigate violations. Report all near-tie flips as well, since an untrained four-class head can flip within BF16 tolerance. Neither 100% argmax agreement alone nor a single near-tie disagreement determines cache correctness.

Check that native FullContext FlashAttention does not mistakenly take the packed-sequence path. Preserve the contiguous logical text axis while retaining native MRoPE geometry. Next, test Window8 against a Window8 offline reference; do not expect equality with FullContext after eviction.

### 12.3 Window8 bounded inference does not imply bounded training

Full-episode activations still grow with episode length. For illustration, 60 observations × 300 visual tokens is already about 18k tokens before text overhead. One BF16 hidden tensor at width 2560 is about 88 MiB at that length; 32 checkpoint inputs alone are about 2.75 GiB. This is a calculation, not a measured peak.

Materializing logits for all 18k positions across a 248,320-token vocabulary would cost about 8.33 GiB in BF16. **Neither option needs that allocation:** A projects only its `T` read states into four logits; B projects only its supervised response prediction states into vocabulary logits. B may additionally chunk those sparse projections if needed. Do not justify a mandatory classifier using an avoidable full-sequence LM-logit allocation.

The per-step Flash implementation also creates repeated gathered K/V tensors. A 3,072-key rectangle has roughly 12 MiB of BF16 K+V at 4 KV heads and dimension 256 for one layer, before other tensors. Checkpointing and gather placement affect the peak. Neither KV eviction nor gradient accumulation makes the activation peak of one long episode constant.

### 12.4 Resource gate and OOM policy

Profile the actual pretrained model with optimizer state allocated, BF16, ZeRO-2, and the planned checkpointing. Run at least two optimizer updates and include short, median, p95, and longest encoded episodes **for each memory/output configuration that will be trained**. Record peak allocated/reserved memory, forward/backward time, valid actions/s, tokens/s, and actual attention/GDN backend. For FullContext deployment, also profile continuation up to the evaluator's 500-step limit; training length 183 is insufficient evidence for deployment sizing.

The external review's 55–65 GB peak, 4–6 hours per epoch on four H100s, and a few milliseconds of added attention are unmeasured estimates. Do not put them into a job reservation or a performance claim as verified values. New-token counts reduce repeated image/MLP work, but do not imply the same wall-clock speedup when the retained attention history grows. Estimate a run budget only after representative measured throughput, including data loading and the long tail.

If a complete episode does not fit, preserve full-gradient semantics in this order:

1. Keep one episode/rank, frozen vision, smaller visual microbatches, and non-reentrant decoder checkpointing.
2. Checkpoint the **gather plus FlashAttention call** within each step if repeated gathered K/V dominates memory. Verify output and gradient parity.
3. Chunk the tokenwise MLP with suitable checkpointing if its intermediate activations dominate. This is an implementation optimization, not temporal truncation.
4. Add validated saved-activation CPU offload for long-tail episodes if needed; measure its throughput cost.
5. If still infeasible, stop the full-gradient run with the offending episode/token count. A lower visual-resolution configuration applied consistently to train/eval or an explicit TBPTT milestone is a separate declared change.

Never silently drop long episodes, detach state, crop trajectories, or assume increasing gradient accumulation fixes an individual-episode OOM. Do not retry one failed rank after others have advanced the optimizer. Abort the incomplete distributed update and resume from a valid boundary with a verified configuration.

No source audit can guarantee that the longest episode fits an 80 GB GPU. The proposed multi-H100 setup is a starting point for profiling, not a resource guarantee.

If FullContext full-episode training is too expensive, keep its successful short/native continuation checks as the integration reference and proceed with the separately declared Window8 training path. Do not make a complete FullContext training run a hidden prerequisite for the bounded-memory project. Report the resulting experiment honestly rather than labeling it a full-attention reproduction.

### 12.5 If TBPTT becomes necessary later

TBPTT can still supervise every action, but it cuts gradient credit across segment boundaries. It requires an explicit functional GDN/conv/KV state interface, detachment of all relevant state at those boundaries, and a declared optimizer-update schedule. For the proposed episode-based variant, keep weights fixed until the episode's segment losses have been accumulated.

Do not turn on `use_cache=True` inside checkpointed training and assume that creates correct differentiable streaming. In-place cache writes and recomputation can conflict; checkpoint wrappers may also alter cache handling. An inference-cache burn-in followed by a checkpointed suffix is not an already-validated fallback. Record `gradient_mode` and segment length in any future TBPTT checkpoint and report.

### 12.6 Stability checks

Use correct loss normalization, BF16, finite-value checks, and global gradient clipping at norm 1.0. Log pre-clipping norm, clipping frequency, selected head/body gradient norms, and recurrent-state norms during the smoke phase. A finite loss alone does not prove correct gradients.

From the first overfit run, log the confusion matrix, per-class accuracy/recall, macro accuracy, and STOP precision/recall. With roughly 1.7% STOP in the audited labels, >95% overall accuracy is not sufficient to establish usable stopping. Inspect the decision before the true final STOP and distinguish false STOP from failure to stop. B additionally reports exact full-response validity and action-switch/action-repeat diagnostics.

Investigate normalization, state dtype, kernel backward, and optimizer behavior separately if NaN/Inf appears. Full-episode training can be stable, but GDN's structure does not guarantee stability of the entire fine-tuned stack. No distillation loss or auxiliary objective is needed before the behavioral-cloning baseline passes these checks.

## 13. Inference API and VLN evaluation

Use a common external interface:

```python
reset(episode_uid: str, instruction: str) -> None
observe(episode_uid: str, step_id: int, rgb_uint8_hwc) -> ActionResult
```

`ActionResult` contains the model class ID, canonical action name, mapped Habitat action ID, processed step ID, model timing, retained KV length, and status. A may include four logits. B may include response token IDs/text and decode-token count; do not invent a four-logit distribution from arbitrary token probabilities.

Port the environment setup and metric loop, not the complete old memory/session implementation. Verify the current evaluator's camera and action settings before comparison: RGB 640×480, HFOV 79°, forward step 0.25 m, turn 15°, maximum 500 steps, and success distance 3 m in the audited setup. Preserve the benchmark protocol or label any change explicitly. [StageVLN evaluator](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/src/qwen_vl/eval/habitat_r2r.py)

Log predicted and executed actions separately. If the evaluator forces STOP at the step limit, record a forced-stop event; it is not a model-predicted STOP. The episode ends there, so do not continue B with a history claiming that an unexecuted action occurred. Kernel/session failures and invalid B generations stay in the evaluation denominator and are reported explicitly.

For an inference-failure episode, set SR and SPL contributions to zero even if a cleanup STOP occurs within the success radius. Their denominator is the complete scheduled evaluation set. Compute navigation error, trajectory length, and nDTW from the last valid partial trajectory where defined; report metric coverage and undefined counts explicitly. Do not impute zero navigation error or silently omit failed episodes. Apply this system-failure convention consistently to both options and record it alongside the simulator's raw metrics.

Report SR, SPL, navigation error, trajectory length, and oracle success/nDTW where the existing metric implementation supports them. Also report action distribution, STOP behavior, invalid-response/termination failures, forced stops, cache bounds, model latency p50/p95, and peak memory. For B, include all decoding and final-commit work in decision latency. Separate simulator rendering, environment stepping, transport, and model computation.

### Historical comparison anchors and confounds

The audited full val_unseen summaries contain 1,839 episodes: SW4 has 655 successes, SR 35.6172% and SPL 30.1788%; Uniform8 has 807 successes, SR 43.8825% and SPL 39.1154%. Use the exact checkpoint/evaluation IDs rather than similarly named results from another experiment. These are useful performance references, not thresholds proving software correctness. [SW4 summary](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/evaluation/r2r_v1_sw4/StageVLN-v1-r2r-sw4/summary.json), [Uniform8 summary](https://github.com/anhdao69/StageVLN-v2/blob/f0641ffdef8bcd7cd70aed9ed9fc2b1b5a4e5403/evaluation/r2r_v0_uniform8/StageVLN-v0-r2r-uniform8/summary.json)

| Factor | New initial plan | Historical evidence / comparison requirement |
|---|---|---|
| Visual merger | Frozen | Prior launcher and v3 saved recipe train it; recover each evaluated checkpoint's actual flags |
| Text LR | Proposed `5e-6` | Prior launcher and v3 saved recipe use `1e-6`; do not treat launcher defaults as proof for every checkpoint |
| Training budget | One full-data epoch first | Report epoch-1 result, actual actions consumed, optimizer updates, and any extension |
| Objective batch | Nominal eight episodes, about 467 actions/full update; native partial tail logged | Actual SW4 metadata records 64 action states/update |
| History/readout | FullContext or Window8; cue classifier or committed-action LM | Different from Uniform8/SW4 prompts and from the old learned-memory writer |
| Invalid response | Explicit failed episode | Old evaluator maps invalid responses to STOP; the audited complete runs recorded zero invalid responses, so that fallback does not explain their observed gaps |
| Attention/runtime | Require verified 5.11 native continuation and selected FA2/FLA paths | An old launcher's environment name or package pin is not proof of the live runtime |

Actual SW4 metadata verifies the dataset identity and batch decomposition but omits some optimizer/epoch fields; the saved v3 epoch manifest contains a fuller executed recipe. Preserve the distinction between intended launcher flags and run evidence. [SW4 saved metadata](https://huggingface.co/anhdao69/StageVLN-v1-r2r-sw4/blob/e584d496a437c3fb88100afd6ac2656864eebeef/training_metadata.json), [v3 completed epoch manifest](https://huggingface.co/anhdao69/StageVLN-v3_mem64_r4-r2r/blob/5c645ffdce72118866f43f6b1c45f5d072f3adf4/training_state/epoch-1/manifest.json)

Do not infer a universal STOP root cause from a rough count near 780. The saved runs have different failure distributions, including long loops and forced termination. Report voluntary STOP before ever entering the goal region, STOP after leaving it, forced stops, and unsuccessful capped trajectories separately; an outcome category does not by itself identify the mechanism causing it.

## 14. Code organization and interfaces in the audited SimpleMemVLN repository

Implement inside the existing `src/qwen_vl` package at the audited SimpleMemVLN `fb03915` revision. Keep `qwen_vl.train.train_qwen`, `QwenSFTTrainer`, and `train.sh` as the entry point, Trainer integration, and launcher. Adapt the parts whose contracts change; do not create a second training framework or require a rewrite of working utilities. SimpleMemVLA remains a methodological reference; StageVLN-v2 is the attributed source for selected episode/evaluation utilities.

The table distinguishes files already present from proposed additions. A listed new module is a responsibility boundary, not a requirement to split a small implementation into more files than it needs.

| File or group | Status at the audited base | Planned responsibility |
|---|---|---|
| `pyproject.toml`, `uv.lock` | Existing; modify | Retain the compatible repository stack, update the required Qwen/FLA dependencies, and record the lock that passes actual runtime gates. |
| `train.sh`, `deepspeed.json` | Existing; modify | Launch the existing module with episode settings; remove Uniform8-only assumptions; make accumulation, ZeRO, warmup, memory mode, and output mode explicit. |
| `src/qwen_vl/train/train_qwen.py` | Existing; extend | Resolve CLI/recipe arguments, select the CUDA device before model construction, load pretrained Qwen and the selected navigation wrapper, build episode data, and run `QwenSFTTrainer`. |
| `src/qwen_vl/train/argument.py` | Existing; extend | Declare the episode, output, memory, processor, and classifier arguments; reject conflicting contracts. Preserve supported ordinary HF CLI arguments. |
| `src/qwen_vl/train/trainer.py` | Existing; extend | Global action-mean A/B loss, explicit optimizer groups, metadata handling, compact logging, and matching save/load/resume integration. |
| `src/qwen_vl/train/sampler.py` | Existing; adapt | Bucket using actual encoded episode lengths; preserve chronology, the declared tail policy, coverage accounting, and deterministic resume. |
| `src/qwen_vl/data/data_qwen.py` | Existing; selectively reuse | Reuse and test the fixed non-thinking template and relevant image/token helpers. The Uniform8 dataset/collator is not the streaming episode implementation. |
| `src/qwen_vl/contracts.py` | Proposed; port/adapt | Versioned navigation configuration, action class/string/Habitat mapping, output and stream contracts. |
| `src/qwen_vl/data/episode_manifest.py` | Proposed; port/adapt | Source identity, instruction/scene/split metadata, ordered observations, label validity, and final STOP checks. |
| `src/qwen_vl/data/episode_serializer.py`, `episode_dataset.py` | Proposed | Shared immutable fragments and one complete episode per item; exact token lengths, target/read positions, native image metadata, and a batch-one unpadded collator. |
| `src/qwen_vl/models/nav_model.py` | Proposed | Thin native Qwen bridge, frozen visual feature extraction, trainable text embeddings, A cue-end head or B sparse LM projection, and an HF-compatible forward contract. No copied text-model forward. |
| `src/qwen_vl/stream/cache.py`, `positions.py` | Proposed | FP32 recurrent storage, whole-step KV FIFO when selected, fresh reset, logical/MRoPE position ledger, and invariant checks. |
| `src/qwen_vl/stream/window_attention.py` | Proposed | Registered text-only Window8 gather-and-FA2 callback; immutable span metadata; dense reference used only in short correctness tests. |
| `src/qwen_vl/stream/session.py` | Proposed | Shared reset/observe interface, A readout, B decode/commit, bounded metadata, and request retry/failure semantics. |
| `src/qwen_vl/eval/habitat_r2r.py` | Proposed; port/adapt | Existing benchmark protocol, episode metrics and failures, trajectory export, and resume accounting. Port the environment loop, not the old model session. |
| `src/qwen_vl/stream/server.py` | Optional new file | Simulator/model process separation only if environment compatibility requires it. |
| `configs/vln_r2r_v0_base.yaml`, `vln_r2r_v0_classification.yaml`, `vln_r2r_v0_qwen_text.yaml`, `vln_memory_window8.yaml` | Proposed recipe files | Shared, output-specific, and independent memory-mode presets from Section 11; resolve into the existing argument dataclasses. |
| `scripts/vln/audit_data.py`, `check_compat.py`, `check_parity.py`, `profile_episode.py` | Proposed tools | Data/token audit, actual-runtime/kernel tests, offline/streaming parity, and measured full-gradient resource profiles. |
| `tests/vln/` | Proposed tests | The semantic and integration checks in Section 16, including Trainer reduction, tail handling, save/load, and resume. |

Add a small recipe-resolution step to the existing entry point: proposed `--vln_config` selects the shared recipe, `--output_config` selects A or B, and optional `--memory_config` selects Window8. These flags and YAML presets do not exist merely because they are specified here. Resolve base, output, and memory recipes in that order, then explicit CLI overrides; validate the result and save it. Continue to support ordinary CLI invocation. Use the existing `--deepspeed` path rather than inventing a second optimizer launcher.

The initial training admission candidates are 65,536 tokens for A and 73,728 for B, subject to the actual serializer audit and resource gate. They are separate from the 262,144-token logical serving ceiling and Window8's resident-KV bounds. Select native FA2 for vision and FullContext text initially; Window8 changes only the text backend. Keep the declared training seed 429 and evaluation protocol seed 42 distinct.

Keep dataset/serialization code independent of CUDA model construction and the simulator independent of Qwen imports. Cache code does not compute training loss. Both offline and streaming execution call the same immutable fragment builders: A uses `vln_observation_stream_v3`, with a fixed post-image cue and no textual `Frame t:` header; B uses `vln_append_only_chat_v1`, based on the audited repository's fixed template, with its deterministic newline masked from the objective.

### Shared data and model interfaces

```python
@dataclass(frozen=True)
class EpisodeTokens:
    input_ids: Tensor
    mm_token_type_ids: Tensor
    image_grid_thw: Tensor
    image_inputs: object
    prefix_span: tuple[int, int]
    step_spans: tuple[tuple[int, int], ...]  # Half-open; B includes full turns.
    action_class_ids: Tensor
    read_positions: Tensor | None          # A: final fixed cue token.
    response_target_positions: Tensor | None  # B: J, not J-1.
    response_action_index: Tensor | None   # Each response target -> action.
    num_actions: int
    loss_weight: float                    # As defined by the chosen tail policy.

class EpisodeSerializer:
    def encode_episode(self, episode, output_mode) -> EpisodeTokens: ...
    def encode_prefix(self, instruction, output_mode) -> PrefixBlock: ...
    def encode_observation(self, rgb, step_id, output_mode) -> ObservationBlock: ...

class SimpleMemVLNForNavigation(nn.Module):
    def forward(self, **explicit_episode_fields) -> EpisodeOutput: ...
    # Interface sketch: implement an explicit accepted field contract.
    # Return a sum of per-action losses and action count, plus optional
    # compact diagnostics. Never require dense vocabulary logits at all tokens.

class StreamSession:
    def reset(self, episode_uid: str, instruction: str) -> None: ...
    def observe(self, episode_uid: str, step_id: int, rgb) -> ActionResult: ...
```

The collator converts episode records into the named tensors/metadata expected by the wrapper and Trainer. Do not let HF's column-removal logic discard `read_positions`, action mappings, `num_actions`, or span metadata. For the initial custom wrapper, set `remove_unused_columns=False` and explicitly consume Trainer-only metadata before forwarding to the native backbone. A future use of automatic removal requires a tested explicit forward signature. Do not pass arbitrary fields into native Qwen just because the wrapper accepts `**kwargs`.

Only frozen visual extraction runs under `no_grad()`. Text embedding lookup and insertion of detached image features must preserve the text embedding graph. The bridge supplies the four-axis position ledger and routes span metadata separately; it does not pass invented mask dictionaries through native HF forward. A never receives the current expert action. B targets appear only under its causal-shift contract.

Preserve the session contract from Section 13: one episode and one in-flight request; append each fragment once; commit B's final EOS and separator; replay a completed request by returning its saved result; invalidate the session after a partial mutating failure. Reset creates fresh cache/state rather than merely zeroing old tensors.

If checkpoint saving writes a backbone directory, a separate classifier, and navigation metadata, implement the matching load path for serving **and** HF Trainer resume. A custom `_save` method alone does not make ordinary Trainer resume work. Restore the trained classifier rather than reinitializing it; validate mode/serializer/codec/cache contracts; preserve optimizer, scheduler, scaler where applicable, RNG, sampler order, and update cursor. Test restart at an optimizer boundary against uninterrupted execution.

## 15. Implementation dependencies and completion gates

The first working configuration is **FullContext + A**. The bounded target is **Window8 + A**; B remains a supported implementation deliverable with separate acceptance gates. The dependencies below do not require a full FullContext training campaign before Window8, or require B before the first complete A training run.

### Gate R — repository integration and native continuation

- [ ] Start from the audited SimpleMemVLN revision in an isolated branch/worktree; preserve unrelated user changes and the existing `qwen_vl` entry point.
- [ ] Extend its environment and argument handling. Save the exact training and serving runtime versions/builds, select the local CUDA device before model loading, and verify which GDN and attention kernels actually execute.
- [ ] Install the FP32 recurrent-cache factory before prefill. Implement the independent logical/MRoPE ledger and fresh episode reset.
- [ ] Pass synthetic no-eviction full-versus-blocked continuation, an isolated GDN-block test, mixed multi-token/single-token updates, recurrent dtype checks, and native FA routing checks. Repeat on the actual GPU/FLA path; CPU fallback tests alone are insufficient.

**Gate:** The native model preserves the specified state across append operations with declared numerical tolerances. The review's claimed CPU test results are not local evidence until the test code is obtained or independently implemented and run.

### Gate D — episode data and the FullContext+A reference

Requires R for model checks; manifest work can proceed independently.

- [ ] Audit the actual manifest: capture-before-action alignment, final STOP, identity/splits, exact encoded lengths for each selected serializer, prefix/step limits, and complete-episode cap. Reject overlength examples explicitly; do not truncate them.
- [ ] Implement the shared serializer and episode collator. For A, use the fixed cue without a textual frame index; verify the final-normalized read position and Habitat mapping. Prepare B's codec/fragment metadata without requiring its decoder yet.
- [ ] Add the randomly initialized A head and FullContext session. Read the final hidden tensor directly, rather than retaining every layer's hidden states.
- [ ] Complete the real-image position/continuation checks and offline-versus-streaming comparisons on three episodes of differing lengths. Compare logits under declared tolerances and report margins and action agreement. Near-tie argmax changes require inspection, not an automatic diagnosis of broken caching.
- [ ] Verify target-free A inputs, unchanged earlier predictions under future-input changes, bounded request bookkeeping, and correct reset/retry/failure behavior.

**Gate:** Data, positions, readout, and native continuation agree on the same immutable token stream. Navigation competence is not required at this point.

### Gate T — existing HF Trainer and evaluator integration

Requires D's serialized episode contract.

- [ ] Adapt `QwenSFTTrainer` to count actions across the actual accumulation window. A sums step CE; B first averages tokens within each action and then sums actions. Use exactly one distributed normalization path as specified in Section 11.
- [ ] Preserve episode metadata through the collator/Trainer boundary; strip fields the native backbone does not accept. Verify explicit classifier LR grouping and tied-parameter deduplication.
- [ ] Adapt encoded-length bucketing and the declared tail policy. Compare a two-rank variable-length update with a single-process reference, including the final partial accumulation window, duplicates/zero weights where applicable, and exact coverage accounting.
- [ ] For the default native Accelerate tail, count repeated episodes as real exposures in the objective and report them. In the audited 10,819-episode, four-rank/GAS-2 setup, verify the expected one repeated episode and final four exposures rather than assuming a complete eight-episode final update. Test actual partial-window normalization; an optional exact-once sampler is a separate change.
- [ ] Verify frozen visual/merger behavior after `train()`, nonzero intended text/head gradients, native GDN backward, and checkpointed-versus-uncheckpointed gradients on a small episode. A gradient reaching an earlier activation is required for temporal credit; shared parameter gradients alone do not establish it.
- [ ] Implement matched save/load/resume and verify the next loss/update after an optimizer-boundary restart. Preserve the sampler order and accounting.
- [ ] Port the Habitat loop and common action interface. Run a small end-to-end rollout check with complete failure accounting; this checks execution, not navigation accuracy.

**Gate:** The reused training stack implements the intended objective and lifecycle. Imports, a finite loss, or an optimizer coverage assertion alone do not satisfy this gate.

### Gate L — resource profile and a small FullContext+A learning run

Requires R, D, and T.

- [ ] Profile short/median/p95/longest encoded episodes with optimizer state allocated, plus long inference up to the declared evaluation horizon. Keep training and serving limits separate.
- [ ] If feasible, overfit 8–16 complete episodes with an explicit debug repeat/update budget, then run a 512-episode pilot and approximately 20 fixed simulator rollouts. Inspect per-class metrics, STOP precision/recall, gradients, and resource measurements; overall action accuracy alone is insufficient.
- [ ] If long FullContext training is too expensive, retain successful short/reference tests and move to Gate W. Complete the learning/profile gates on Window8 instead. Do not require a full epoch, full validation campaign, CPU-offload implementation, or successful longest-episode FullContext training merely to begin Window8.

**Gate:** A measured configuration learns the small set and performs rollouts before its full-data run. A full FullContext training campaign is optional.

### Gate W — Window8 bounded streaming

Requires R, D, and T and the FullContext numerical reference; normally follows the small FullContext pilot, with the cost exception above.

- [ ] Register the text-only Window8 callback and verify immutable span metadata survives gradient checkpointing without constructing a dense episode mask.
- [ ] Evict complete expired step groups before the new step's first append, at full-attention layers only. Preserve GDN/conv state and absolute positions; enforce KV and metadata bounds.
- [ ] Compare direct FA2 outputs and Q/K/V gradients with a short explicit-mask oracle, including GQA and variable groups. Cover response-containing groups when B is added.
- [ ] Compare Window8 offline and streaming execution beyond eviction under its own visibility rule. FullContext is not the expected output after eviction.
- [ ] Profile and complete the overfit/pilot/rollout learning gates for Window8+A. Start from the same pretrained Qwen for a controlled comparison; label any FullContext warm-start as a separate transfer experiment.

**Gate:** The bounded policy is numerically consistent with its own offline computation and has measured learning/runtime behavior. Start with W=8; W=16/32 sweeps and memory-attribution interventions are later experiments.

### Gate F — first complete A training run

Requires the correctness, resource, and learning gates for the selected memory mode. It does **not** require B. A FullContext run requires R/D/T/L; a Window8 run requires R/D/T/W and the completed small-set learning gate.

- [ ] Select the first full-data configuration from measured pilot evidence; run one epoch and the complete fixed val_unseen evaluation before deciding whether to extend training.
- [ ] Record actual labels/episodes consumed, optimizer updates, repeated/zero-weight tail slots, checkpoint provenance, failures, STOP behavior, latency by trajectory age, and measured memory.
- [ ] Compare historical SW4/Uniform8 results with the confound table and exact evaluation identities. Re-time baselines under a matched runtime before making efficiency claims. Their SR is a performance reference, not a software-correctness gate.
- [ ] A matched FullContext-versus-Window8 campaign is useful when affordable; it is not a prerequisite for reporting the first verified run.

**Gate:** The selected A policy has a reproducible full-data result with honest resource and failure accounting. This does not establish GDN-specific long-term memory or guarantee strong SR.

### Gate B — existing Qwen LM-head output

This branch requires D/T's shared interfaces and the reference for its selected memory mode; Window8 B additionally requires W's attention/cache implementation. It can proceed after the first A result or in parallel with that training run once the shared contracts are stable.

- [ ] Implement B's immutable action/closure fragments and sparse `J-1 → J` objective, averaging response-token CE inside each action. Keep the fixed separator masked and use the pretrained LM-head LR.
- [ ] Add bounded greedy decoding, strict parsing, and once-only commitment of every body token, EOS, and newline. Invalid output remains an explicit failed episode.
- [ ] Test forced-token and generated-history replay with identical committed IDs, mixed one-token/multi-token cache updates, and no eviction inside the current response group.
- [ ] Repeat the distributed loss/tail check with unequal response lengths, resource profile, complete-episode overfit/pilot, and rollouts. Do not require perfect untrained response validity to test the decoder.
- [ ] Report copy/switch/STOP diagnostics and all response decode/commit time. A-versus-B remains an interface/history comparison, not a pure output-head ablation.

**Gate:** Both requested output interfaces are implemented and validated under their declared contracts. B's validation is required for B delivery or results; it does not delay the first full-data A run.

## 16. Required correctness tests

These tests address concrete failure modes in the new semantics; they are not intended to duplicate every line of implementation.

| Test | What must be demonstrated |
|---|---|
| Native continuation regression | Actual train/serving runtimes carry multi-token GDN/conv state; test no-eviction full versus chunked execution, then the GDN block separately. |
| Native Flash routing | Contiguous logical text positions do not trigger unintended packing; native MRoPE geometry is preserved. |
| Observation/action alignment | First, turning, and terminal observations correspond to actions about to execute. |
| Serializer stability | The fixed repository template and approved fragments preserve earlier IDs, images and positions; if testing a native-template substitution, include its historical-assistant trap. |
| A target alignment | One label per fixed cue-end read position, no shift, no current gold action in the input; cue tokens count toward the step budget. |
| B target alignment | First body token, later body tokens, and assistant EOS use `J-1 → J`; fixed scaffolding/separator is masked; no double shift. |
| Causality | Changing future images or future actions leaves earlier predictions unchanged; in B, changing target `x_j` cannot change its predictor at `j-1`. Include preprocessing. |
| Attention rectangle and registration | Window8 FA2 output and Q/K/V gradients match the short explicit-mask reference; text-only registration, kwargs/checkpoint metadata, and no dense-mask construction are verified. |
| Window boundary | At step 8 and beyond, expired groups are removed before append; B decoding stays within its current group. |
| Position ledger | Blockwise native MRoPE and logical positions match the same full token stream; eviction never renumbers surviving keys. |
| Recurrent dtype and backend | All GDN cache matrices remain FP32 through prefix, image blocks and single-token decode; record internal autocast/backend policy and conv dtype, not only storage dtype. |
| Conditional parity | A readouts and B supervised-token logits match offline versus streaming execution under the same selected memory rule and committed IDs; Window8 covers eviction. |
| Token commitment | Controlled one- and multi-token responses commit every body token, EOS, and separator once; next observation never replays them. |
| Reset/retry/failure | Fresh episode equals fresh session; a completed retry leaves state unchanged; a partial failure invalidates the session. |
| Gradient history | Later loss reaches a chosen earlier source activation. Shared parameter gradients alone do not prove temporal credit; isolate GDN if claiming GDN-specific influence. |
| Loss reduction | The retained custom Trainer matches a global action-mean gradient/update reference with unequal episodes, unequal B response lengths, GAS>1 and the partial final update. |
| Length buckets and coverage | One global sampler plus one Accelerate sharding path covers all complete episodes; logged unique/repeated counts, partial-tail collectives and deterministic resume agree. No per-pool padding or silent drops. |
| Checkpoint and resume | Checkpointed versus uncheckpointed small-episode gradients agree; loading rejects output/serializer/codec/EOS contract conflicts; resume preserves update/order accounting. |
| Memory budget | Window8 keeps state/metadata bounded and enforces the 4,096-token guard; FullContext obeys its declared growing-cache/context limits and is profiled to the evaluation horizon. |
| Imbalance and action feedback | Per-class and STOP metrics expose failures hidden by overall accuracy; B transition/copy diagnostics use actual sequence evidence, not only class frequencies. |
| Failure accounting | Invalid B outputs, cap exhaustion, forced STOP, and session failures are visible; inference failures contribute SR/SPL zero, with partial-trajectory metric coverage reported. |

Declare tolerances before performance experiments. Use tighter tolerances on short controlled reference calculations and justified BF16 tolerances on the pretrained stack; log maximum/RMS differences and action agreement. Do not demand bitwise equality between chunk and recurrent kernels or relax tolerances silently to hide a growing discrepancy. Apply the margin-aware argmax interpretation from Section 12.2 instead of an unconditional 100% agreement requirement on near-tied logits.

Parity is tested with dropout disabled. A successful source audit is not a substitute for any runtime gate above.

## 17. Planned execution commands and evidence

The following are the intended CLI contracts to implement. They are not currently available commands merely because they appear in this plan. Run from the repository root and supply the real manifest paths.

```bash
python -m scripts.vln.audit_data --manifest data/r2r_train.jsonl --output-modes classification qwen_text --out artifacts/data_audit

python -m scripts.vln.check_compat --output_config configs/vln_r2r_v0_classification.yaml --vln_config configs/vln_r2r_v0_base.yaml --native-continuation --check-flash-routing --out artifacts/compat

python -m scripts.vln.check_parity --output-mode classification --memory-mode full_context --episode-list data/parity3.json --out artifacts/parity_full

python -m scripts.vln.check_parity --output-mode classification --memory-mode window8 --steps 20 --out artifacts/parity_window8

python -m scripts.vln.check_parity --output-mode qwen_text --memory-mode window8 --forced-actions --steps 20 --out artifacts/parity_qwen

torchrun --nproc_per_node=4 -m scripts.vln.profile_episode --vln_config configs/vln_r2r_v0_base.yaml --output_config configs/vln_r2r_v0_classification.yaml --manifest data/r2r_train.jsonl --length-buckets short median p95 longest --out artifacts/profile_full

torchrun --nproc_per_node=4 -m qwen_vl.train.train_qwen --vln_config configs/vln_r2r_v0_base.yaml --output_config configs/vln_r2r_v0_classification.yaml --manifest data/r2r_train.jsonl --episode-limit 16 --debug-repeat-episodes --max-optimizer-updates 200 --run-name full_classifier_overfit16

torchrun --nproc_per_node=4 -m qwen_vl.train.train_qwen --vln_config configs/vln_r2r_v0_base.yaml --output_config configs/vln_r2r_v0_classification.yaml --memory_config configs/vln_memory_window8.yaml --manifest data/r2r_train.jsonl --episode-limit 512 --run-name window8_classifier_pilot512

python -m qwen_vl.eval.habitat_r2r --checkpoint outputs/window8_classifier_pilot512/selected --split val_unseen --episode-list data/smoke_eval20.json --out artifacts/window8_rollout_smoke
```

Run the corresponding profile/overfit/pilot commands for each selected memory/output mode, adding `--memory_config configs/vln_memory_window8.yaml` for Window8 and the Qwen-text output config for B. The thin resolver exposes `--vln_config`, `--output_config` and `--memory_config` while keeping `qwen_vl.train.train_qwen` as the entry point; these flags must be implemented before the commands work. Omit the memory override for FullContext. The `episode-limit` argument selects complete episodes deterministically and records selected IDs; it does not truncate their steps. A one-epoch default may be insufficient for deliberate overfitting of 16 episodes; provide an explicit debug-only repeat/update budget while keeping every selected episode intact. Before full-data training, remove the pilot limit/debug repeat setting and inspect the resolved one-epoch update schedule. FullContext's 512-episode pilot is optional after its resource gate; Window8 need not wait for a complete FullContext campaign.

Required evidence artifacts: resolved config/source lock, data/token audit, kernel/gradient/parity results, resource profiles, overfit and pilot curves, checkpoint provenance, and per-episode rollout metrics/failures. Record actual tests passed and measurements; do not mark a gate complete because an import succeeds or a script file exists.

## 18. What to improve after v0 works

First establish a reproducible behavioral-cloning baseline with the selected memory mode. Then change one declared factor at a time: `W`, image token budget, trainable merger/vision, decoding policy, episode curriculum, long-trajectory training strategy, or memory mechanism. Distillation and paired-history supervision are possible later methods, not requirements for the initial system.

**A versus B is not a pure output-head ablation in this plan.** They differ in read position, prompt tokens, previous-action access, teacher forcing, decode calls, and token/cache budgets. Equal `W=8` does not mean equal retained bytes: A's cap is 96 MiB of KV and B's is 112 MiB. Report these differences when comparing results. A future controlled study should match historical information and compute/token budgets where the question requires it.

A read-only LM branch that discards generated action writes would be another architecture. It requires correct state branching or restoration for GDN, conv, KV, and positions; it is intentionally deferred. Likewise, a result showing improved SR does not by itself demonstrate long-term GDN memory. Build isolated memory/causality controls before making that research claim.

The implementation target remains concrete: **pretrained Qwen3.5, complete-episode action supervision, verified native GDN continuation, a FullContext reference and a bounded Window8 policy, and both fully specified output interfaces.** The quality, speed, and research value of the bounded policy must be established by the recorded experiments.

## 19. External-review disposition

This section distinguishes the earlier methodological review from the new review of the actual repository. Source inspection and sampler checks do not certify unexecuted model/GPU tests.

### 19.1 Earlier review and decisions retained

| Review point | Assessment | Plan action |
|---|---|---|
| Transformers 5.3.0 breaks native multi-token GDN/conv continuation | Confirmed in source; no GPU reproduction performed here | Exact 5.11.0 pin in real model runtimes; first numerical gate; independent GDN-block test |
| FullContext is a simpler initial reference | Reasonable engineering recommendation | Add no-eviction bring-up and optional learning pilot; retain Window8 as the bounded target |
| 20.6 versus 88.3 proves native GDN plus Window8 is the wrong design | Incorrect transfer of a different ablation | Explain the 16-token recurrent bottleneck with four-decision BPTT and Appendix E; require direct VLN evidence |
| All information older than Window8 lives only in GDN | Too strong | Retain the contextual-KV relay caveat and isolated memory-attribution tests |
| Full history fits easily and costs only a few extra milliseconds | KV arithmetic is plausible; total resource claim unverified | Separate training/inference bounds; profile to 500 rollout steps; no promised VRAM/time |
| Option A must precede B because B is label leakage | A-first is a useful debugging choice; the universal leakage claim is false | Bring up cue-end A first; retain causal B and add copy/exposure diagnostics |
| A should use a fixed post-image cue | Valid small design choice, not a proven performance improvement | Adopt a versioned fixed `Action:` cue with no target content |
| Seed classifier rows from forward/left/right/stop embeddings | Plausible optional experiment, not required | Keep explicit random initialization; document verbalizer and logit-scale checks if compared later |
| Repository starting point | Earlier SimpleMemVLA-base assumption superseded by the newly verified SimpleMemVLN repo | Extend SimpleMemVLN `fb03915`; port selected StageVLN utilities; keep SimpleMemVLA as a methodological reference |
| Bucket episodes by length | Sound efficiency suggestion | Use encoded-length pools, one global order, rank balance and logged native-repeat tail coverage |
| R2R snapshot has 10,819 episodes and 631,244 labels | Supported by saved data/run evidence | Record snapshot counts, 1,353 updates/epoch, and distinguish future manifests |
| Capture-before-action and final STOP are already fully checked | Evidence is insufficient | Retain raw alignment and terminal-placement preflight gates |
| Remove SDPA because GQA handling is defective | No source evidence of a GQA defect; runtime parity remains untested | Defer optional SDPA for scope, not because it is known incorrect |
| Prefer `AttentionInterface.register` | Confirmed applicable to the pinned unpadded B1 contract | Use text-only callback and existing kwargs; no copied text forward by default |
| Keep the text position axis contiguous | Confirmed wrapper behavior | Separate logical text axis from native MRoPE and test Flash routing |
| Evaluate epoch 1 and report baseline recipe differences | Sound experimental practice | One-epoch first report and explicit confound/provenance table |
| Early STOP is the same dominant failure in every old run | Overgeneralized | Report exact per-run stopping/loop/forced-stop outcomes; do not assign an unproven cause |

Neither the review nor this revision demonstrates that native GDN caused the old v3 failure, that Window8 will match FullContext, or that the proposed training will fit/run at a specified speed. Those are separate, testable claims. The changed engineering order makes them easier to investigate while preserving the requested research objective.


### 19.2 New repository review: accepted and qualified points

| New review claim | Assessment | Updated implementation decision |
|---|---|---|
| Plan targets SimpleMemVLA but actual repo is SimpleMemVLN Uniform8 | Confirmed by the public `fb03915` source | Rebase the code map, loader, Trainer and launch configuration on `src/qwen_vl` |
| Transformers 5.3.0 breaks multi-token GDN carry | Source-confirmed; the reported CPU errors were not independently reproduced | Exact 5.11.0 train/serving pin and explicit multi-token/single-token GPU gates |
| No FLA in the lock causes the declared setup to use Torch GDN | Correct for an environment created from that dependency policy; live bindings still need inspection | Add pinned FLA and assert chunk/recurrent bindings on all GDN layers |
| Torch fallback always undoes FP32 cache | Overstated: working tensors are promoted before state conversion; outer autocast can change chunk arithmetic | Separate FP32 cache storage from internal arithmetic; explicit autocast reference and real FLA continuation tests |
| Historical U8/SW4 jobs therefore used the fallback | Not established by the repository lock | Recover live-run fingerprints or re-time comparable runtimes; do not invent a historical cause |
| Every full episode is rejected at 12,800 tokens | Incorrect universal claim; the cap is inadequate for the corpus's long episodes | Audit exact lengths and rejected counts; use separate A/B training admission candidates |
| 65,536 safely covers full-episode training for both A and B | Not implied by B's current conservative bound | A training candidate 65,536; B candidate 73,728 or a smaller audited value; keep serving limits separate |
| Per-row reduction becomes the wrong action weighting | Correct under the selected global-action objective | Count actions and preserve per-action token means for B |
| Keep the existing HF Trainer | Appropriate and source-supported for this exact subclass | Preserve its complete loss override/flag and single world-size factor; test gradients and partial tail |
| Preserve the Torch/FA/DeepSpeed stack | Sensible minimal-change candidate; metadata is not runtime certification | Keep existing lock choices where compatible; change HF/add FLA; run import/kernel/backward gates |
| FLA can run without causal-conv1d | Correct: GDN and convolution functions are selected independently | Permit logged Torch-conv + FLA-GDN mix; add fused convolution only if profiling justifies it |
| One repeated tail episode on four ranks | Correct for B1/GAS2/native Accelerate tail | Log 10,819 unique, 10,820 exposures, 1,353 updates, last update four exposures; eight ranks have different repeat counts |
| Encode every image in one frozen vision call | Causally valid under per-image attention, not a VRAM guarantee | Default to small visual microbatches; keep text embeddings outside `no_grad` |
| Remove A's frame-number text | Reasonable simplification, not proof of numeric generalization failure | Adopt `vln_observation_stream_v3`; keep step IDs/absolute positions as metadata |
| Require 100% argmax agreement for random-head BF16 parity | Too brittle without logit margins; insufficient by itself | Use declared numerical tolerance, margin-aware disagreement investigation and full reporting |
| Finish full-data FullContext before starting Window8; immediately sweep 8/16/32 | Unnecessary expansion of the initial implementation | Native correctness and a small learning pilot first; Window8 can precede a full FullContext campaign |
| Four passing CPU tests are attached | Only the prose checklist was received | Treat numbers as reviewer-reported; implement/obtain and run tests before claiming pass |
| Appendix E's 60-unit / 5,888-token details remain unconfirmed | Available explicitly in the primary HTML text | Cite the appendix while keeping its video units separate from VLN step windows |

**Chosen first configuration:** pretrained Qwen3.5-4B, FullContext + A, every action supervised in one differentiable episode forward, frozen vision/merger, native FA2 and verified FLA, correct HF Trainer global-action loss. **Next bounded configuration:** Window8 + A trained with its own matching visibility rule. B is an independent supported output option; its implementation does not block the first validated A full-data run. The best SR/window/output interface remains an experimental result, not something established by this review.
