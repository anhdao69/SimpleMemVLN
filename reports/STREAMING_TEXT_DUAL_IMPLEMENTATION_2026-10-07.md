# Window8 text dual-lane implementation and training qualification

## Scope and provenance

This work implements the user's selected fresh-base, two-epoch joint R2R/RxR
text-output experiment on branch `streaming_text_dual`. Its parent is the actual
GitHub `streaming_logits_dual` tip `6b0f30a`, four commits ahead of the local
`fe9914d` checkout. Those intervening commits update evaluation, transport and
reporting. They were reviewed and included; the old worktrees were not pulled
over or edited. The new local worktree is
`/Users/hoanganh692004/Desktop/SimpleMemVLN-streaming_text_dual`; the server
worktree is `/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN-streaming_text_dual`.

The reviewed sources include `plans/dual_lane.md`,
`plans/implementation_plan_dual_lane.md`, the detailed dual-lane implementation
report, the consolidated October 6 evaluation report, historical Window8 text
resource/recovery reports, and the live implementation of serialization,
streaming, attention, loss, optimizer grouping and checkpoint recovery.

The consolidated evaluation report records 52.20% R2R SR for the existing
Window8 text epoch-2 baseline and mixed results for FullContext candidate
dual-lane policies. Those are checkpoint comparisons, not proof that this new
text/lane combination improves navigation. No SR/SPL result is claimed here.

## Model and causal behavior

The earlier dual-lane implementation already supports text output at the
configuration, serialization and session levels. Its executed real-model gates
primarily qualified candidate no-history policies. This branch adds an explicit
text production recipe and text-specific real-model qualification rather than
inventing a second lane architecture or replacing its pretrained namespaces.

The pinned backbone is Qwen3.5-4B revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Vision remains frozen. Four parallel
lanes at zero-based layers 16, 20, 24 and 28 add 21,054,000 parameters. Each lane
reads the same normalized hidden stream as its native linear-attention mixer.
The residual/MLP and native parameter names are retained. Lane output projection
starts at zero; all recurrent runtime states begin at zero for each episode.

For each layer's key-by-value state R, only the final token of a completed
observation/action group enables decay and delta-rule correction. Prefix,
observation and ordinary response-token roles remain nonwriters. The update is
`R' = exp(g) R + beta k (v - (exp(g) R)^T k)^T`, followed by the query read.
Queries, gates and the exact normalization follow the inherited implementation.
The lane reads at token rate and writes at step rate; there is no learned initial
episode state or new token. Four FP32 state banks total 1 MiB per session.

Text uses `vln_append_only_chat_v1`, autoregressive canonical action strings and
assistant EOS supervision. Generated ordinary action tokens are appended with
FEEDBACK roles. EOS plus the existing separator is appended last; only its final
token has STEP_END role. This preserves action history, as explicitly described
in the user's selected experiment. A group's write occurs after its decision.
Training averages response-token CE within each action and normalizes over the
global action count across ranks and accumulation slots. There is no candidate
four-row head, class reweighting, auxiliary loss or trajectory truncation.

Window8 retains the instruction prefix and the current plus preceding seven
complete observation/feedback groups in native full-attention KV. Native GDN,
lane states and logical multimodal positions persist through eviction. They
reset only at episode boundaries. Training remains a complete differentiable
episode, with decoder checkpointing and CPU activation offload above 65,536
tokens; it is not truncated BPTT.

## Bounded attention batching

The original Window8 implementation calls FlashAttention separately for every
step group. The new optional `runtime.window_attention_batch_steps` setting
defaults to 1, retaining that original implementation. The production candidate
uses 16; accepted values are integers 1–64. The setting is carried in saved
navigation metadata and must match for strict resume.

`_BatchedWindowAttention` packs bounded batches of independent windows for
`flash_attn_varlen_func`. Each packed sequence has exactly the same query span,
instruction prefix and up-to-eight-group KV span as the serial implementation.
FA2's bottom-right causal alignment is applied separately to each window. The
prefix itself is a separate causal sequence. Packing never permits another
window's keys to become visible.

The custom autograd function saves only original Q/K/V. Backward reconstructs
one bounded window batch, computes its VJP and scatter-adds contributions into
the original shared prefix/history gradient buffers. Overlapping KV gradients
are accumulated in reversed group order, matching the serial structure. There
is no retained all-episode duplicated KV tensor and no detached temporal gradient.
As in the original custom function, higher-order derivatives are unsupported.
Cached streaming inference retains the original single-append attention path.

## Correctness evidence

The pinned GPU suite passed **231 tests, zero skipped**, with 89 existing FLA/
TileLang deprecation warnings. Seven new tests cover packed windows: an
independent FP64 dense causal-mask oracle checks values and derivatives at
1e-10/1e-9 tolerances for variable-length groups, eviction boundaries, several
batch sizes and a partial final batch. Saved tensor storage is bounded by the
original Q/K/V storage. Actual H100 BF16 FA2 tests compare serial versus varlen
forward/backward, requiring gradient relative norm error below 2% as well as
elementwise bounds. Existing cache, serialization, lane, optimizer and recovery
unit tests remain passing.

`scripts/vln/check_text_step_lane.py` loads the actual pinned model, tokenizer
and real R2R/RxR observations. It verifies unchanged tokens/targets/images and
exact native-vs-zero-lane logits and loss. All lane parameters participate in
backward; first output-projection gradient norms are nonzero. After explicitly
opening output projections, the final action loss produces finite nonzero QKV
and beta-projection gradients in every lane. Frozen vision receives no gradient.

The separate serial-vs-batched full-model comparison measured vocabulary-logit
max error 0.5625, RMS 0.054025 and loss-sum difference 0.003580. These are BF16
numerical differences, not bitwise equivalence. The declared inherited gate is
`atol=0.75, rtol=0.03`, with vocabulary-logit RMS <=0.15; it was not loosened after
observing the result. FP64 derivatives are checked independently by the dense
oracle; BF16 optimizer trajectories can still diverge numerically.

Text streaming replay uses the real `StreamSession.observe` decoding loop,
capturing its unmodified vocabulary logits while forcing only the selected
argmax token to the canonical expert response. Thus offline and online histories
match. Each dataset supplies 17 real observations, crossing eviction at steps
8/9/16; the last label is explicitly changed to STOP for this fixture. The gate
also checks retry idempotency, bounded KV, sidecar token accounting and reset.
It is a numerical replay, not a closed-loop navigation evaluation.

| Dataset | Scored response tokens | Max score error | RMS error | Argmax agreement |
|---|---:|---:|---:|---:|
| R2R | 50 | 0.734375 | 0.061592 | 100% |
| RxR | 50 | 0.500000 | 0.042387 | 100% |

## Performance measurements

One H100 ran matched attention forward/backward benchmarks with head layout
16Q/4KV and head dimension 256, prefix 128, 320 tokens/group, one warmup and two
timed repetitions per setting. These include packing and gradient scatter.

| Groups | Serial, seconds | Batch 16, seconds | Batch 32, seconds |
|---|---:|---:|---:|
| 64 | 0.03519 | 0.03037 | 0.02919 |
| 183 | 0.09677 | 0.08023 | 0.07741 |
| 627 | 0.33197 | 0.26762 | 0.25920 |

At 627 groups, batch 16 reduced this attention-only time by 19.4%, while peak
allocated memory increased from 7.756 to 9.073 GiB. Batch 32 used 10.488 GiB.
Batch 16 was selected to retain more GPU headroom for full-model training.

A separate four-rank full-model profile used the same complete median-length
episode, 79 observations/25,203 tokens, global batch eight and two optimizer
updates from base. Second-update wall compute was 7.2203 s serial versus 7.0944 s
batch 16 (maximum across ranks), approximately 1.7% lower. This is a small
single-fixture measurement; it does not establish a universal speedup or the
absolute fastest possible training implementation. Startup-inclusive durations
70.07/64.06 s include loading and loader startup and are not steady-state speed.
No observation, token, training step or attention visibility was removed for speed.

## Resource qualification

Tests ran in existing interactive allocation **4659**, worker-3, four idle H100
80 GB GPUs, 60 CPUs and 768 GiB RAM. The separate worker-1 job was untouched.
Every resource profile uses the actual four-rank Trainer, ZeRO-2, BF16, GAS2,
GPU optimizer state, activation checkpointing/offload and two complete updates.
Profiles repeat declared real episodes to fill eight slots; this is stress
testing, not unique corpus exposure. CPU optimizer offload was unnecessary.

The full manifest's exact encoded-length audit identifies:

- R2R: `PuKPg4mmafe:6967:6967`, 183 observations, 58,290 tokens.
- RxR: `82sE5b5pLXE:31149:22206`, 627 observations, 199,735 tokens.
- Immediately below activation offload: `Vvot9Ly1tCj:13476:79656`, 205
  observations, 65,526 tokens.

The longest RxR profile passed both updates on all ranks. Maximum reserved GPU
memory was **76.9434 GiB**; maximum guarded host working set was **417.872 GiB**.
Total cgroup usage reached 768 GiB including reclaimable filesystem cache.
Afterward `memory.events` recorded **oom=0 and oom_kill=0**; high/max reclaim
events were nonzero. The host guard counts dirty/writeback cache and excludes
estimated clean inactive cache, matching the existing reviewed supervisor.
The production guard is 700 GiB against the allocation's 768 GiB limit.

These checks demonstrate fit on the tested workload and runtime. They do not
guarantee absence of every future OOM. Production has periodic complete recovery
checkpoints, and the interactive allocation remains available if a step fails.

## Checkpoint and production launch record

The independent distributed text-loss reference passed all tested accumulation
and tail cases (global action counts 9/10/5), with maximum parameter error zero.
A four-rank interrupted/resumed recovery reference reproduced final weights,
scheduler values and sample order exactly.

The real text smoke used eight explicitly truncated diagnostic episodes (four
R2R and four RxR), global batch eight, two optimizer updates, and complete
recovery saves at each update. It trained, exported and strictly reloaded. A
second process resumed checkpoint-1 and completed update 2, then strictly
reloaded its export. The production run does not use these diagnostic weights.
The resumed export preserved all 21,054,000 lane parameters bit-for-bit relative
to the checkpoint file; all four learned output projections were nonzero.
Multi-step reload checks produced finite loss sums 1.539811 (R2R) and 5.423224
(RxR). Real-model BF16 training is not claimed bitwise deterministic: the
STOP-only reload fixture loss was 0.251069 uninterrupted versus 0.256484 resumed.

Recovery saves occur every 250 updates. The inherited completion-marker
protocol retains two complete rolling recovery checkpoints plus both epoch
checkpoints; an incomplete save cannot be selected as a restart point. This
bounds checkpoint growth while retaining optimizer, scheduler and rank RNG
state. The launcher refuses an existing output directory, checks the source
commit and file hashes, verifies the full manifest hash and 30,815 records,
requires four GPUs and at least 400 GiB free disk, and checks the requested
2-epoch/7,704-update/232-warmup schedule before starting.

Raw qualification evidence is under
`/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN-streaming_text_dual-evidence`.


The final mixed-length stress test replayed eight distinct RxR episodes,
including the 627-observation episode, for three complete optimizer updates.
All four ranks completed the expected exposures; peak reserved GPU memory was
76.8848 GiB, guarded host working set 295.6010 GiB, and no host guard triggered.
The supervisor reported PASS and the independent profile verifier reported
`MIXED_TAIL_VERIFIED`. This complements the all-longest and offload-boundary
profiles by exercising changing episode sizes across accumulation slots.


## Production identity

The production source is commit `a7e442cbd3c43cf3ec238eaa527e9c67f5ce2408`,
pushed to `origin/streaming_text_dual`. Its tracked files were compared
byte-for-byte against the GPU-tested development checkout and copied into a
detached Git worktree with tracked files made read-only. Later report-only
commits do not change that running source.

Run root:
`/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN-streaming_text_dual-runs/window8_text_base_e2_20261007`.
The launch passed its production preflight in Slurm step **4659.45**, worker-3.
The parent interactive shell **4659.0** remains available.

The complete manifest is
`/mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN/artifacts/r2r_rxr15_train_20260929.jsonl`,
SHA-256 `5deb425d2594ce96931dd6ce12bd6084b066cd44f04808c1dd3ee2e4b3672933`.
`qualification.json` in the run root records source file hashes and the
qualification results. `launch_requested.json` records the exact command and
source revision. Training logs are `supervisor/command.log`, sampled RAM is
`supervisor/host_memory.jsonl`, and model/checkpoint output is `train/`.

The run uses the original pinned Qwen3.5-4B snapshot directly, without an
initial-policy checkpoint or resume argument: two epochs over the full joint
manifest, Window8, canonical text action history, four H100 ranks and GAS2.


At **2026-10-07 04:27:32 UTC**, all four ranks had completed **12/7,704**
updates with contiguous progress and finite logged losses (latest 0.3506).
All four schedule gates passed `(7704, 232)`. Observed production maximum
reserved GPU memory was 66.7559 GiB and maximum guarded host RAM 332.9547 GiB.
The launch verifier wrote `launch_verified.json`. The first-50 campaign report
had not yet run at this verification point. Startup/early length-bucket timings
are insufficient for a reliable completion ETA. Training remains active;
completion of both epochs and navigation quality are future results.

To inspect this run from the login node:

```bash
squeue --steps -j 4659
tail -f /mnt/data/vmo-ai-task/anhdh35/SimpleMemVLN-streaming_text_dual-runs/window8_text_base_e2_20261007/supervisor/command.log
```

If the owned training step fails, allocation 4659 and its interactive shell
remain available. Diagnose the recorded error first, select only a checkpoint
with a valid `RECOVERY_COMPLETE.json`, and use the existing strict resume path
with the same source/config/manifest. The fresh launcher deliberately refuses
to overwrite this run; it is not an automatic retry loop.
