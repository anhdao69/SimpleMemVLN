# Candidate-logit FullContext R2R evaluation

## Checkpoint and source

- Code: `streaming_logits` branch, based on `origin/streaming_logits@51887d2`.
- Hub checkpoint: `anhdao69/SimpleMemVLN-R2R-RxR15deg-FullContext-CandidateLogits`, immutable revision `d44754764ab94ddf1d8e16022fe91d1a5f9e1a0c`, `epoch-1` (training update 3,852).
- All 11 files in the published `SHA256SUMS.json` passed local SHA256 verification, including the 10,350,179,723-byte `pytorch_model.bin`.
- Navigation metadata: `vln_candidate_logits_v1`, `candidate_logits`, trainable LM rows, A/B/C/D token IDs 32/33/34/35, candidate-token feedback, FullContext, four actions mapped to Habitat 1/2/3/0.

## Evaluation contract

The evaluator runs the model in the Qwen environment and Habitat 0.2.4 in its separate simulator environment. Each observation is transported losslessly as raw RGB, then the branch's `StreamSession` selects one of four candidate logits, appends deterministic feedback, and returns the canonical action ID. The evaluator checks that class, action name, and Habitat ID agree before executing the action. It records the model prediction separately from a forced cap STOP, retains zero SR/SPL for failed episodes, and journals every episode with an immutable resume contract.

R2R `val_unseen`: 1,839 episodes; 640×480 RGB; 0.25 m forward; 15° turn; 500-step cap; 3.0 m success threshold; deterministic seed 42 and scene-grouped order. The dataset, Habitat config, episode-ID list and order, scenes directory, base model assets, seed, protocol, cap, and transport all exactly match the earlier FullContext-B R2R epoch-2 evaluation contract. The policies use different serializers and model weights as intended.

## Verification and run state

- One-episode Habitat smoke passed with 50 decisions, no evaluator failures, a model-selected STOP, and zero generated text tokens. Runtime reported `candidate_logits` and `full_context`.
- Candidate and evaluator unit tests: 7 passed. Full suite excluding the existing H100-only kernel dispatch assertion: 94 passed, 6 skipped, 1 deselected. This host reports compute capability 12.0, so the H100 TileLang backend assertion is not a valid gate for this host.
- Full evaluation completed. Output: `/storage/anhdh35/SimpleMemVLN/artifacts/evaluation/candidate-logits-r2r/epoch-1`; log: sibling `epoch-1.log`.
- The journal contains 1,839 unique episodes and the summary records `complete: true`, 0 failures, 874 successes, **47.53% SR**, **43.45% SPL**, 6.25 m navigation error, and 56.17% oracle success. There were 1,676 model-selected STOPs and 163 forced cap STOPs.

## Completed FullContext-B comparison

| R2R `val_unseen` policy | Successes / 1,839 | SR | SPL | Failures |
|---|---:|---:|---:|---:|
| Candidate logits, epoch 1 | 874 | 47.53% | 43.45% | 0 |
| FullContext-B text, epoch 1 | 891 | 48.45% | 44.81% | 1 |
| FullContext-B text, epoch 2 | 916 | 49.81% | 45.40% | 0 |

Candidate epoch 1 is 0.92 percentage points below B epoch 1 and 2.28 points below B epoch 2 on SR. This is a checkpoint comparison, not an isolated action-head ablation: the candidate run uses a different policy and its own training trajectory.
