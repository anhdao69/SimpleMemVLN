# DINOv3 ViT-B/16 G3 execution

User selected the official ViT-B/16 checkpoint and authorized training on 2026-10-19.
Preserve ViT-L support on its existing branch; implement B support separately on
c2_dinov3_g3_vitb. Explicit teacher profile chooses 768D, 12 blocks, 12 heads and the
pinned official B configuration/weight hashes. Teacher final block is 11;
**student native GDN supervision remains zero-based decoder 24**.

All geometry/pooling, G3 sampling, constant 0.1 coefficient, full BPTT and native
inference behavior stay as designed. Student decoder becomes 4096 → 1024 → 768.
Production remains fresh base, joint R2R+RxR15deg, seed 430, one-epoch cosine,
3,852 updates, 116 warmup, effective batch 8 on 4 H100 GPUs, CPU optimizer and activation RAM.

The official B snapshot downloaded successfully at revision
5931719e67bbdb9737e363e781fb0c67687896bc. Weight SHA256:
9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b.
Config Git blob: 9f0c03c12a21df9678269d96dac7b59819349792.

Qualification is sequential and fails closed on worker-2, allocation 4925:
pretrained teacher batches 1/2/4/8 and frozen-gradient check; native FLA and
checkpointed GDN with 768D head; longest complete R2R (183 frames), 2 updates; longest complete
RxR (627 frames), 3 updates; mixed-dataset reference/recovery at checkpoint 2 → 3;
distributed valid-pair normalization; serving export replay and reset parity.
Only successful qualification may automatically invoke the production launcher.
All smoke outputs are separate and production starts from base, not smoke weights.

Diagnostic max_steps=3 uses a short diagnostic scheduler (with the declared
warmup), as existing engineering profiles do. It is not the production schedule.
Production uses the one-epoch schedule guard and full manifest without update cap.

## Executed qualification and production

All qualification stages passed on worker-2 in step 4925.143. The driver
automatically started the full fresh-base run after serving validation passed.
The production schedule guard confirmed 3,852 total updates and 116 warmup
updates on every rank. The full manifest has 30,815 unique episodes,
3,128,624 action exposures, and a maximum serialized length of 199,735 tokens.

See `reports/dinov3_g3_vitb/training.md` and its evidence directory for measured
throughput, memory, recovery tolerances, serving checks, and startup status.
Production runs the sealed implementation commit 18ff9a4; later documentation
updates do not modify the running source.
