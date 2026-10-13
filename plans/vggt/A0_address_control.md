# A0 — GeoAddress architecture control, no VGGT loss

> **For agentic workers:** implement this together with A1's shared runtime module. A0 and A1 must start independently from identical initial weights, not sequential fine-tuning.

**Goal:** isolate the effect of adding small native Q/K adapters before testing teacher-guided addressing.
**Architecture:** rank8 residuals on raw query/key heads0 and1 at GDN decoder24, image tokens only; no extra recurrent state.
**Tech stack:** original native GDN kernel and cache paths.
**Spec:** master and common infrastructure.

## Global constraints

One modified layer24, no other layer/head sweep. No lane, current alignment, state-read probes, teacher targets, episodic KV, external geometry or action-interface change. Full BPTT. Teacher-free A0 is deliberately one of the five new runs.

## Review focus

- Up projections zero, down projections random: do not zero both and block initial adapter gradients.
- Residual added before native Q/K normalization, not to already-rotated attention Q/K.
- Raw heads0/1 are expanded according to the native GDN grouping, not confused with full-attention heads.
- Image mask must also be supplied correctly in incremental serving; missing metadata cannot silently bypass adapters.
- Existing native weights and tied storage retain their namespace/identity.

## 1. Exact attachment

Take normalized decoder24 mixer input x. Native QKV projection/causal convolution/SiLU and split produce raw Q/K arrays with native key-head dimension128. Let selected raw heads be [0,1], combined width256.

Adapters:

`q_down=Linear(2560,8,bias=False)`
`q_up=Linear(8,256,bias=False)`
`k_down=Linear(2560,8,bias=False)`
`k_up=Linear(8,256,bias=False)`.

No activation between down and up in v1. Bias-free matrix parameter count:45056. Initialize down matrices Normal(0,.02); up matrices exact zero. Use the same isolated adapter seed in A0/A1.

For image rows only:

`q_selected += reshape(q_up(q_down(x)), [2,128])`
`k_selected += reshape(k_up(k_down(x)), [2,128])`.

Implement as an out-of-place scatter/add; do not mutate tensors saved for backward. Unselected heads unchanged. Text/decision/action/feedback rows have mask=false, hence no direct address residual. They can still be affected indirectly through state/history as intended.

After addition, use original repeated-head expansion and FLA's Q/K L2 normalization and scale exactly once. V, z, beta, g and native output/norm remain their native equations and train normally.

## 2. State and serving

Native recurrent state has exactly its previous shape; no R state or extra cache entry. New parameters are saved under e.g. `backbone.model.language_model.layers.24.linear_attn.geoaddress.*`.

Serving wrapper must preserve three native modes:

1. First prefill, no previous recurrent/conv state.
2. Multi-token observation append with previous conv/recurrent state.
3. Single-token action decoding with previous state.

Image mask is derived from exact input token IDs in `nav_model.hidden` and passed explicitly through decoder kwargs. Mask must match [B,L] on every append. Text-only block produces an all-false mask. Recurrent update count/logical positions are untouched; no teacher or correspondence calculation at inference.

Do not globally monkeypatch the Transformers class. Patch only the selected instance with an idempotent version-checked installer retaining the original unbound forward. Coordinate this installer with research GDN preparation; no double adapter addition.

## 3. Training

Navigation loss ONLY, same base weights/seed/dataset/update schedule as A1. Native LR5e-6; adapters LR5e-5. Full BPTT, frozen vision, original Window8. A0 needs no target loader, query decoder or probe rows.

First update may give nonzero gradients to up matrices and zero to down matrices; after up matrices open, gradients should reach down matrices. The adapter is not required to produce nonzero output before training.

A0 is not an optimization baseline that freezes the trunk: the trunk trains identically to A1. Holding native weights frozen in only one arm would defeat the control.

## 4. Tasks

### A0.1: pure adapter
**Files:** `models/geoaddress.py`, `tests/vln/test_geoaddress.py`.
**Interface:** `forward(x, q_raw, k_raw, image_mask) -> (q_raw_new,k_raw_new)`.
- [ ] Parameter count45056 and zero-init native equality.
- [ ] Masks, head selection, nonimage identity, nonfinite rejection and single-token dimensions.
- [ ] First/second-update gradient-opening tests and A0/A1 initialization identity.

### A0.2: native integration
**Files:** `models/gdn_research.py`, `models/nav_model.py`, runtime installer.
- [ ] Compare native zero-adapter prefill/continuation/decode outputs/state with identical inputs.
- [ ] Active-adapter CPU reference and pinned GPU kernel parity with repeated-head mapping.
- [ ] Save/load retains native namespaces, tied LM storage and all four matrices.

### A0.3: lifecycle/qualification
- [ ] Test reset/retry/Window8 eviction, varied action responses and absolute positions.
- [ ] Update contracts/export so old loaders reject unsupported GeoAddress metadata rather than ignore it.
- [ ] Full longest resource gate, short genuine optimizer/reload smoke, then launch A0.

## 5. Results

A0–V0 measures the effect of this small adapter/training path, subject to historical baseline recipe differences. A1–A0 is the stronger teacher-effect comparison. Added parameters/cost are small but not zero; report actual latency. Do not claim native learned content addresses previously contained no geometry.
