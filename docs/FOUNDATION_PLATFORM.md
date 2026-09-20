# otto-von-grad Platform Foundation Document

> **Purpose:** This document captures the intent, current state, target layering, open design questions, and planning order for turning otto-von-grad (OVG) from "an autograd engine that happens to ship a GPT" into the platform on which focused language models, vision transformers, and image generators are built. It is a living foundation document, not yet an implementation specification. `AGENTS.md` remains the authoritative description of what exists *today*; this document describes where it is going and why.

---

## Intent

OVG becomes a layered C11 machine-learning platform. The bottom layer is a general autograd engine and training harness that knows nothing about models. Above it sit model-agnostic building blocks. Above those sit small, separable modality packages — language, vision, generation — each of which can eventually be consumed or published on its own. Applications live in sibling repositories and link only the layers they need.

The owner's stated goal: build out focused GPTs/LLMs, vision transformers, and image generators on top of OVG without tying OVG to any one of them.

---

## Motivating Driver

Two consumers exist today and they want different things from the library. `lambda` (a lambda-calculus GPT curriculum) uses the full stack through the GPT model. `vexilloscope` (a ViT flag classifier) uses the autograd engine and the transformer blocks but nothing language-related — and still had to link the tokenizer and sampler because the library was one target.

Both consumers hand-roll the same training machinery: Adam state, gradient accumulation, learning-rate schedules, evaluation loops, checkpointing. vexilloscope's `main.c` is 1,294 lines, most of it harness rather than model. When a third project (an image generator) arrives it will write the same code a third time — unless the platform absorbs it first.

The layering work of 2026-09-20 (see Brownfield Baseline) made the boundaries real in the build graph. This document decides what goes in each layer next.

---

## Brownfield Baseline (as of 2026-09-20)

State on the `refactor` branch, commits `93ff570`..`5f83f16`, unpushed at time of writing:

- **Build graph:** three static targets with dependencies strictly downward, plus an umbrella.
  - `ovg_core` — `Tensor` (N-D, `TG_MAX_DIMS = 4`), ~30 differentiable ops each paired with its backward, topological-sort backward, SGD and Adam (CPU and GPU), gradient-norm clipping, RNG, binary checkpoint I/O (format v2, shape-validated), fatal-error hook, CUDA kernels with cuBLAS matmul and BF16 matmul.
  - `ovg_nn` — `TgLinear`, `TgSelfAttention` (causal and encoder), `TgBlock` (pre-norm, dropout, stochastic depth), `TgTransformer`.
  - `ovg_lm` — `TgGPT` + `TgGPTConfig`, byte-level `TgVocab` tokenizer, argmax / top-k sampling, `tg_generate`.
  - `ottovongrad` — INTERFACE umbrella over all three, preserved so existing consumers link unchanged.
- **Public surface:** headers in `include/ovg/` (PUBLIC, included by bare name); `src/` PRIVATE with two internal headers (`cuda_ops.h`, `tg_cuda_internal.h`). `attention` renamed `tg_attention`. `tg_cuda.h` narrowed to the five functions consumers call.
- **Ops added today:** `tg_concat(a, b, axis)` — inverse of `tg_slice`, CPU + CUDA, needed for the ViT CLS token and later for UNet skip connections.
- **Dead code removed:** `legacy/` scalar autograd; fifteen v1 CUDA kernel wrappers with no callers.
- **Demo:** `examples/candide.c` (target `candide`), paths resolved via `OVG_EXAMPLES_DIR` so it runs from any directory.
- **Tests:** 80 (CUDA build) / 71 (CPU build), one file per module, linked against `ovg_lm`. The test binary therefore cannot detect a layer violation; that boundary is enforced by grep and by building each target standalone, as documented in `AGENTS.md` Verification.
- **Consumers:**
  - `lambda` builds against current OVG with no changes to its CMake.
  - `vexilloscope` was stuck on the v1 API since the v2 N-D migration (2026-05-28). Its port is in progress in that repo: a mechanical pass (`->data` → `TG_DATAF`, `->rows/->cols` → `shape[]`, two-int `tg_new` → N-D form), `tg_concat_rows` → `tg_concat`, `tg_row_slice` → `tg_slice` + `tg_reshape` on the now-3D encoder output, `cuda_smoke.c` deleted. It compiles with zero warnings and links `ovg_nn` only. Its v1 weights file cannot load — v1 stored LayerNorm affine and FFN bias parameters pre-tiled to `[seq_len, C]`, v2 stores `[1, C]` — so a retrain is required.
- **Includes remain bare.** The namespaced form (`#include "ovg/tg_ops.h"`, `include/ovg/lm/`) is deferred until a modality package is actually about to be published separately.

Known ceilings carried forward from `AGENTS.md`:

- Eager dispatch, one kernel launch per op, no fusion, no graph reuse. vexilloscope at batch 1 with 1,025 tokens runs ~3 steps/s on an RTX 4070 Super.
- `topo_sort` is recursive; `TG_MAX_GRAPH = 8192` nodes.
- `TG_MAX_DIMS = 4`. Batched multi-head attention already uses all four (`[B, H, T, D]`).
- BF16 is CUDA-only and matmul-only; every other op runs in F32.

---

## North Star

```text
applications          lambda · vexilloscope · <image-generation repo> · …     (sibling repos)
                             │
modality packages     ovg_lm · ovg_vision · ovg_diffusion                      (each separable)
                      gpt/tokenizer/sampler · patch embed/pooling · unet/schedule/sampler
                             │
building blocks       ovg_nn    linear, attention, transformer block/stack,
                               conv / residual blocks, norms
                             │
engine + harness      ovg_core  tensor, ops (incl. conv family), autograd,
                               optimizers + schedules + accumulation, checkpoint, RNG, CUDA
```

Properties the North Star must have:

1. **Dependencies point strictly downward.** A lower layer never includes a header from a higher one. This is already true and is checked by grep and per-target builds.
2. **Each modality package is publishable on its own.** `ovg_lm` is the template: a small set of modules that assume one kind of data, linking `ovg_nn`, with its own tests. `ovg_vision` and `ovg_diffusion` follow the same shape.
3. **The harness lives below the modalities.** A new project should get optimizers, schedules, accumulation, clipping, checkpointing, and an evaluation-loop skeleton without writing them.
4. **Applications own data.** Loading, augmentation, dataset manifests, CLI, and reporting stay in the application repos.

---

## Placement Rule

Extends the rule already in `AGENTS.md` → Library Layers:

| A piece of code that… | belongs in |
|---|---|
| operates on `Tensor` with no notion of a model (an op, an optimizer, a schedule, checkpoint I/O) | `ovg_core` |
| is a parameterized module any architecture could reuse (a linear, an attention, a conv block, a norm) | `ovg_nn` |
| assumes a modality — tokens, pixels as images, noise timesteps | the matching modality package |
| loads, augments, or describes data; parses a CLI; writes reports | the application repo |

If a lower layer needs something from a higher one, the thing is in the wrong layer: move it down rather than adding the include.

---

## Planning Method

Each planning topic below is resolved in the same shape:

- **Overview:** what the topic covers.
- **Why it matters:** what ambiguity or risk the decision removes.
- **Constraints:** what the decision must respect.
- **Non-goals:** what the decision deliberately does not solve.
- **Options:** viable approaches and their tradeoffs.
- **Follow-ups:** questions or experiments needed before closing the decision.
- **Decision:** the chosen direction once settled, or `Open`.

Reasoning is preserved alongside decisions so that later specifications can derive requirements with minimal interpretation.

---

## Specification Strategy

The platform is specified in phases, each producing one implementable slice with a clear completion signal. The order is chosen by leverage: what removes the most duplicated code across the most consumers first, and what the next concrete model actually needs.

### First Spec Target

```text
docs/SPEC_PLATFORM_PHASE1_HARNESS.md
```

Scope: the training harness (optimizer state as a struct, LR schedules, gradient accumulation helper, checkpoint format v3 with optimizer state and metadata header, exact-resume on by default), followed by migrating `examples/candide.c`, `../lambda`, and `../vexilloscope` onto it. Completion signal: all three consumers train through the harness, and vexilloscope's `main.c` shrinks by the harness code it no longer owns.

### Phase 1 Non-Goals

- Any new op beyond what the harness itself needs.
- `TgPatchEmbed`, `ovg_vision`, or other vision blocks.
- The conv family.
- Namespaced includes or package extraction.
- Performance work (batching in vexilloscope, kernel fusion).

### Later Specs

```text
SPEC_PLATFORM_PHASE2_VISION.md       # TgLinear fix + rename; new ovg_vision target with TgPatchEmbed + pooling; vexilloscope adopts both
SPEC_PLATFORM_PHASE3_CONV.md         # iterative topo_sort + growable graph; conv family, group_norm, silu/sigmoid, sinusoidal embedding — driven by a first DDPM
SPEC_PLATFORM_PHASE4_PACKAGES.md     # ovg_diffusion target, namespaced includes, extraction readiness
```

Names and boundaries may shift; the principle should not: spec only the next implementable slice.

---

## Decision Log

| Topic | Status | Decision |
|---|---|---|
| Four-tier layering | Closed | engine+harness (`ovg_core`) → building blocks (`ovg_nn`) → modality packages (`ovg_lm`, later `ovg_vision`, `ovg_diffusion`) → applications in sibling repos. Dependencies strictly downward. |
| Placement rule | Closed | op/optimizer/schedule → core; reusable parameterized module → nn; assumes a modality → package; data/CLI/reports → app. |
| `ovg_lm` as the package template | Closed | New modality packages copy `ovg_lm`'s shape: a few modules, links `ovg_nn`, own tests, no data loading. |
| `ottovongrad` umbrella | Closed | Kept indefinitely as the everything-included default; consumers are encouraged but not required to link a narrower target. |
| Bare vs. namespaced includes | Closed (deferred) | Stay bare until a package is about to be published separately; then switch in one commit, verifiable by zero `C1083` across consumers. |
| Training harness placement | Closed | In `ovg_core`, finishing `tg_train.h`. Exact-resume is default on: checkpoints save optimizer state and step; parameter-only v2 files still load with moments rebuilt. |
| `TgLinear` fate | Closed | Fix in place and rename to `tg_linear.h`: bias `[1, n_out]` expanded at forward, no baked-in batch, no out-parameter. |
| Conv family scope | Closed | Driven by a first concrete DDPM (~32×32 RGB, small UNet, one attention level). First `conv2d` is im2col + cuBLAS GEMM, no cuDNN. `group_norm` is a separate op; `layer_norm` untouched. |
| Vision-block placement | Closed | Create `ovg_vision` now (Phase 2). Further vision consumers are imminent, which satisfies the second-consumer rule ahead of time. |
| Graph and dimension limits | Closed | `TG_MAX_DIMS` stays 4 — no plans need more. `topo_sort` goes iterative with growable graph capacity as a Phase 3 prerequisite. |

---

## Training Harness

### Overview

The code between "I have a loss tensor" and "the parameters moved": optimizer state, learning-rate schedule, gradient accumulation across micro-steps, gradient clipping, the step counter, and periodic checkpointing. Also the skeleton of an evaluation loop (set `tg_training = 0`, run forward on held-out data, aggregate a metric, restore).

### Why It Matters

This is the single largest body of code duplicated across consumers. A survey of the three applications on 2026-09-20:

| Concern | `examples/candide.c` | `lambda/src/main.c` | `vexilloscope/src/main.c` |
|---|---|---|---|
| Adam moment allocation and step | yes | yes | yes |
| Warmup + cosine LR schedule | — | — | yes |
| Gradient accumulation | — | — | yes |
| Gradient-norm clipping | — | — | yes |
| Eval loop | yes | yes | yes (27 sites) |
| Weight save/load | `tg_checkpoint` | `tg_checkpoint` | own 362-line format in `vit.c` |

vexilloscope's is the most complete and is the reference implementation to lift. Its hand-rolled weight format is also what broke silently across the v1→v2 shape change; `tg_checkpoint` validates shapes on load and would have failed loudly.

### Constraints

- Must work identically on CPU and CUDA tensors; `tg_adam_step` / `tg_adam_step_gpu` already exist and differ only in where the moment buffers live.
- Must not impose a data loader or a batch abstraction. The harness steps on a loss tensor the caller produced.
- Must remain explicit and inspectable. No callbacks-within-callbacks; a training loop written against the harness should still read as a loop.
- Checkpoint format may gain optimizer state (moments, step) but must stay backward-compatible with format v2 parameter-only files, or bump the magic and say so.

### Non-Goals

- Distributed or multi-GPU training.
- A dataset / dataloader abstraction (applications own data).
- Mixed-precision policy beyond what `tg_cast` already offers.
- Logging or metrics sinks beyond returning numbers to the caller.

### Options

#### Option A: Harness in `ovg_core`

Add `tg_optim.h` (an `TgAdam` struct owning `m`/`v` buffers, CPU or device, with `tg_adam_create / step / free`), `tg_sched.h` (pure functions: `tg_lr_warmup_cosine(step, total, warmup, base, floor)`, linear, constant), and an accumulation helper that zeroes, runs N `tg_backward_accum`, clips, steps. Extend `tg_checkpoint` with an optional optimizer-state section.

- Pro: one target; every consumer already links core; matches "operates on `Tensor` with no notion of a model".
- Pro: `tg_train.h` already holds SGD/Adam/clip — this is finishing that file, not opening a new concern.
- Con: core grows a little further from "just autograd".

#### Option B: Fourth target `ovg_train` between core and nn

Same code, own static library linking `ovg_core`. `ovg_nn` does not depend on it; modality packages and applications link it explicitly.

- Pro: keeps `ovg_core` minimal and makes "engine" vs "harness" a build-graph fact.
- Con: a fourth target for ~300 lines; every consumer links it anyway; the `ovg_nn`-doesn't-need-it distinction buys nothing today since nothing links `ovg_nn` without training.
- Con: `tg_train.h` (backward, SGD, Adam, clip) is already in core, so the split would either move existing code up or leave the optimizer family straddling two targets.

### Follow-ups

- Confirm whether lambda's curriculum loop needs anything the vexilloscope harness lacks (e.g. per-phase LR resets) before fixing the schedule API.
- Closed: optimizer state is saved by default (see Decision).

### Decision

Closed (2026-09-20): **Option A.** The harness lives in `ovg_core`, finishing `tg_train.h`; the split in Option B has no consumer that would benefit from it. Revisit only if a consumer ever wants `ovg_nn` without training.

**Exact-resume is default on.** The owner regularly stops and restarts long runs, so a resumed run must continue from the same optimizer state, not from zeroed moments. Consequences for the Phase 1 spec:

- The checkpoint gains an optimizer-state section (Adam `m`/`v` per parameter, step counter, and the harness's schedule position) and a small metadata header (see Open Questions: self-describing checkpoints). This is format v3 with a new magic; `tg_checkpoint_load` continues to accept v2 parameter-only files and reports that moments were rebuilt.
- Saving optimizer state is the default; a flag disables it for "weights-only" exports (inference artifacts, sharing).
- Device-resident moments (`tg_adam_step_gpu`) are synced to host on save; the harness owns that, not the caller.

---

## `TgLinear` Fate

### Overview

`tg_mlp.h` provides `TgLinear`, a v1-era convenience layer. Its bias is `[batch, n_out]` with the comment "no broadcasting yet"; it requires the batch size at construction; its forward returns an intermediate through an out-parameter.

### Why It Matters

It is in `ovg_nn` as the canonical "linear layer" but `TgBlock` does not use it (the block does its own `tg_matmul` + `tg_reshape` + `tg_expand_dim` + `tg_add`), and its only consumer is `vexilloscope/src/mlp_classifier.c`. As written it teaches the wrong pattern for v2.

### Constraints

- vexilloscope must keep compiling through whichever choice is made.
- Whatever remains in `ovg_nn` as a linear layer must follow the v2 idiom: bias `[1, n_out]`, expanded at forward time, batch size not baked in.

### Non-Goals

- A general "module" abstraction with virtual forward/params. Modules stay concrete structs.

### Options

#### Option A: Fix in place

Bias becomes `[1, n_out]`; forward expands it to the input's leading dims; drop the `batch` field and the `xw_out` out-parameter; `tg_linear_params` stays.

- Pro: keeps a named linear layer, which every modality package will want (`TgGPT`'s output projection, `TgPatchEmbed`, a ViT head).
- Con: touches vexilloscope's `mlp_classifier.c`.

#### Option B: Delete

Remove `tg_mlp.[ch]`; vexilloscope's classifier does `tg_matmul` + bias expansion inline as `TgBlock` does.

- Pro: less surface; one fewer thing that can drift.
- Con: three future packages re-implement a linear layer; the inline pattern is five lines but easy to get wrong (shape of the bias expansion).

### Follow-ups

- Rename to `tg_linear.h` if kept — `tg_mlp.h` misdescribes a single linear layer.

### Decision

Closed (2026-09-20): **Option A, fix in place.** Rename `tg_mlp.h` → `tg_linear.h`; bias becomes `[1, n_out]` and is expanded to the input's leading dims at forward time; the `batch` field and the `xw_out` out-parameter are removed; `tg_linear_params` stays. Sequenced in Phase 2 alongside the vision blocks so vexilloscope is touched once. The reasoning that carried it: every modality package needs a named linear layer, and the inline expansion pattern is short but easy to get wrong.

---

## Vision Blocks

### Overview

What a vision transformer needs above `ovg_nn`'s transformer: a patch embedding (linear projection of flattened patches + learned positional embedding + optional CLS token prepend) and a pooling choice at the output (CLS-token extract or mean over tokens). vexilloscope implements both today in `patch_embedding.c` (73 lines) and one line of `vit.c`.

### Why It Matters

These are model-agnostic — any ViT, and later the encoder side of a latent diffusion model, needs them — but they currently live in one application. The CLS extract is also where the v2 port bit: the encoder now returns `[1, T, C]`, and the correct extraction is `tg_slice(enc, 1, 0, 1)` followed by `tg_reshape` to `[1, C]`. That knowledge should live in the library once, not in every consumer.

### Constraints

- Patch extraction from pixels (`img_patchify`) is image handling and stays application-side; the library block takes `[B, n_patches, patch_size]` or `[n_patches, patch_size]` and returns tokens.
- Must accept both 2D `[T, C]` (batch 1) and 3D `[B, T, C]` input the way `tg_block_forward` does, and document which it returns.
- Uses `tg_concat` for the CLS prepend — no bespoke op.

### Non-Goals

- Image decoding, letterboxing, augmentation.
- Convolutional patch embedding (that is Phase 3 territory).

### Options

#### Option A: `TgPatchEmbed` + `tg_pool_cls` / `tg_pool_mean_tokens` in `ovg_nn`

- Pro: immediately reusable; vexilloscope shrinks; the ViT recipe becomes "patch embed → transformer → pool → linear".
- Con: `ovg_nn` starts to hold things that are only used by vision — but they do not *assume* pixels, only a token layout.

#### Option B: Create `ovg_vision` now and put them there

- Pro: cleaner story for later extraction.
- Con: a package with one consumer and ~100 lines; premature by the rule that a package is justified by a second consumer or an imminent publish.

### Follow-ups

- Check whether `tg_mean_rows` (2D-only) should generalise to a mean over a chosen axis so mean-pooling works on `[B, T, C]`.

### Decision

Closed (2026-09-20): **Option B — create `ovg_vision` now, in Phase 2.** The owner has further vision consumers imminent (at minimum the encoder side of an image-generation project), which satisfies the second-consumer rule ahead of time; creating the package while it is ~100 lines is cheaper than extracting it later. `ovg_vision` links `ovg_nn`, mirrors `ovg_lm`'s shape (a few modules, own tests, no data loading), and initially holds `TgPatchEmbed` and the token-pooling helpers. vexilloscope becomes its first consumer and links `ovg_vision` instead of `ovg_nn`.

---

## Conv Family and Image Generation

### Overview

The ops a UNet-style image generator needs that `ovg_core` lacks, checked against `tg_ops.h` on 2026-09-20:

| Needed | Present |
|---|---|
| `conv2d`, `conv_transpose2d` | no |
| `avg_pool2d` / `max_pool2d` / nearest `upsample2d` | no |
| `group_norm` (or `batch_norm`) | no — only `layer_norm` over the last axis |
| `silu`, `sigmoid` | no — `tanh`, `relu`, `gelu` exist |
| sinusoidal timestep / position embedding | no |
| `concat` for skip connections | **yes** (added 2026-09-20) |

Above the ops: residual conv blocks, down/up-sampling blocks, an attention block over flattened spatial positions (reuse `TgSelfAttention`), and the UNet assembly. Above that, in a modality package: a noise schedule and a sampler (DDPM / DDIM), mirroring `ovg_lm`'s tokenizer + sampler pair.

### Why It Matters

This is the one target project the current platform cannot serve at all, and the one with the largest new-op surface. It is also the one most exposed to the performance and graph-size ceilings.

### Constraints

- Every new op ships with its paired backward, a CPU path, a CUDA path, and tests — the existing rule.
- `TG_MAX_DIMS = 4`. Activations `[B, C, H, W]` fit exactly. Attention inside a UNet must flatten spatial positions first (`[B, C, H, W]` → `tg_reshape` → `[B, HW, C]` → attention → reshape back); this is a reshape, not a new dimension, but it must be stated in the block's contract and tested.
- `TG_MAX_GRAPH = 8192` with a recursive `topo_sort`. A UNet's node count scales with depth × blocks × ops-per-block and is not obviously under the cap. Converting `topo_sort` to iterative and either raising or making dynamic the graph capacity should be treated as a prerequisite, not a follow-up, once a first UNet is sketched.
- Eager dispatch. A 32×32 DDPM is fine; 64×64 with attention will be slow. Batching is the first lever and must work from the start (no batch-1 designs).
- cuDNN is not a dependency today and stays off the table: it would hide exactly the tensor math this project exists to make visible.
- im2col memory: the unrolled buffer for a 3×3 filter is ~9× the input activation. At 32×32 with 128 channels that is ~1.2M floats per image; at 64×64 with 256 channels ~9.4M per image. On a 12 GB card, batch size becomes memory-bound before compute-bound — a reason to start at 32×32 and a reason the direct-kernel upgrade path must stay open behind the same `tg_conv2d` signature.

### Non-Goals

- Latent diffusion / a VAE in the first pass.
- Text conditioning / cross-attention in the first pass.
- Any performance work beyond "batched and not pathological".

### Options

#### Option A: Full conv family up front

Implement conv2d, conv_transpose2d, both pools, upsample, group_norm, silu, sigmoid, sinusoidal embedding as one spec before any model.

- Pro: complete toolbox; later models do not stall on a missing op.
- Con: speculative — some of it (max_pool, conv_transpose) may never be used if upsample+conv is preferred; large spec with no completion signal beyond "tests pass".

#### Option B: Driven by a first concrete model

Pick a small DDPM (e.g. 32×32, 3 channels, a handful of residual blocks, one attention level). Specify exactly the ops *it* needs — likely conv2d, group_norm, silu, nearest upsample, avg_pool or strided conv, sinusoidal embedding — build them, build the model, and let "the model trains and samples something recognizable" be the completion signal.

- Pro: every op has a consumer on day one; the spec has a real acceptance test; the graph-size and dimension constraints get exercised on a real network.
- Con: a second model may need an op the first did not; that is a small follow-up spec, not a redesign.

### Follow-ups

- Sketch the first UNet's op count to size it against `TG_MAX_GRAPH` before committing to the capacity change (the change is a prerequisite regardless; the sketch sets the initial capacity).
- Closed: im2col + GEMM (see Decision).
- Closed: separate `group_norm` op (see Decision).

### Decision

Closed (2026-09-20): **Option B — driven by a first concrete model.** The Phase 3 spec names a small DDPM (about 32×32 RGB, a few residual blocks, one attention level over flattened spatial positions) and builds exactly the ops it needs — expected: `conv2d` (also used strided for downsampling), `group_norm`, `silu`, nearest `upsample2d`, sinusoidal timestep embedding. `conv_transpose2d` and `max_pool2d` are not built until a model calls for them. The completion signal is that the model trains and sampling from noise produces recognizable images, not that op tests pass in isolation. The `topo_sort` / graph-capacity change is a prerequisite of the same phase.

Two implementation choices are fixed here so the spec does not reopen them:

- **`conv2d` is im2col + cuBLAS GEMM.** It rides on `tg_matmul`, the best-tested path in the library, and adds no dependency. A direct kernel is a permitted later replacement behind the same `tg_conv2d` signature if a profile shows conv dominating; cuDNN is not.
- **`group_norm` is its own op**, `tg_group_norm(a, gamma, beta, n_groups, eps)` over `[B, C, H, W]` with per-channel affine. `tg_layer_norm` is not generalised; it is on every transformer's hot path and its contract stays as is. The duplicated mean/variance arithmetic is accepted.

---

## Graph and Dimension Limits

### Overview

Three compile-time constants shape what models can be built: `TG_MAX_DIMS = 4`, `TG_MAX_PARENTS = 8`, `TG_MAX_GRAPH = 8192`, plus the recursive `topo_sort`.

### Why It Matters

They have been invisible because every model so far is a modest transformer. The image-generation direction is the first that plausibly reaches `TG_MAX_GRAPH`, and the first where `TG_MAX_DIMS` forces a design choice (spatial attention via reshape).

### Constraints

- Raising `TG_MAX_DIMS` changes `struct Tensor` layout, every shape loop, every CUDA kernel that takes `s0..s3`, and the checkpoint format. It is a large change and should not be made speculatively.
- Making `topo_sort` iterative is a contained change in `tg_train.c` with existing tests covering backward correctness.

### Non-Goals

- Dynamic-shape or dynamic-graph support.

### Options

- **Keep 4 dims; go iterative on `topo_sort`; make graph capacity grow on demand** (a `realloc`ing node list rather than a fixed array). Minimal blast radius; unblocks deep UNets; spatial attention uses reshape as documented.
- **Raise `TG_MAX_DIMS` to 5.** Only justified by a concrete need (3D conv, grouped attention with an extra axis). None identified.

### Follow-ups

- Measure the node count of the first UNet sketch and of vexilloscope's ViT to know how close current models already are.

### Decision

Closed (2026-09-20): **keep `TG_MAX_DIMS = 4`** — the owner has no plans that need a fifth dimension, and spatial attention in a UNet is handled by reshape. **`topo_sort` becomes iterative with a growable node list** as a Phase 3 prerequisite, so a UNet's depth is never bounded by a compile-time constant. Raising `TG_MAX_DIMS` is off the table until a concrete 5D need appears.

---

## Performance Posture

Not a planning topic with options — a statement of position so it is not relitigated in every spec.

OVG prioritises correctness, readability, and explicit tensor math over speed. Eager dispatch and one-kernel-per-op are accepted for now. The performance levers, in the order they should be pulled when a project actually needs them:

1. **Batching.** v2 made every op N-D and attention batched; consumers should use it (vexilloscope currently trains at batch 1).
2. **Fewer host syncs.** The `_no_sync` cross-entropy variants exist for this reason; the harness should default to them and read losses back periodically.
3. **BF16 matmul** where numerically safe; already available via `tg_cast`.
4. **Fusion of obvious pairs** (bias-add + activation, layer-norm affine) as hand-written kernels — only with a measured hotspot.

Nothing below that line (graph compilation, memory planning, cuDNN) is on the roadmap.

---

## Open Questions Not Yet Assigned to a Topic

- Closed with the harness decision: `tg_checkpoint` format v3 carries a metadata header (step, RNG seed, and a caller-supplied config blob) alongside optimizer state. The exact header fields are a Phase 1 spec question.
- Does `ovg_lm` need a KV cache before any "focused GPT" is considered done, or is O(T²) generation acceptable at the sequence lengths those projects use?
- When `ovg_vision` and `ovg_diffusion` exist, does `ovg_lm` keep its name or do all three adopt a common pattern (`ovg_text`?) — cosmetic, decide at Phase 4.
