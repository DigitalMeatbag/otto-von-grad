# otto-von-grad Platform Foundation Document

> **Purpose:** This document captures the intent, current state, target layering, open design questions, and planning order for turning otto-von-grad (OVG) from "an autograd engine that happens to ship a GPT" into the platform on which focused language models, vision transformers, and image generators are built. It is a living foundation document, not yet an implementation specification. `AGENTS.md` remains the authoritative description of what exists *today*; this document describes where it is going and why.

---

## Vision

> This section describes the *result* of the planned work in product terms, for use in product-shaped
> investigations (who wants this, what it competes with, what it must not claim). The engineering intent
> and plan follow it.

### One line

otto-von-grad is the shared, from-scratch C11 machine-learning substrate behind a family of focused,
self-contained applications — built to make those applications easier to audit, reproduce, and deploy
with a small, controlled dependency surface.

### Thirty seconds

otto-von-grad is not intended to be sold as a general-purpose ML framework. It is the internal platform
used to build focused products for processes where auditability, offline operation, reproducibility, or
dependency independence matter. Instead of carrying a Python runtime and a large framework into every
product, each application is built on a compact C stack whose tensor operations, gradients, kernels, and
model structure can be inspected end to end.

The engine and training harness provide the common machinery each application would otherwise rebuild:
autograd, optimizers, schedules, accumulation, and checkpoints that resume where they stopped. Lean
modality packages add only the reusable language, vision, or generation components demanded by concrete
applications. Product-specific data handling, policy, reporting, interfaces, and deployment behavior
remain in the application repositories.

The result is an application-building advantage: narrow models trained on a single consumer GPU, shipped
inside products with controlled operational boundaries, and understood deeply enough to support credible
claims about how each application was built, what it depends on, and how it behaves.

### What makes it product infrastructure rather than a hobby

OVG is not the product a customer buys; the focused application built on it is. OVG earns its place by
making successive applications faster to build and easier to own end to end. Its differentiator is
**legibility**: the implementation is inspectable down to the arithmetic, CPU deployments can have a
very small dependency surface, and optional acceleration dependencies are explicit. That foundation
helps each application support stronger claims about auditability, offline operation, reproducibility,
and deployment control than would be practical with a general-purpose ML stack.

The platform succeeds when product code is dominated by the customer's process and policy rather than
repeated ML plumbing, and when a shipped application can be reviewed, rebuilt, and operated within a
clearly defined boundary. Operator count, modality breadth, and third-party framework adoption are not
success metrics by themselves.

### Claims the pitch may make, and the evidence behind each

| Claim | Basis today | Watch |
|---|---|---|
| Complete stack in a few thousand lines | ~7k lines including tests and CUDA kernels | Roughly doubles through Phase 3; re-count before quoting a number. |
| Small, controlled dependency surface | CPU uses the C toolchain/runtime; acceleration adds the CUDA toolkit, runtime, driver, and cuBLAS | Keep dependencies explicit and bounded; do not describe this as "no supply chain." |
| Every op has its gradient beside it | Enforced coding rule; 120 tests | — |
| Exact resume from a v3 checkpoint | Built (Phase 1, 2026-09-20): `tg_checkpoint_save_run` / `load_run` carry Adam moments, step, and the RNG word; `test_checkpoint_exact_resume` proves a stopped-and-resumed run matches an uninterrupted one bitwise on CPU | Library randomness only; application-side `rand()` is not restored. v2 files load weights only. |
| Independently usable modality packages | `ovg_lm` and `ovg_vision` exist; `ovg_diffusion` Phase 3b | Two of three today. |
| Proven by real applications | `lambda` (GPT) and `vexilloscope` (ViT) both build; vexilloscope has no trained weights (the retrain was dropped, see Brownfield Baseline) | Image generation has no application yet. |
| Trains on a single consumer GPU | RTX 4070 Super, 12 GB, is the reference machine | Yes for focused models; see below. |

### Claims the pitch must not make

- **Not "fast."** Dispatch is eager, one kernel per op, no fusion. A batch-1 ViT with 1,025 tokens
  runs ~3 steps/s on the reference GPU. The pitch leans on understanding and ownership, not throughput.
  If an investigation surfaces a buyer whose need is throughput, that is a signal against fit, not a gap
  to paper over.
- **Not "large models."** Focused models are the design point: single GPU, thousands to low millions of
  parameters, one task. Nothing in the plan targets scale.
- **Not "production inference infrastructure."** No serving layer, no quantisation, no export format
  beyond the library's own checkpoint. A model built on OVG ships as a C program that links OVG.

### Who this is plausibly for

Product investigations should test these rather than assume them:

- **Operators of constrained or controlled processes** — embedded targets, air-gapped systems, and
  environments where a large runtime and dependency tree are difficult to approve or operate.
- **Teams that need to own an application end to end** — including its model, deployment boundary,
  update path, and evidence about how it was trained and evaluated.
- **Users with a narrow, valuable workflow** that can be served by a focused classifier, reasoner, or
  generator running locally instead of a general-purpose remote model.

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

State as of commit `5f83f16` (the range `93ff570`..`5f83f16`, merged to `master` and pushed):

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
  - `vexilloscope` was stuck on the v1 API since the v2 N-D migration (2026-05-28). Its port is committed in that repo (`ef4b230`): a mechanical pass (`->data` → `TG_DATAF`, `->rows/->cols` → `shape[]`, two-int `tg_new` → N-D form), `tg_concat_rows` → `tg_concat`, `tg_row_slice` → `tg_slice` + `tg_reshape` on the now-3D encoder output, `cuda_smoke.c` deleted. It compiles with zero warnings and links `ovg_nn` only. Its v1 weights file cannot load — v1 stored LayerNorm affine and FFN bias parameters pre-tiled to `[seq_len, C]`, v2 stores `[1, C]` — so a retrain would be required before it could classify again. **The owner dropped that retrain on 2026-09-20**; vexilloscope stays a build-and-migration consumer with no deployed weights until a retrain is chosen.
- **Includes remain bare.** The namespaced form (`#include "ovg/tg_ops.h"`, `include/ovg/lm/`) is deferred until a modality package is actually about to be published separately.
- **Phase 1 (training harness, 2026-09-20, `docs/SPEC_PLATFORM_PHASE1_HARNESS.md`):** `ovg_core` gained `TgAdam` (`tg_optim.h`), `tg_lr_warmup_cosine` / `tg_lr_warmup_linear` (`tg_sched.h`), the eval guard and `TgMeter` (`tg_train.h`), `tg_rng_get_state` / `set_state`, and checkpoint format v3 (`tg_checkpoint_info` / `save_run` / `load_run` with `TG_LOAD_RESUME` / `TG_LOAD_INIT_FROM_WEIGHTS`; v2 files still load weights-only). Tests: 101 (CUDA) / 89 (CPU). All three consumers train through the harness: `candide` and `lambda` resume exactly from periodic `save_run` checkpoints (lambda skips a phase already at its step budget); vexilloscope's `main.c` shrank from 1,294 to 1,257 lines and keeps its own weight format until Phase 2.
- **Phase 2 (vision, 2026-09-20, `docs/SPEC_PLATFORM_PHASE2_VISION.md`):** `TgLinear` fixed and renamed (`tg_mlp.h` → `tg_linear.h`; bias `[1, n_out]` expanded at forward, no `batch` field, no out-parameter). `ovg_vision` created (`TgPatchEmbed`, `tg_pool_cls`, `tg_pool_mean_tokens`; links `ovg_nn`) and linked by the `ottovongrad` umbrella beside `ovg_lm`. Tests: 120 (CUDA) / 105 (CPU). vexilloscope links `ovg_vision`, dropped `patch_embedding.c` and the `VXWT` format, and trains through `tg_checkpoint_save_run` with auto-resume from `--weights` (`vit.c` 362 → 191 lines; `main.c` 1,257 → 1,335, the resume block being the growth). The retrain that would produce the first v3 `vit_weights.bin` was dropped by the owner on 2026-09-20; it is not a prerequisite of Phase 3a and no phase is gated on it.

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
                               conv residual / down / up blocks
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
| does not assume a data type, but every plausible consumer shares one modality (a patch embedding, token pooling) | the matching modality package |
| loads, augments, or describes data; parses a CLI; writes reports | the application repo |

If a lower layer needs something from a higher one, the thing is in the wrong layer: move it down rather than adding the include.

**Second-consumer rule.** A modality package is created when a second consumer for it exists or is imminent, or when it is about to be published on its own. Until then, candidate modules stay in `ovg_nn` if they do not assume a data type, or in the first consumer's repo if they do.

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

Scope: the training harness (optimizer state as a struct, LR schedules, gradient accumulation helper, an evaluation-loop skeleton, checkpoint format v3 with optimizer state, RNG state and a metadata header, exact-resume on by default with an explicit init-from-weights load mode), followed by migrating `examples/candide.c`, `../lambda`, and `../vexilloscope` onto it. Completion signal: all three consumers train through the harness, and vexilloscope's `main.c` shrinks by the harness code it no longer owns.

### Phase 1 Non-Goals

- Any new op beyond what the harness itself needs.
- `TgPatchEmbed`, `ovg_vision`, or other vision blocks.
- The conv family.
- Namespaced includes or package extraction.
- Performance work (batching in vexilloscope, kernel fusion).

### Later Specs

```text
SPEC_PLATFORM_PHASE2_VISION.md       # TgLinear fix + rename; new ovg_vision target with TgPatchEmbed + pooling; vexilloscope adopts both
SPEC_PLATFORM_PHASE3A_CONV.md        # iterative topo_sort + growable graph; conv2d, group_norm, silu, upsample2d, sinusoidal embedding; conv residual/down/up blocks in ovg_nn — sized by the DDPM sketch
SPEC_PLATFORM_PHASE3B_DIFFUSION.md   # new ovg_diffusion target (UNet assembly, noise schedule, sampler); the first DDPM trains and samples recognizably
SPEC_PLATFORM_PHASE4_PACKAGES.md     # namespaced includes, extraction readiness
```

Names and boundaries may shift; the principle should not: spec only the next implementable slice.

---

## Decision Log

| Topic | Status | Decision |
|---|---|---|
| Four-tier layering | Closed | engine+harness (`ovg_core`) → building blocks (`ovg_nn`) → modality packages (`ovg_lm`, later `ovg_vision`, `ovg_diffusion`) → applications in sibling repos. Dependencies strictly downward. |
| Placement rule | Closed | op/optimizer/schedule → core; reusable parameterized module → nn; assumes a modality, or reusable only within one modality → package; data/CLI/reports → app. Packages are created under the second-consumer rule. |
| `ovg_lm` as the package template | Closed | New modality packages copy `ovg_lm`'s shape: a few modules, links `ovg_nn`, own tests, no data loading. |
| `ottovongrad` umbrella | Closed | Kept indefinitely as the everything-included default; consumers are encouraged but not required to link a narrower target. |
| Bare vs. namespaced includes | Closed (deferred) | Stay bare until a package is about to be published separately; then switch in one commit, verifiable by zero `C1083` across consumers. |
| Training harness placement | Closed | In `ovg_core`, finishing `tg_train.h`. Exact-resume is default on: checkpoints save optimizer state, step, and RNG state; load takes an explicit mode (resume, or init-from-weights for warm starts); parameter-only v2 files still load with moments zeroed and step reset. |
| `TgLinear` fate | Closed | Fix in place and rename to `tg_linear.h`: bias `[1, n_out]` expanded at forward, no baked-in batch, no out-parameter. |
| Conv family scope | Closed | Driven by a first concrete DDPM (~32×32 RGB, small UNet, one attention level). First `conv2d` is im2col + cuBLAS GEMM, no cuDNN. `group_norm` is a separate op; `layer_norm` untouched. `ovg_diffusion` (UNet assembly, schedule, sampler) is created in Phase 3b so the DDPM can be trained and sampled; Phase 3a is the graph change, the ops, and the conv blocks. |
| Vision-block placement | Closed | Create `ovg_vision` now (Phase 2). Further vision consumers are imminent, which satisfies the second-consumer rule ahead of time. |
| Graph and dimension limits | Closed | `TG_MAX_DIMS` stays 4 — no plans need more. `topo_sort` goes iterative with growable graph capacity as a Phase 3a prerequisite. |

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
| Eval loop | yes | yes | yes |
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
- Con: a fourth target for a few hundred lines; every consumer links it anyway; the `ovg_nn`-doesn't-need-it distinction buys nothing today since nothing links `ovg_nn` without training.
- Con: `tg_train.h` (backward, SGD, Adam, clip) is already in core, so the split would either move existing code up or leave the optimizer family straddling two targets.

### Follow-ups

- Closed: lambda's curriculum loop uses a constant LR (3e-4) with no schedule, and chains phases by loading phase N−1's checkpoint as phase N's starting weights (`lambda/src/main.c:108-113`). It needs nothing from the schedule API; it needs the init-from-weights load mode (see Decision).
- Closed: optimizer state is saved by default (see Decision).

### Decision

Closed (2026-09-20): **Option A.** The harness lives in `ovg_core`, finishing `tg_train.h`; the split in Option B has no consumer that would benefit from it. Revisit only if a consumer ever wants `ovg_nn` without training.

**Exact-resume is default on.** The owner regularly stops and restarts long runs, so a resumed run must continue from the same optimizer state, not from zeroed moments. Consequences for the Phase 1 spec:

- The checkpoint gains an optimizer-state section (Adam `m`/`v` per parameter and the step counter — schedules are pure functions of step, so no separate schedule position is stored) and a small metadata header: format version, RNG state, and a caller-supplied config blob. The exact header fields are a Phase 1 spec question. RNG state is the xorshift32 word: `rand()` is used only by the init-time fill helpers, so every per-step draw (dropout, drop-path, sampling) is on the xorshift stream and one word restores it. This is format v3 with a new magic; `tg_checkpoint_load` continues to accept v2 parameter-only files, re-initializes moments to zero, resets the step counter to 0 (Adam's bias correction depends on it), and reports that it did so.
- **Load takes an explicit mode.** `resume` (the default) restores parameters, optimizer state, step, and RNG state. `init-from-weights` restores parameters only and leaves the optimizer fresh at step 0. Warm-starting one run from another's checkpoint — lambda's curriculum phases are the existing case — uses the second; the default must never silently carry a previous run's moments and step into a new run.
- Saving optimizer state is the default; a flag disables it for "weights-only" exports (inference artifacts, sharing). Whether a weights-only export is written as v2 or as v3 with an empty optimizer section is a Phase 1 spec question.
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

- Done (2026-09-20, Phase 2): renamed to `tg_linear.h` / `tg_linear.c` and fixed in the same commit; `tg_mlp.h` no longer exists and nothing includes it.

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
- Convolutional patch embedding (that is Phase 3a territory).

### Options

#### Option A: `TgPatchEmbed` + `tg_pool_cls` / `tg_pool_mean_tokens` in `ovg_nn`

- Pro: immediately reusable; vexilloscope shrinks; the ViT recipe becomes "patch embed → transformer → pool → linear".
- Con: `ovg_nn` starts to hold things that are only used by vision — but they do not *assume* pixels, only a token layout.

#### Option B: Create `ovg_vision` now and put them there

- Pro: cleaner story for later extraction.
- Con: a package with one consumer and ~100 lines; premature by the rule that a package is justified by a second consumer or an imminent publish.

### Follow-ups

- Closed (2026-09-20, `docs/SPEC_PLATFORM_PHASE2_VISION.md` → Decisions this spec makes): `tg_mean_rows` stays 2D; `tg_pool_mean_tokens` composes `tg_transpose → tg_reshape → tg_mean_rows → tg_reshape`. Two extra full-size copies per forward, accepted under the Performance Posture; a `tg_mean_axis` in `ovg_core` behind the same helper signature is permitted later if a profile asks for it.

### Decision

Closed (2026-09-20): **Option B — create `ovg_vision` now, in Phase 2.** The owner has further vision consumers imminent (at minimum the encoder side of an image-generation project), which satisfies the second-consumer rule ahead of time; creating the package while it is ~100 lines is cheaper than extracting it later. `ovg_vision` links `ovg_nn`, mirrors `ovg_lm`'s shape (a few modules, own tests, no data loading), and initially holds `TgPatchEmbed` and the token-pooling helpers. vexilloscope becomes its first consumer and links `ovg_vision` instead of `ovg_nn`. These modules do not assume pixels, only a token layout; they are placed in `ovg_vision` under the placement rule's modality-consumer clause, not because they assume a data type.

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

Above the ops, in `ovg_nn`: residual conv blocks, down/up-sampling blocks, and an attention block over flattened spatial positions (reuse `TgSelfAttention`). Above that, in `ovg_diffusion`: the UNet assembly (it takes a timestep embedding, so it assumes noise timesteps), a noise schedule, and a sampler (DDPM / DDIM), mirroring `ovg_lm`'s model + tokenizer + sampler shape.

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

Closed (2026-09-20): **Option B — driven by a first concrete model.** The Phase 3 spec names a small DDPM (about 32×32 RGB, a few residual blocks, one attention level over flattened spatial positions) and builds exactly the ops it needs — expected: `conv2d` (also used strided for downsampling), `group_norm`, `silu`, nearest `upsample2d`, sinusoidal timestep embedding. `conv_transpose2d` and `max_pool2d` are not built until a model calls for them. The completion signal is that the model trains and sampling from noise produces recognizable images, not that op tests pass in isolation. The `topo_sort` / graph-capacity change is a prerequisite of the same phase. Because the signal requires a schedule and a sampler, **`ovg_diffusion` is created in Phase 3**, holding the UNet assembly, the noise schedule, and the sampler; the conv blocks below it go in `ovg_nn`. Phase 4 is then namespaced includes and extraction readiness only.

Phase 3 is specified as two slices so that each stays implementable on its own: **3a** — the iterative `topo_sort` and growable graph, the ops listed above, and the conv residual/down/up blocks in `ovg_nn`, with the op set fixed by a sketch of the 3b model and a completion signal of "op tests pass, the blocks build, and a UNet-shaped graph exceeds the old 8192-node cap without fataling"; **3b** — the `ovg_diffusion` target and the DDPM itself, carrying the recognizable-samples signal. The sketch is written first, as part of 3a's spec, so 3a builds nothing 3b does not use.

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

Closed (2026-09-20): **keep `TG_MAX_DIMS = 4`** — the owner has no plans that need a fifth dimension, and spatial attention in a UNet is handled by reshape. **`topo_sort` becomes iterative with a growable node list** as a Phase 3a prerequisite, so a UNet's depth is never bounded by a compile-time constant. Raising `TG_MAX_DIMS` is off the table until a concrete 5D need appears.

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

- Does `ovg_lm` need a KV cache before any "focused GPT" is considered done, or is O(T²) generation acceptable at the sequence lengths those projects use?
- When `ovg_vision` and `ovg_diffusion` exist, does `ovg_lm` keep its name or do all three adopt a common pattern (`ovg_text`?) — cosmetic, decide at Phase 4.
