# CODEX.md

Codex instructions for **otto-von-grad**, a from-scratch tensor autograd engine in
C11. Use `AGENTS.md` as the broad architecture reference and treat this file as
the Codex workflow layer.

## Working Style

- Read nearby code before editing; this project values explicit tensor math and
  inspectability over clever abstractions.
- Keep changes narrow and easy to review. Avoid unrelated refactors, formatting
  churn, or metadata updates.
- Preserve user work in the git tree. If unrelated files are dirty, leave them
  alone.
- Prefer `rg` / `rg --files` for repo searches.
- Use `apply_patch` for manual edits.

## Project Invariants

- C11 only; do not introduce external ML dependencies.
- All tensor ops are 2D `[rows x cols]`.
- No silent broadcasting. Bias tensors are explicitly expanded to full shapes.
- Shape mismatches should fail loudly.
- `persistent = 1` marks tensors that survive `tg_free_graph`.
- Correctness and readability beat performance.
- `TG_MAX_PARENTS = 8` and `TG_MAX_GRAPH = 4096`.

## Autograd Rules

- Tensor ops live in `src/tg_ops.c` / `src/tg_ops.h`.
- Every differentiable op needs both a forward function and its paired
  `_backward` implementation.
- When editing an existing op, inspect both the forward path and backward path.
- Keep CPU behavior correct even when adding or changing CUDA dispatch.
- If an op allocates auxiliary data for backward, make ownership obvious and
  verify `tg_free_graph` can clean up non-persistent graph tensors safely.

## CUDA Rules

- CUDA is optional and gated by `OVG_CUDA=ON`.
- `OVG_CUDA_ENABLED` is propagated from CMake when CUDA is enabled.
- CPU and CUDA dispatch should stay symmetric: if an op has GPU support, both
  forward and backward behavior should match the CPU path.
- Do not make CUDA required for normal builds.

## Build And Verification

CPU build:

```powershell
cmake -B build -G Ninja
cmake --build build
.\build\otto_von_grad_tests.exe   # test suite — exits 0 on all-pass
.\build\otto_von_grad.exe         # GPT character-level demo (candide.txt)
```

CUDA build:

```powershell
cmake -B build -G Ninja -DOVG_CUDA=ON
cmake --build build
.\build\otto_von_grad_tests.exe
```

For code changes, at minimum run the CPU build and `otto_von_grad_tests.exe` unless
the change is docs-only or the local toolchain is unavailable. For CUDA-specific
changes, attempt the CUDA configure/build path and report if the environment cannot
support it.

## File Map

- `src/tg_tensor.c`, `src/tg_tensor.h`: tensor lifecycle, allocation, printing.
- `src/tg_ops.c`, `src/tg_ops.h`: differentiable tensor operations and backward
  functions.
- `src/tg_train.c`, `src/tg_train.h`: topo-sort backward pass, grad zeroing,
  SGD, Adam, graph cleanup.
- `src/tg_mlp.c`, `src/tg_mlp.h`: linear layer helper.
- `src/attention.c`, `src/attention.h`: causal and encoder self-attention.
- `src/tg_block.c`, `src/tg_block.h`: pre-norm transformer block, including
  encoder block creation.
- `src/tg_transformer.c`, `src/tg_transformer.h`: stack of transformer blocks.
- `src/tg_gpt.c`, `src/tg_gpt.h`: token/positional embeddings, transformer,
  output projection.
- `src/tg_cuda.cu`, `src/tg_cuda.h`: tensor CUDA upload/download lifecycle.
- `src/cuda_ops.cu`, `src/cuda_ops.h`: CUDA kernels for tensor ops.
- `src/ovg_error.c`, `src/ovg_error.h`: centralized fatal error handler (`ovg_fatal`,
  `ovg_set_fatal_handler`).
- `src/tg_rng.c`, `src/tg_rng.h`: RNG state and seeding (`tg_seed`, `tg_seed_from_entropy`,
  `tg_rng_xorshift32`). Owns both the `rand()` seed and the xorshift32 dropout state.
- `src/tg_cuda.cu`, `src/tg_cuda.h`: tensor CUDA upload/download lifecycle.
- `src/cuda_ops.cu`, `src/cuda_ops.h`: CUDA kernels for tensor ops.
- `src/main.c`: GPT character-level demo using `candide.txt`.
- `tests/ovg_test.h`: test assertion macros (`OVG_CHECK`, `OVG_CHECK_NEAR`, etc.).
- `tests/test_main.c`: test runner entry point.
- `tests/test_ops.c`: ops forward/backward + error-path tests.
- `tests/test_train.c`: optimizer and backward pass tests.
- `tests/test_attention.c`: causal and encoder attention tests.
- `tests/test_gpt.c`: GPT shape and param collection tests.
- `legacy/`: historical scalar autograd learning exercises (not compiled).

## Include Style

New code should include the narrow header it needs:

```c
#include "tg_ops.h"    // tensor ops and Tensor via tg_tensor.h
#include "tg_train.h"  // backward, optimizers, graph cleanup
```

Files that only perform forward passes usually need only `tg_ops.h`.

## Common Pitfalls

- Do not add broadcasting as a convenience fix.
- Do not hide tensor math behind new abstractions unless the surrounding code
  already uses that pattern.
- Do not manually free intermediate graph tensors after a training step; use
  `tg_free_graph(loss)` for non-persistent graph nodes.
- Do not forget headers when adding public functions.
- Do not update CUDA kernels without checking that the CPU fallback still works.
