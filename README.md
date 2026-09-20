# otto-von-grad

A reverse-mode autograd engine and neural-network toolkit written from scratch in C11. No external ML libraries.

N-D tensors up to 4D, batched multi-head attention, pre-norm transformer blocks, a GPT-style language model, optional CUDA acceleration through cuBLAS, and opt-in BF16 mixed precision.

The library is built as three layers, each a static CMake target that depends only on the one below it:

| Target | What it holds |
|---|---|
| `ovg_core` | `Tensor`, differentiable ops with paired backward functions, `tg_backward`, SGD/Adam, gradient clipping, RNG, checkpoint I/O, fatal-error hook, CUDA kernels |
| `ovg_nn` | `TgLinear`, `TgSelfAttention` (causal and encoder), `TgBlock`, `TgTransformer` with stochastic depth |
| `ovg_lm` | `TgGPT`, byte-level `TgVocab` tokenizer, argmax / top-k sampling, `tg_generate` |
| `ottovongrad` | umbrella INTERFACE target: all three |

Consumers link the narrowest layer they need. A vision model links `ovg_nn`; a language model links `ovg_lm`.

The full API reference — every signature, shape convention, and constraint — is in [`AGENTS.md`](AGENTS.md). It is written for coding agents but is the single source of truth for humans too.

---

## A taste

```c
#include "tg_ops.h"
#include "tg_train.h"

int main(void) {
    int shape_w[2] = {4, 3}, shape_x[2] = {3, 2};
    Tensor *W = tg_new(2, shape_w);   /* [4, 3] */
    Tensor *X = tg_new(2, shape_x);   /* [3, 2] */
    W->persistent = 1;                /* parameters survive tg_free_graph */
    X->persistent = 1;
    tg_fill_randn(W, 0.1f);
    tg_fill_randn(X, 1.0f);

    Tensor *Y    = tg_matmul(W, X);   /* [4, 2] */
    Tensor *loss = tg_sum(tg_pow(Y, 2.0f));

    tg_backward(loss);                /* W->grad and X->grad now populated */
    tg_sgd_step(&W, 1, 0.01f);

    tg_free_graph(loss);              /* frees Y, the pow node, and loss */
    tg_free(W);
    tg_free(X);
    return 0;
}
```

Shapes must match exactly — there is no silent broadcasting. Use `tg_reshape` and `tg_expand_dim` to make shapes line up explicitly.

---

## Build

Default preset: Visual Studio 2026, CUDA on, Release, outputs flattened into `build\`.

```powershell
cmake --preset default             # configure (fresh clone, or after CMakeLists changes)
cmake --build --preset default     # build everything
.\build\otto_von_grad_tests.exe    # 73 tests; exits 0 on all-pass
.\build\candide.exe                # GPT demo: trains on examples/data/candide.txt, generates text
```

Other presets share the same `build\` directory, so reconfigure when switching:

```powershell
cmake --preset debug && cmake --build --preset debug   # Debug + CUDA
cmake --preset cpu   && cmake --build --preset cpu     # Release, CPU-only (CUDA tests skipped)
```

Tests also run through CTest: `ctest -C Release --test-dir build`.

Requirements: CMake ≥ 3.20 and a C11 compiler. The `default` and `debug` presets additionally need the CUDA Toolkit (tested with 12.8, `sm_89`); set `-DOVG_CUDA=OFF` or use the `cpu` preset without it.

---

## CUDA

With `OVG_CUDA=ON`, every op dispatches to a GPU kernel when its inputs are on the device, matmul goes through cuBLAS, and `OVG_CUDA_ENABLED` is defined for consumers. Move a tensor with `tg_to_cuda(t)` / `tg_from_cuda(t)` and release device memory with `tg_cuda_free(t)`. Every op keeps a correct CPU path; BF16 is CUDA-only.

---

## Reproducibility and errors

The RNG seed is printed at startup so any run can be replayed with `tg_seed(...)`:

```
[ovg] rng seed: 0x5f3759df
```

Shape mismatches and other misuse are fatal: `ovg_fatal()` prints the message and calls `exit(1)`. Install a hook to log or capture it first:

```c
#include "ovg_error.h"

static void my_handler(const char *msg) { /* log, then return to let exit(1) run */ }
ovg_set_fatal_handler(my_handler);
```

The test suite uses this hook with `setjmp`/`longjmp` to assert on error paths in-process; see `tests/README.md`.

---

## Usage as a dependency

Side-by-side checkout:

```cmake
if(NOT TARGET ottovongrad)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../otto-von-grad"
                     "${CMAKE_CURRENT_BINARY_DIR}/otto-von-grad")
endif()
target_link_libraries(your_target PRIVATE ovg_nn)   # or ovg_core / ovg_lm / ottovongrad
```

Or via FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(ottovongrad
    GIT_REPOSITORY https://github.com/DigitalMeatbag/otto-von-grad.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(ottovongrad)
target_link_libraries(your_target PRIVATE ottovongrad)
```

Public headers are in `include/ovg/` and are included by bare name (`#include "tg_ops.h"`). `OVG_CUDA` is inherited from the parent project's cache; set it before `add_subdirectory` if you need a different value.

---

## Layout

```text
include/ovg/   public headers (one per module; prefix tg_, plus ovg_error.h)
src/           implementation, CUDA kernels, and two private headers
tests/         test suite (ovg_test.h macros + one file per module)
examples/      candide.c demo and its corpus
```
