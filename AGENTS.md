# AGENTS.md

# otto-von-grad

A lightweight tensor autograd engine written in C.

Implements reverse-mode autodiff, multi-head self-attention, transformer blocks,
and a GPT-style language model — all from scratch with no external ML dependencies.

Optional CUDA acceleration via `OVG_CUDA=ON`.

---

## Repository Structure

```text
otto-von-grad/
  src/
    tg_tensor.c / tg_tensor.h       — Tensor struct, lifecycle, print helpers
    tg_ops.c / tg_ops.h             — all differentiable ops + backward functions
    tg_train.c / tg_train.h         — tg_backward (topo sort), tg_sgd_step, tg_adam_step
    tg_mlp.c / tg_mlp.h             — TgLinear convenience layer
    attention.c / attention.h       — TgSelfAttention, multi-head forward
    tg_block.c / tg_block.h         — TgBlock (pre-norm transformer block)
    tg_transformer.c / tg_transformer.h  — TgTransformer (stack of blocks)
    tg_gpt.c / tg_gpt.h             — TgGPT (embeddings + transformer + output projection)
    tg_cuda.cu / tg_cuda.h          — CUDA tensor lifecycle (tg_to_cuda, tg_from_cuda)
    cuda_ops.cu / cuda_ops.h        — CUDA kernels for all ops
    ovg_error.c / ovg_error.h       — centralized fatal error handler (ovg_fatal, ovg_set_fatal_handler)
    main.c                          — demo: GPT training on candide.txt
  tests/
    ovg_test.h                      — minimal test assertion macros
    test_ops.c                      — ops forward + backward correctness
    test_train.c                    — backward pass, grad accumulation, optimizer step
    test_attention.c                — causal + encoder attention
    test_gpt.c                      — GPT forward shape, param collection
    test_main.c                     — test runner entry point
  legacy/
    value.c / value.h               — scalar autograd (learning exercise, not compiled)
    mlp.c / mlp.h                   — scalar MLP (learning exercise, not compiled)
  candide.txt                       — corpus for GPT character-level demo
```

---

## Tensor Autograd Ops

All ops in `tg_ops.c / tg_ops.h`. Tensor struct and lifecycle in `tg_tensor.h`.

`TG_MAX_PARENTS = 8`, `TG_MAX_GRAPH = 4096`.

### Arithmetic

* `tg_add(a, b)`       — element-wise
* `tg_sub(a, b)`       — element-wise
* `tg_mul(a, b)`       — element-wise
* `tg_pow(a, p)`       — element-wise a^p
* `tg_scale(a, s)`     — element-wise a * s
* `tg_matmul(a, b)`    — [m,k] @ [k,n] → [m,n]
* `tg_transpose(a)`    — [r x c] → [c x r]

### Activations

* `tg_tanh(a)`
* `tg_relu(a)`

### Reductions

* `tg_sum(a)`          — all elements → [1x1]
* `tg_mean(a)`         — all elements → [1x1]
* `tg_mean_rows(a)`    — [R x C] → [1 x C], mean over rows

### Slicing / Combining

* `tg_slice_cols(a, start, end)` — [R x C] → [R x (end-start)]
* `tg_concat_cols(parts, n)`     — n × [R x C] → [R x (n*C)]
* `tg_embed(weight, ids, T)`     — gather T rows from weight [V x C] by integer ids → [T x C]

### Attention / Normalization

* `tg_causal_mask(scores)`       — mask future columns in [seq x seq]
* `tg_layer_norm_rows(a, eps)`   — normalize each row over columns
* `tg_softmax_rows(a)`           — softmax along each row

### Regularization

* `tg_dropout(a, p)` — inverted dropout; pass-through when `tg_training == 0` or `p == 0`

### Loss

* `tg_cross_entropy(logits, targets)` — mean CE [1x1], targets one-hot

---

## Training API

```c
void  tg_backward(Tensor *root);                       // full backward + zero grads
void  tg_backward_accum(Tensor *root);                 // backward, accumulate grads (no zero)
void  tg_zero_grads(Tensor **params, int n);           // zero grad arrays for param list
void  tg_sgd_step(Tensor **params, int n, float lr);
void  tg_adam_step(Tensor **params, float **m, float **v, int n,
                   float lr, int t, float b1, float b2, float eps);

// GPU Adam — moment buffers stay on device
void  tg_adam_step_gpu(Tensor **params, float **m_gpu, float **v_gpu, int n,
                       float lr, int t, float b1, float b2, float eps);

void  tg_free_graph(Tensor *root);                     // free non-persistent graph tensors
```

---

## Attention Module

File: `attention.c / attention.h`

```c
typedef struct {
    Tensor *Wq, *Wk, *Wv, *Wo;
    int embed_dim, head_dim, n_heads, causal;
} TgSelfAttention;

TgSelfAttention tg_attention_create(int embed_dim, int n_heads);          // causal = 1
TgSelfAttention tg_attention_create_encoder(int embed_dim, int n_heads);  // causal = 0
void            tg_attention_free(TgSelfAttention *a);
Tensor         *tg_attention_forward(TgSelfAttention *a, Tensor *X);
```

---

## Transformer Block

File: `tg_block.c / tg_block.h`

Pre-norm architecture (LayerNorm → Attention → residual → LayerNorm → FFN → residual).

```c
typedef struct {
    TgSelfAttention attn;
    Tensor *W1, *B1, *W2, *B2;
    int embed_dim, hidden_dim;
    float dropout;
} TgBlock;

TgBlock  tg_block_create(int embed_dim, int hidden_dim, int seq_len, int n_heads);
void     tg_block_free(TgBlock *b);
Tensor  *tg_block_forward(TgBlock *b, Tensor *X);
```

Note: `tg_block_create` always creates **causal** blocks. For encoder (non-causal) blocks,
construct `TgSelfAttention` directly via `tg_attention_create_encoder`.

Note: `B1` and `B2` are expanded to `[seq_len x ...]` — no broadcasting.

---

## Transformer Stack

File: `tg_transformer.c / tg_transformer.h`

```c
TgTransformer tg_transformer_create(int n_blocks, int embed_dim, int hidden_dim,
                                    int seq_len, int n_heads);
Tensor       *tg_transformer_forward(TgTransformer *t, Tensor *X);
```

---

## GPT Model

File: `tg_gpt.c / tg_gpt.h`

```c
TgGPT   tg_gpt_create(int vocab_size, int embed_dim, int hidden_dim,
                      int seq_len, int n_blocks, int n_heads);
Tensor *tg_gpt_forward(TgGPT *g, const int *token_ids);   // int[seq_len] → [T x V] logits
int     tg_gpt_collect_params(TgGPT *g, Tensor **params, int max_params);
```

---

## CUDA Support

Enabled via `OVG_CUDA=ON` at configure time. When enabled:
- `OVG_CUDA_ENABLED` is defined globally (propagates to consumers via CMake `PUBLIC`)
- All ops have matching CUDA kernels in `cuda_ops.cu`
- `tg_to_cuda(t)` uploads a tensor to device; `tg_from_cuda(t)` syncs back to host
- Forward/backward ops automatically dispatch to GPU kernels when `t->on_cuda == 1`

VS 2026 (MSVC 14.50+) requires `-allow-unsupported-compiler` — already set in CMakeLists.txt.

---

## Build Commands

```powershell
# CPU only
cmake -B build -G Ninja
cmake --build build
.\build\otto_von_grad.exe           # GPT demo (trains on candide.txt)
.\build\otto_von_grad_tests.exe     # test suite — 23 tests, exits 0 on all-pass

# With CUDA
cmake -B build -G Ninja -DOVG_CUDA=ON
cmake --build build
.\build\otto_von_grad_tests.exe
```

On VS 2026 the configure step handles the unsupported-compiler flag automatically.

---

## Error Handling

All fatal errors in the library call `ovg_fatal()` (in `ovg_error.h/c`) instead of `exit(1)` inline.
Consumers can install a pre-exit hook:

```c
#include "ovg_error.h"
ovg_set_fatal_handler(my_logger);  // call before any threads are spawned (not thread-safe)
```

The hook receives the formatted message string. `ovg_fatal` **always calls `exit(1)` after the
hook returns** — the handler is a logging/capture hook, not a recovery path.

Tests use `setjmp`/`longjmp` to capture fatal calls in-process. See `tests/test_ops.c` for the
pattern. Call `ovg_set_fatal_handler(NULL)` after each capture block to restore the default and
reset the re-entrancy guard.

---

## Include Style

New code should include the specific sub-header it needs:

```c
#include "tg_ops.h"    // ops + Tensor struct (via tg_tensor.h)
#include "tg_train.h"  // tg_backward, tg_sgd_step (+ Tensor struct)
```

Files that only perform forward passes (no backward) need only `tg_ops.h`.

---

## Coding Style

* Straightforward C (C11). No OOP patterns, no macro-heavy metaprogramming.
* Prefer explicit tensor shapes in comments.
* Shape mismatches should fail loudly — no silent broadcasting.
* Keep memory ownership obvious. `persistent = 1` marks tensors that survive `tg_free_graph`.
* Correctness and inspectability over performance.

---

## Known Constraints

* **No broadcasting** — bias tensors are manually expanded to full shapes.
* **2D tensors only** — all ops assume `[rows x cols]`.
* **Eager dispatch** — one kernel launch per op; no graph compilation or kernel fusion.
* **Performance** is not the current priority. Correctness and readability are.
* **topo_sort is recursive** — stack depth equals graph depth. Safe for current models (depth ~200). If `TG_MAX_GRAPH` is raised and model depth grows large, convert to iterative.

---

## Important Guidance For Agents

When modifying code:

* Preserve explicit tensor math — do not hide operations behind abstractions.
* Preserve the backward function paired with each op in `tg_ops.c`.
* Do not introduce external ML libraries.
* Do not add broadcasting without explicit discussion — it changes semantics for existing callers.
* When adding a new op, add both the forward function and its `_backward` counterpart.
* When adding a CUDA kernel, ensure the CPU path remains correct and the dispatch logic in the op is symmetric.
