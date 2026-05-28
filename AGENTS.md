# AGENTS.md

# otto-von-grad

A lightweight tensor autograd engine written in C (C11).

Implements reverse-mode autodiff, N-D tensors (up to 4D), batched multi-head self-attention, transformer blocks, a GPT-style language model, and optional CUDA acceleration via cuBLAS. No external ML libraries.

Optional CUDA acceleration via `OVG_CUDA=ON`.

---

## Repository Structure

```text
otto-von-grad/
  src/
    tg_tensor.c / tg_tensor.h       — Tensor struct, lifecycle, print helpers, tg_numel, TG_DATAF
    tg_ops.c / tg_ops.h             — all differentiable ops + backward functions
    tg_train.c / tg_train.h         — tg_backward (topo sort), tg_sgd_step, tg_adam_step
    tg_mlp.c / tg_mlp.h             — TgLinear convenience layer
    attention.c / attention.h       — TgSelfAttention, batched multi-head forward
    tg_block.c / tg_block.h         — TgBlock (pre-norm transformer block)
    tg_transformer.c / tg_transformer.h  — TgTransformer (stack of blocks)
    tg_gpt.c / tg_gpt.h             — TgGPT, TgGPTConfig (embeddings + transformer + output projection)
    tg_tokenizer.c / tg_tokenizer.h — TgVocab, tg_read_file, tg_vocab_build/encode/decode, tg_tokenize
    tg_sample.c / tg_sample.h       — tg_sample_argmax, tg_sample_topk, tg_generate
    tg_checkpoint.c / tg_checkpoint.h — tg_checkpoint_save / tg_checkpoint_load (binary format v2)
    tg_cuda.cu / tg_cuda.h          — CUDA tensor lifecycle (tg_to_cuda, tg_from_cuda)
    cuda_ops.cu / cuda_ops.h        — CUDA kernels for all ops + cuBLAS dispatch
    ovg_error.c / ovg_error.h       — centralized fatal error handler (ovg_fatal, ovg_set_fatal_handler)
    tg_rng.c / tg_rng.h             — RNG state and seeding (tg_seed, tg_seed_from_entropy)
    main.c                          — GPT training demo (candide.txt); saves/resumes checkpoints
  tests/
    ovg_test.h                      — minimal test assertion macros
    test_ops.c                      — ops forward + backward correctness, BF16, N-D matmul
    test_train.c                    — backward pass, grad accumulation, optimizer step
    test_attention.c                — causal + encoder attention, batch parity
    test_gpt.c                      — GPT forward shape (batch=1 and batch=2), param collection
    test_tokenizer.c                — vocab build, encode/decode round-trip, tokenize
    test_checkpoint.c               — save/load round-trip, bad magic, count mismatch
    test_sample.c                   — argmax, top-k determinism, index bounds
    test_main.c                     — test runner entry point
  legacy/
    value.c / value.h               — scalar autograd (learning exercise, not compiled)
    mlp.c / mlp.h                   — scalar MLP (learning exercise, not compiled)
  data/
    text/
      candide.txt                   — corpus for GPT character-level demo
    checkpoints/
      model.bin                     — saved after each training run (gitignored)
```

---

## Tensor Struct

```c
typedef enum { TG_DTYPE_F32 = 0, TG_DTYPE_BF16 = 1 } TgDtype;

struct Tensor {
    int          ndim;
    int          shape[TG_MAX_DIMS];  /* TG_MAX_DIMS = 4 */
    TgDtype      dtype;               /* TG_DTYPE_F32 (default) or TG_DTYPE_BF16 (CUDA-only) */
    void        *data;                /* float* for F32; __nv_bfloat16* for BF16 */
    float       *grad;                /* always FP32 */
    float       *cache;               /* op-specific scratch (freed by tg_free) */
    float        aux;                 /* scalar op parameter (eps, axis, scale, etc.) */
    Tensor      *parents[TG_MAX_PARENTS];
    int          n_parents;
    TgBackwardFn backward_fn;
    int          persistent;
    int          visited;
    float       *cuda_data;           /* device data pointer (cast to bfloat16* when BF16) */
    float       *cuda_grad;           /* device grad pointer (always float*) */
    float       *cuda_cache;
    int          on_cuda;
};

/* TG_MAX_PARENTS = 8, TG_MAX_GRAPH = 8192, TG_MAX_DIMS = 4 */

/* Convenience macros */
#define TG_DATAF(t)   ((float *)(t)->data)   /* valid when dtype == TG_DTYPE_F32 */
static inline int tg_numel(const Tensor *t); /* product of shape[0..ndim-1] */
```

Factory: `tg_new(int ndim, const int shape[])`. All tensors are `TG_DTYPE_F32` and zeroed.
`ndim < 2` or `ndim > TG_MAX_DIMS` is a fatal error. Scalars are `[1, 1]`, bias vectors are `[1, C]`.

---

## Tensor Autograd Ops

All ops in `tg_ops.c / tg_ops.h`. Every op has a paired `_backward` function — keep them together.

### Shape / reshape

* `tg_reshape(a, ndim, shape)` — same numel, new shape; ndim ≥ 2; fatal if element count mismatches
* `tg_expand_dim(a, axis, n)` — shape[axis] must be 1; tiles n times; backward sums over axis
* `tg_slice(a, axis, start, len)` — contiguous slice; backward scatters gradient back
* `tg_transpose(a, dim0, dim1)` — swap any two axes; works for any ndim

### Arithmetic

* `tg_add(a, b)` — element-wise; shapes must match exactly (no silent broadcasting)
* `tg_sub(a, b)` — element-wise
* `tg_mul(a, b)` — element-wise
* `tg_pow(a, p)` — element-wise a^p
* `tg_scale(a, s)` — element-wise a×s
* `tg_matmul(a, b)` — N-D: `A[..., M, K] @ B[..., K, N] → [..., M, N]`
  - Same ndim: leading dims must match.
  - `ndim(b) < ndim(a)`: b broadcast over a's leading dims (e.g. `[B,T,C] @ [C,C_out]`). Only second operand may be lower-ndim.
  - CUDA: cuBLAS `SgemmStridedBatched` (F32), `cublasGemmEx` (BF16 with `CUBLAS_COMPUTE_32F`).

### Activations

* `tg_tanh(a)`, `tg_relu(a)`, `tg_gelu(a)` — element-wise

### Reductions

* `tg_sum(a)` — all elements → [1,1]
* `tg_mean(a)` — all elements → [1,1]
* `tg_mean_rows(a)` — [R×C] → [1×C], mean over rows (2D only)

### Attention / normalization

* `tg_causal_mask(scores)` — masks future positions across all leading dims; last two dims must be equal (square)
* `tg_softmax(a, axis)` — softmax along specified axis; CUDA dispatch for 2D last-axis, CPU general
* `tg_layer_norm(a, gamma, beta, eps)` — normalizes over `ndim-1` (feature axis); `gamma`/`beta` must be `[1, C]` (exactly C elements); pass them directly without pre-expansion

### Regularization

* `tg_dropout(a, p)` — inverted dropout; pass-through when `tg_training == 0` or `p == 0`

### Loss

* `tg_cross_entropy(logits, targets)` — mean CE [1×1]; targets one-hot
* `tg_cross_entropy_no_sync`, `tg_cross_entropy_sparse`, `tg_cross_entropy_sparse_no_sync`

### Embedding / precision

* `tg_embed(weight, ids, T)` — gather T rows from weight [V×C] by integer ids → [T×C]
* `tg_cast(a, dtype)` — F32↔BF16 precision conversion; BF16 is CUDA-only; `tg_cast(a, a->dtype)` is fatal; backward is FP32 passthrough

---

## Training API

```c
void  tg_backward(Tensor *root);                       // full backward + zero grads
void  tg_backward_accum(Tensor *root);                 // backward, accumulate grads (no zero)
void  tg_zero_grads(Tensor **params, int n);           // zero grad arrays for param list
void  tg_sgd_step(Tensor **params, int n, float lr);
void  tg_adam_step(Tensor **params, float **m, float **v, int n,
                   float lr, int t, float b1, float b2, float eps);
void  tg_adam_step_gpu(Tensor **params, float **m_gpu, float **v_gpu, int n,
                       float lr, int t, float b1, float b2, float eps);  // GPU, moment buffers on device
void  tg_free_graph(Tensor *root);
float tg_clip_grad_norm(Tensor **params, int n, float max_norm, float eps);
// CUDA path: clips on device, returns 0.0f
```

---

## Attention Module

File: `attention.c / attention.h`

Input shape: `[B, T, C]`. Output shape: `[B, T, C]`.

```c
typedef struct {
    Tensor *Wq, *Wk, *Wv, *Wo;   /* all [C, C] */
    int embed_dim, head_dim, n_heads, causal;
} TgSelfAttention;

TgSelfAttention tg_attention_create(int embed_dim, int n_heads);          // causal = 1
TgSelfAttention tg_attention_create_encoder(int embed_dim, int n_heads);  // causal = 0
void            tg_attention_free(TgSelfAttention *a);
Tensor         *tg_attention_forward(TgSelfAttention *a, Tensor *X);      // X: [B, T, C]
```

Forward sequence: `Q/K/V = X @ W[q/k/v]` → `reshape [B,T,H,D]` → `transpose(1,2) → [B,H,T,D]` → `Q @ K^T → [B,H,T,T]` → `scale → causal_mask → softmax(axis=3) → @ V → transpose(1,2) → reshape [B,T,C] → @ Wo`.

No per-head loop — removed in v2. No `n_heads > TG_MAX_PARENTS` ceiling.

---

## Transformer Block

File: `tg_block.c / tg_block.h`

Pre-norm architecture: LayerNorm → Attention → dropout/drop-path → residual → LayerNorm → FFN → dropout/drop-path → residual.

Accepts `[B, T, C]` input; 2D `[T, C]` is automatically reshaped to `[1, T, C]` internally.

```c
typedef struct {
    TgSelfAttention attn;
    Tensor *gamma1, *beta1;  /* LN affine scale/shift before attention  [1, C] */
    Tensor *gamma2, *beta2;  /* LN affine scale/shift before FFN        [1, C] */
    Tensor *W1, *B1;         /* FFN weights [C, ffn_dim] and biases [1, ffn_dim] */
    Tensor *W2, *B2;         /* FFN weights [ffn_dim, C] and biases [1, C] */
    int   embed_dim, hidden_dim;
    float dropout;
    float drop_path_rate;
} TgBlock;

TgBlock  tg_block_create(int embed_dim, int hidden_dim, int seq_len, int n_heads);         // causal
TgBlock  tg_block_create_encoder(int embed_dim, int hidden_dim, int seq_len, int n_heads); // non-causal
void     tg_block_free(TgBlock *b);
Tensor  *tg_block_forward(TgBlock *b, Tensor *X);
```

`seq_len` is accepted for API compatibility but no longer affects parameter shapes. `gamma`/`beta` are passed as `[1, C]` directly to `tg_layer_norm`. Bias tensors (`B1`, `B2`) are expanded to `[B, T, C/H]` at runtime via `tg_reshape + tg_expand_dim` for use with `tg_add` — no static pre-tiling.

---

## Transformer Stack

```c
TgTransformer tg_transformer_create(int n_blocks, int embed_dim, int hidden_dim,
                                    int seq_len, int n_heads);
TgTransformer tg_transformer_create_encoder(int n_blocks, int embed_dim, int hidden_dim,
                                            int seq_len, int n_heads,
                                            float max_drop_path_rate);
Tensor       *tg_transformer_forward(TgTransformer *t, Tensor *X);
```

`tg_transformer_create_encoder` applies a linear stochastic-depth schedule across blocks.

---

## GPT Model

```c
typedef struct {
    int vocab_size, embed_dim, hidden_dim, seq_len, n_blocks, n_heads;
} TgGPTConfig;

TgGPT   tg_gpt_create(int vocab_size, int embed_dim, int hidden_dim,
                      int seq_len, int n_blocks, int n_heads);
TgGPT   tg_gpt_create_from_config(const TgGPTConfig *cfg);

/* token_ids: flat [batch_size × seq_len] int array (row-major).
   Returns logits [batch_size * seq_len, vocab_size]. */
Tensor *tg_gpt_forward(TgGPT *g, const int *token_ids, int batch_size);

int     tg_gpt_collect_params(TgGPT *g, Tensor **params, int max_params);
// param count = 3 + n_blocks * 12
```

`PosEmb` is `[T, C]` and expanded to `[B, T, C]` each forward pass via `tg_reshape + tg_expand_dim`. Logits `[B, T, V]` are reshaped to `[B*T, V]` before returning — compatible with `tg_cross_entropy`.

---

## Tokenizer

File: `tg_tokenizer.c / tg_tokenizer.h`

Character-level byte vocabulary.

```c
typedef struct { char chars[256]; int ids[256]; int size; } TgVocab;

char    *tg_read_file(const char *path, int *out_len);             // malloc'd; caller frees
TgVocab  tg_vocab_build(const char *text, int len);
int      tg_vocab_encode(const TgVocab *v, char c);                // ovg_fatal on unknown char
char     tg_vocab_decode(const TgVocab *v, int id);                // ovg_fatal on bad id
int     *tg_tokenize(const char *text, int len, const TgVocab *v); // malloc'd; caller frees
```

---

## Sampling

```c
int  tg_sample_argmax(const Tensor *logits, int row);
int  tg_sample_topk(const Tensor *logits, int row, float temperature, int top_k);
void tg_generate(TgGPT *g, const TgVocab *v,
                 const int *context, int ctx_len,
                 int steps, float temperature, int top_k,
                 void (*on_token)(char c, void *ud), void *userdata);
```

`logits` is 2D `[T, vocab_size]`. `tg_generate` sets `tg_training=0` and passes `batch_size=1`.

---

## Checkpoints

Binary format v2 (little-endian):
- `uint32` magic = `0x00475632`
- `int32` n (param count)
- Per param: `int32 ndim`, `int32 shape[TG_MAX_DIMS]`, then `numel` floats

Validates magic, count, and per-tensor shapes on load. Incompatible with v1 checkpoints (magic `0x00475643`).

---

## CUDA Support

Enabled via `OVG_CUDA=ON`. When enabled:
- `OVG_CUDA_ENABLED` is defined globally (propagates to consumers via `PUBLIC`)
- All ops dispatch to GPU kernels when `t->on_cuda == 1`
- cuBLAS handles matmul: `SgemmStridedBatched` (N-D F32), `cublasSgemm` (2D F32), `cublasGemmEx` (BF16 with `CUBLAS_COMPUTE_32F`)
- `tg_to_cuda(t)` uploads F32 tensor to device; `tg_from_cuda(t)` syncs back
- `tg_cuda_alloc(t)` allocates device data+grad; element size is dtype-aware (2 bytes for BF16, 4 for F32)
- Every op's forward and backward has a correct CPU path. BF16 tensors are CUDA-only (assert on CPU).

---

## Build Commands

```powershell
cmake --preset default             # configure: VS2026, CUDA, Release
cmake --build --preset default     # build
.\build\otto_von_grad.exe          # GPT demo (trains on candide.txt)
.\build\otto_von_grad_tests.exe    # test suite — 59 tests, exits 0 on all-pass
```

Non-default presets:

```powershell
cmake --preset debug   && cmake --build --preset debug   # Debug + CUDA
cmake --preset cpu     && cmake --build --preset cpu     # Release, no CUDA
```

---

## RNG and Seeding

```c
void     tg_seed(uint32_t seed);       // seed rand() + xorshift32, log seed to stdout
void     tg_seed_from_entropy(void);   // seed from OS entropy, then call tg_seed()
uint32_t tg_rng_xorshift32(void);      // raw xorshift32 (used internally by tg_dropout)
float    tg_rng_uniform(void);         // uniform float in [0, 1)
```

---

## Error Handling

```c
#include "ovg_error.h"
ovg_set_fatal_handler(my_handler);  // install pre-exit hook (not thread-safe)
```

`ovg_fatal()` always calls `exit(1)` after the hook returns. Tests use `setjmp`/`longjmp` to capture fatal calls in-process.

---

## Include Style

```c
#include "tg_ops.h"    // ops + Tensor struct (via tg_tensor.h)
#include "tg_train.h"  // tg_backward, optimizers (+ Tensor struct)
```

---

## Coding Style

* C11. No OOP patterns, no macro-heavy metaprogramming.
* Tensors are N-D up to `TG_MAX_DIMS = 4`. Shape convention throughout: `[B, T, C]` for sequences, `[B, H, T, D]` for attention scores.
* No silent broadcasting — shape mismatches must fail loudly via `ovg_fatal`.
* Every op in `tg_ops.c` has a paired `_backward` function — keep them together.
* `persistent = 1` marks parameters. `tg_free_graph` handles intermediates.
* When adding CUDA kernels, preserve the symmetric CPU fallback path.
* Correctness and readability over performance.

---

## Known Constraints

* **No silent broadcasting.** Use `tg_expand_dim` (and `tg_reshape` to cross ndim boundaries) to make shapes explicit.
* **BF16 is CUDA-only.** `tg_cast` to BF16 asserts on CPU. Non-matmul ops operate on F32; cast back before layernorm, gelu, etc.
* **Static graph per step.** No dynamic computation graphs; `tg_free_graph` after each step.
* **`tg_cross_entropy` is 2D.** Reshape `[B, T, V]` logits to `[B*T, V]` before CE (`tg_gpt_forward` does this automatically).
* **`tg_mean_rows` is 2D-only.** Takes `[R, C]`, returns `[1, C]`.
* **`topo_sort` is recursive.** Safe at current model depths (~200). If `TG_MAX_GRAPH` needs raising significantly, convert to iterative.
* **Eager dispatch.** One kernel launch per op; no graph compilation or kernel fusion.

---

## Important Guidance For Agents

When modifying code:

* Preserve explicit tensor math — do not hide operations behind abstractions.
* Preserve the backward function paired with each op in `tg_ops.c`.
* Do not introduce external ML libraries.
* Do not add silent broadcasting — it changes semantics for all callers.
* When adding a new op, add both the forward function and its `_backward` counterpart.
* When adding a CUDA kernel, ensure the CPU path remains correct and the dispatch logic is symmetric.
* `tg_block_forward` reshapes 2D `[T, C]` input to `[1, T, C]` at the start — the whole block operates in 3D. Attention output is `[B, T, C]` (3D), not `[B*T, C]`.
* `tg_gpt_forward` returns `[B*T, V]` (2D), ready for cross-entropy.
* The checkpoint magic is `0x00475632` (v2). The v1 magic `0x00475643` is rejected on load.
