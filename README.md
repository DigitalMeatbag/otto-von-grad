# otto_von_grad

Reverse-mode autograd engine and neural-network infrastructure, written from scratch in C.

N-D tensors (up to 4D), batched multi-head attention, optional CUDA acceleration with cuBLAS, and opt-in BF16 mixed precision.

---

## Tensor Autograd API

### Tensor lifecycle (`tg_tensor.h`)

```c
/* Factory: ndim in [2, TG_MAX_DIMS=4]; shape[] must be positive.
   All tensors are allocated TG_DTYPE_F32 and zeroed. */
Tensor *tg_new(int ndim, const int shape[]);
void    tg_free(Tensor *t);

/* Fill helpers */
void    tg_fill(Tensor *t, float val);
void    tg_fill_uniform(Tensor *t, float lo, float hi);
void    tg_fill_randn(Tensor *t, float scale);
void    tg_fill_xavier_uniform(Tensor *t);
void    tg_fill_xavier_normal(Tensor *t);

/* Scalar read (syncs from GPU if on_cuda) */
float   tg_scalar_value(Tensor *t);
void    tg_print(const Tensor *t, const char *name);

/* Total element count: product of shape[0..ndim-1] */
static inline int tg_numel(const Tensor *t);

/* Cast data pointer to float* — valid when dtype == TG_DTYPE_F32 */
#define TG_DATAF(t)  ((float *)(t)->data)
```

`Tensor.persistent = 1` marks a tensor as exempt from `tg_free_graph` (use for parameters
and pre-allocated inputs you want to free manually).

**Tensor struct (key fields):**

```c
struct Tensor {
    int      ndim;
    int      shape[TG_MAX_DIMS];   /* TG_MAX_DIMS = 4 */
    TgDtype  dtype;                /* TG_DTYPE_F32 (default) or TG_DTYPE_BF16 (CUDA-only) */
    void    *data;                 /* float* for F32; __nv_bfloat16* for BF16 */
    float   *grad;                 /* always FP32 */
    int      persistent;
    int      on_cuda;
    /* ... backward linkage, parents, etc. */
};
```

**Allocation examples:**

```c
tg_new(2, (int[]){T, C})         /* [T × C] */
tg_new(3, (int[]){B, T, C})      /* [B × T × C] */
tg_new(4, (int[]){B, H, T, D})   /* [B × H × T × D] */
```

### Ops (`tg_ops.h`)

All ops return a new `Tensor*` that participates in the computation graph. No silent broadcasting — shape mismatches are fatal errors.

#### Arithmetic

| Op | Shape | Notes |
|---|---|---|
| `tg_add(a, b)` | same → same | element-wise; shapes must match exactly |
| `tg_sub(a, b)` | same → same | |
| `tg_mul(a, b)` | same → same | |
| `tg_pow(a, p)` | same → same | element-wise a^p |
| `tg_scale(a, s)` | same → same | element-wise a×s |
| `tg_matmul(a, b)` | `[...,M,K] @ [...,K,N] → [...,M,N]` | N-D; second operand may be 2D for weight broadcast |

**`tg_matmul` shape rules:**
- Same ndim: leading dims must match element-wise.
- `ndim(b) < ndim(a)`: b is broadcast over a's leading dims. Typical weight projection: `[B,T,C] @ [C,C_out] → [B,T,C_out]`.
- First-operand-lower is fatal.

#### Reshape / expand

| Op | Shape | Notes |
|---|---|---|
| `tg_reshape(a, ndim, shape)` | same numel → new shape | ndim ≥ 2; element count must match |
| `tg_expand_dim(a, axis, n)` | shape[axis] must be 1 → n | tiles along axis; backward sums over axis |
| `tg_slice(a, axis, start, len)` | shape[axis] reduced | backward scatters gradient back |
| `tg_transpose(a, dim0, dim1)` | swap two axes | works for any ndim |

#### Activations

| Op | Notes |
|---|---|
| `tg_tanh(a)` | element-wise |
| `tg_relu(a)` | element-wise |
| `tg_gelu(a)` | tanh-approx GELU |

#### Reductions

| Op | Shape | Notes |
|---|---|---|
| `tg_sum(a)` | any → [1,1] | all elements |
| `tg_mean(a)` | any → [1,1] | all elements |
| `tg_mean_rows(a)` | [R,C] → [1,C] | column-wise mean |

#### Normalization / attention

| Op | Shape | Notes |
|---|---|---|
| `tg_softmax(a, axis)` | same → same | softmax along specified axis (CUDA: last axis only) |
| `tg_causal_mask(scores)` | `[...,T,T]` → same | -1e9 on future positions; last two dims must be equal |
| `tg_layer_norm(a, gamma, beta, eps)` | same → same | normalizes over `ndim-1`; gamma/beta must have exactly C elements (C = a's last dim), typically `[1,C]`; do not pre-expand to a's full shape |

#### Loss

| Op | Shape | Notes |
|---|---|---|
| `tg_cross_entropy(logits, targets)` | [T,V],[T,V] → [1,1] | mean CE, targets one-hot |
| `tg_cross_entropy_no_sync(...)` | same | CUDA path skips scalar host sync |
| `tg_cross_entropy_sparse(logits, ids, n, smooth)` | [T,V], int[T] → [1,1] | integer labels + optional label smoothing |
| `tg_cross_entropy_sparse_no_sync(...)` | same | sparse CE, no CUDA sync |

#### Other

| Op | Shape | Notes |
|---|---|---|
| `tg_embed(weight, ids, T)` | [V,C], int[T] → [T,C] | gather rows by token id |
| `tg_dropout(a, p)` | same → same | inverted dropout; no-op when `tg_training==0` or `p==0` |
| `tg_cast(a, dtype)` | same → same | F32↔BF16 precision conversion; BF16 requires `on_cuda==1` |

### Training (`tg_train.h`)

```c
extern int tg_training;  // 1 = training (default), 0 = inference — disables dropout

void  tg_backward(Tensor *root);
void  tg_backward_accum(Tensor *root);
void  tg_zero_grads(Tensor **params, int n);
void  tg_sgd_step(Tensor **params, int n, float lr);
void  tg_adam_step(Tensor **params, float **m, float **v, int n,
                   float lr, int step, float beta1, float beta2, float eps);
float tg_clip_grad_norm(Tensor **params, int n, float max_norm, float eps);
// CUDA path: clips on device, returns 0.0f (norm not read back to host)
void  tg_free_graph(Tensor *root);
```

`tg_free_graph` walks the graph in topological order and frees every tensor not marked `persistent`. Call it on `loss` at the end of each step.

---

## Mixed Precision (BF16)

BF16 is an opt-in CUDA-only path. FP32 is the default.

```c
/* Typical mixed-precision pattern */
Tensor *W     = tg_new(2, (int[]){C, C});
tg_to_cuda(W);
Tensor *W_bf16 = tg_cast(W, TG_DTYPE_BF16);   /* F32 → BF16 */

Tensor *X_bf16 = tg_cast(X_cuda, TG_DTYPE_BF16);
Tensor *Y      = tg_matmul(X_bf16, W_bf16);   /* BF16×BF16 → F32 output */
                                               /* internally CUBLAS_COMPUTE_32F */
```

- `grad` is always `float*` (FP32) regardless of `dtype` — the optimizer reads FP32 gradients.
- `tg_cast(a, a->dtype)` is a fatal error.
- All non-matmul ops operate on FP32 tensors; cast to FP32 before passing to layernorm, gelu, etc.
- Optimizer moment buffers (Adam m/v) must remain FP32.

---

## Architecture Modules

### TgSelfAttention (`attention.h`)

Multi-head self-attention. Input shape: `[B, T, C]`. Output shape: `[B, T, C]`.
Internally uses `reshape → transpose → batched matmul` — no per-head loop.

```c
TgSelfAttention tg_attention_create(int embed_dim, int n_heads);          // causal (GPT)
TgSelfAttention tg_attention_create_encoder(int embed_dim, int n_heads);  // non-causal (ViT)
Tensor         *tg_attention_forward(TgSelfAttention *a, Tensor *X);
```

Weights `Wq/Wk/Wv/Wo` are `[C, C]`. Batched attention scores are `[B, H, T, T]` with cuBLAS `SgemmStridedBatched`.

### TgBlock (`tg_block.h`)

Pre-norm transformer block. Accepts `[B, T, C]` input (2D `[T, C]` is auto-reshaped to `[1, T, C]`).

```c
TgBlock  tg_block_create(int embed_dim, int hidden_dim, int seq_len, int n_heads);
TgBlock  tg_block_create_encoder(int embed_dim, int hidden_dim, int seq_len, int n_heads);
Tensor  *tg_block_forward(TgBlock *b, Tensor *X);
```

`TgBlock.dropout` (float, default 0) — set after creation to enable dropout.
`TgBlock.drop_path_rate` (float, default 0) — stochastic depth; applied to residual branches during training. Set automatically by `tg_transformer_create_encoder`.

LayerNorm `gamma`/`beta` are `[1, C]` and are passed directly to `tg_layer_norm` — no expansion. FFN biases `B1`/`B2` are `[1, ffn_dim]`/`[1, C]` and expanded to `[B, T, ffn_dim]`/`[B, T, C]` via `tg_reshape` + `tg_expand_dim` for `tg_add`. `seq_len` is accepted but ignored (bias shapes are no longer seq-len-specific).

### TgTransformer (`tg_transformer.h`)

Stack of `TgBlock`s.

```c
TgTransformer tg_transformer_create(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads);
TgTransformer tg_transformer_create_encoder(int n_blocks, int embed_dim, int hidden_dim,
                                            int seq_len, int n_heads, float max_drop_path_rate);
Tensor       *tg_transformer_forward(TgTransformer *t, Tensor *X);
```

### TgGPT (`tg_gpt.h`)

Token + positional embeddings → transformer → output projection.

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

### TgVocab / tokenizer (`tg_tokenizer.h`)

Character-level vocabulary over raw bytes.

```c
typedef struct { char chars[256]; int ids[256]; int size; } TgVocab;

char    *tg_read_file(const char *path, int *out_len);
TgVocab  tg_vocab_build(const char *text, int len);
int      tg_vocab_encode(const TgVocab *v, char c);
char     tg_vocab_decode(const TgVocab *v, int id);
int     *tg_tokenize(const char *text, int len, const TgVocab *v);
```

### Sampling (`tg_sample.h`)

```c
int  tg_sample_argmax(const Tensor *logits, int row);
int  tg_sample_topk(const Tensor *logits, int row, float temperature, int top_k);
void tg_generate(TgGPT *g, const TgVocab *v,
                 const int *context, int ctx_len,
                 int steps, float temperature, int top_k,
                 void (*on_token)(char c, void *ud), void *userdata);
```

`logits` is expected to be 2D `[T, vocab_size]` (the reshaped output of `tg_gpt_forward`).
`tg_generate` manages the sliding context window and passes `batch_size=1` internally.

### Checkpoints (`tg_checkpoint.h`)

```c
int tg_checkpoint_save(const char *path, Tensor **params, int n);  // 0 on success, -1 on error
int tg_checkpoint_load(const char *path, Tensor **params, int n);  // silent -1 if file missing
```

v2 format: magic `0x00475632`, then per-tensor `[int32 ndim, int32 shape[4], float data]`.
Validates magic, param count, and per-tensor shapes on load. Incompatible with v1 checkpoints.

---

## CUDA Support

Enabled via `-DOVG_CUDA=ON` (the `default` preset). When enabled:

- `OVG_CUDA_ENABLED` is defined globally.
- All ops dispatch to GPU kernels when `t->on_cuda == 1`.
- cuBLAS handles all matmul: `cublasSgemmStridedBatched` for N-D, `cublasSgemm` for 2D, `cublasGemmEx` for BF16.
- `tg_to_cuda(t)` uploads F32 tensor to device; `tg_from_cuda(t)` syncs back.

```c
void tg_to_cuda(Tensor *t);
void tg_from_cuda(Tensor *t);
void tg_cuda_free(Tensor *t);
```

The CPU path is always correct and symmetric. Every op has a working CPU fallback.

---

## Build

Default preset: VS2026, CUDA enabled, Release mode. Outputs land directly in `build\`.

```powershell
# First configure (fresh clone or after deleting build/)
cmake --preset default

# Every subsequent build
cmake --build --preset default
.\build\otto_von_grad.exe
```

Non-default presets:

```powershell
cmake --preset debug  && cmake --build --preset debug   # Debug + CUDA
cmake --preset cpu    && cmake --build --preset cpu     # Release, CPU-only
```

CMake produces three targets:
- `ottovongrad` — static library (links `CUDA::cudart CUDA::cublas` when CUDA enabled)
- `otto_von_grad` — GPT character-level demo (candide.txt); reports train/val loss, generates text, saves checkpoint
- `otto_von_grad_tests` — test suite

---

## Testing

```powershell
cmake --build --preset default
.\build\otto_von_grad_tests.exe   # 59 tests; exits 0 on all-pass

# Or via CTest
ctest -C Release --test-dir build
```

---

## Error Handling

```c
#include "ovg_error.h"

void my_handler(const char *msg) { /* logging, capture, etc. */ }
ovg_set_fatal_handler(my_handler);  // call before spawning threads
```

`ovg_fatal()` calls `exit(1)` after the hook returns. Tests use `setjmp`/`longjmp` to capture fatal calls in-process; see `tests/test_ops.c` for the pattern.

---

## RNG (`tg_rng.h`)

```c
void     tg_seed(uint32_t seed);
void     tg_seed_from_entropy(void);
uint32_t tg_rng_xorshift32(void);
float    tg_rng_uniform(void);       // [0, 1)
```

The chosen seed is printed at startup so any run can be reproduced:

```
[ovg] rng seed: 0x5f3759df
```

---

## Usage as a Dependency

```cmake
# Side-by-side checkout
if(NOT TARGET ottovongrad)
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../otto-von-grad"
                     "${CMAKE_CURRENT_BINARY_DIR}/otto-von-grad")
endif()
target_link_libraries(your_target PRIVATE ottovongrad)
```

Or via FetchContent:

```cmake
include(FetchContent)
FetchContent_Declare(ottovongrad
    GIT_REPOSITORY https://github.com/DigitalMeatbag/otto-von-grad.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(ottovongrad)
target_link_libraries(your_target PRIVATE ottovongrad)
```
