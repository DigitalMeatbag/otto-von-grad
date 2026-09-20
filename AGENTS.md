# AGENTS.md

# otto-von-grad

A lightweight tensor autograd engine and neural-network toolkit written in C (C11), built as three layers: `ovg_core` (reverse-mode autodiff over N-D tensors up to 4D, optimizers, checkpoints, optional CUDA/cuBLAS), `ovg_nn` (linear, batched multi-head attention, transformer blocks), and `ovg_lm` (GPT-style language model, byte tokenizer, sampling). No external ML libraries. This file is the single source of truth for any coding agent working in this repo (`CLAUDE.md` imports it verbatim).

Optional CUDA acceleration via `OVG_CUDA=ON`.

This repo is a **library**, built as three layered CMake targets (`ovg_core` → `ovg_nn` → `ovg_lm`, see [Library Layers](#library-layers)). It is consumed by sibling repos checked out next to it (`../vexilloscope`, a ViT classifier; `../lambda`, a lambda-calculus GPT curriculum) via `add_subdirectory(../otto-von-grad)` and `target_link_libraries(... ottovongrad)`. Application code, experiments, and corpora belong in those repos, not here. The only application here is `examples/candide.c`, the candide.txt GPT demo that doubles as the end-to-end smoke test.

---

## Repository Structure

```text
otto-von-grad/
  include/ovg/                      — public headers (PUBLIC include dir; consumers #include these by bare name)
    [core]
    tg_tensor.h                     — Tensor struct, lifecycle, print helpers, tg_numel, TG_DATAF
    tg_ops.h                        — all differentiable ops
    tg_train.h                      — tg_backward, tg_sgd_step, tg_adam_step, grad clipping, eval guard (tg_eval_begin/end, TgMeter)
    tg_optim.h                      — TgAdam: optimizer state + zero_grads / accumulate / update
    tg_sched.h                      — tg_lr_warmup_cosine / tg_lr_warmup_linear (pure functions of step)
    tg_checkpoint.h                 — weights-only save/load, tg_checkpoint_info / save_run / load_run (binary format v3; reads v2)
    tg_cuda.h                       — CUDA tensor lifecycle (tg_to_cuda, tg_from_cuda)
    tg_rng.h                        — RNG state and seeding (tg_seed, tg_seed_from_entropy, tg_rng_get_state / set_state)
    ovg_error.h                     — centralized fatal error handler (ovg_fatal, ovg_set_fatal_handler)
    [nn]
    tg_mlp.h                        — TgLinear convenience layer
    tg_attention.h                  — TgSelfAttention
    tg_block.h                      — TgBlock (pre-norm transformer block)
    tg_transformer.h                — TgTransformer (stack of blocks)
    [lm]
    tg_gpt.h                        — TgGPT, TgGPTConfig
    tg_tokenizer.h                  — TgVocab, tg_read_file, tg_vocab_build/encode/decode, tg_tokenize
    tg_sample.h                     — tg_sample_argmax, tg_sample_topk, tg_generate
  src/                              — implementation (PRIVATE include dir)
    [core → ovg_core]
    tg_tensor.c                     — Tensor lifecycle, fill helpers, print
    tg_ops.c                        — every op's forward + paired _backward function
    tg_train.c                      — topo sort, backward, optimizers, eval guard
    tg_optim.c                      — TgAdam over the raw tg_adam_step / tg_adam_step_gpu
    tg_sched.c                      — learning-rate schedules
    tg_checkpoint.c                 — binary checkpoint I/O (v3 writer; v2 + v3 readers)
    tg_rng.c                        — xorshift32 + seeding
    ovg_error.c                     — ovg_fatal + handler hook
    tg_cuda.cu                      — CUDA tensor upload/sync/alloc
    tg_cuda_internal.h              — device plumbing for ops/train/optim/checkpoint (tg_cuda_alloc, cache buffers, grad zeroing, raw float buffers); private
    cuda_ops.cu / cuda_ops.h        — CUDA kernels for all ops + cuBLAS dispatch (internal only, not exported)
    [nn → ovg_nn]
    tg_mlp.c                        — TgLinear
    tg_attention.c                  — batched multi-head attention forward
    tg_block.c                      — TgBlock forward
    tg_transformer.c                — TgTransformer forward + stochastic-depth schedule
    [lm → ovg_lm]
    tg_gpt.c                        — embeddings + transformer + output projection
    tg_tokenizer.c                  — character-level byte vocabulary
    tg_sample.c                     — sampling + generation loop
  examples/
    candide.c                       — GPT training demo; links ovg_lm; saves/resumes checkpoints; builds as `candide`
    data/candide.txt                — corpus for the demo
    data/checkpoints/model.bin      — saved after each demo run (gitignored)
  docs/
    FOUNDATION_PLATFORM.md          — platform direction: target layering, open decisions, phased plan
  tests/
    ovg_test.h                      — minimal test assertion macros
    test_ops.c                      — ops forward + backward correctness, BF16, N-D matmul
    test_train.c                    — backward pass, grad accumulation, optimizer step
    test_optim.c                    — TgAdam parity/reset/accumulate/step counter, schedules, eval guard, RNG state
    test_attention.c                — causal + encoder attention, batch parity
    test_gpt.c                      — GPT forward shape (batch=1 and batch=2), param collection
    test_tokenizer.c                — vocab build, encode/decode round-trip, tokenize
    test_checkpoint.c               — weights-only round-trip, bad magic, count mismatch; v3 run-state round-trip, info, v2 compat, load modes, exact resume, .tmp replacement
    test_sample.c                   — argmax, top-k determinism, index bounds
    test_main.c                     — test runner entry point
```

---

## Library Layers

The library is three static targets with dependencies pointing strictly downward. Nothing in a lower
layer may include a header from a higher one.

| Target | Contents | Links |
|---|---|---|
| `ovg_core` | tensor, autograd ops, backward/optimizers, `TgAdam`, LR schedules, eval guard, RNG, checkpoint I/O, error handler, CUDA kernels | (CUDA runtime/cuBLAS when `OVG_CUDA=ON`) |
| `ovg_nn`   | model-agnostic building blocks: `TgLinear`, `TgSelfAttention`, `TgBlock`, `TgTransformer` | `ovg_core` |
| `ovg_lm`   | language-model toolkit: `TgGPT`, `TgVocab`/tokenizer, sampling/generation | `ovg_nn` |
| `ottovongrad` | INTERFACE umbrella = everything above | `ovg_lm` |

Consumers link the narrowest target that has what they need: a vision model links `ovg_nn`, a GPT
links `ovg_lm`, `ottovongrad` is the everything-included default. The intent is for `ovg_lm` to be
packaged on its own eventually; keep it free of anything a non-LM consumer would need, and keep
`ovg_core`/`ovg_nn` free of anything that assumes tokens or text.

The target layering, the open design questions, and the phased plan for getting there are in
`docs/FOUNDATION_PLATFORM.md`. This file describes what exists; that one describes where it is going.

Placing new code:

* Operates on `Tensor` with no notion of a model → `ovg_core` (`tg_ops.c` for a differentiable op).
* A reusable layer or block that any architecture could use → `ovg_nn`.
* Assumes a vocabulary, token ids, sequences of text, or a GPT → `ovg_lm`.
* If a lower layer needs something from a higher one, the thing is in the wrong layer — move it
  down rather than adding the include.

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

All ops in `src/tg_ops.c` / `include/ovg/tg_ops.h`. Every op has a paired `_backward` function — keep them together.

### Shape / reshape

* `tg_reshape(a, ndim, shape)` — same numel, new shape; ndim ≥ 2; fatal if element count mismatches
* `tg_expand_dim(a, axis, n)` — shape[axis] must be 1; tiles n times; backward sums over axis
* `tg_slice(a, axis, start, len)` — contiguous slice; backward scatters gradient back
* `tg_concat(a, b, axis)` — join two tensors along `axis`; all other dims must match exactly; F32 only; backward slices the gradient to each parent (inverse of `tg_slice`)
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
                       float lr, int t, float b1, float b2, float eps);  // GPU, moment buffers on device; only declared when OVG_CUDA_ENABLED
void  tg_free_graph(Tensor *root);
float tg_clip_grad_norm(Tensor **params, int n, float max_norm, float eps);
// CUDA path: clips on device, returns 0.0f
```

### TgAdam (`tg_optim.h`)

The struct path wraps the raw `tg_adam_step` / `tg_adam_step_gpu` above (which remain public) and
is numerically identical to them. Moments follow the params: host buffers, or device buffers when
the params are on CUDA. **Create `TgAdam` after moving params to their device.**

```c
typedef struct {
    Tensor **params;     /* borrowed */
    int      n_params;
    float  **m, **v;     /* per-param moments; host or device buffers */
    int      on_cuda;    /* fixed at create */
    int      step;       /* completed updates; 0 after create/reset */
    float    beta1, beta2, eps;
} TgAdam;

TgAdam tg_adam_create(Tensor **params, int n_params, float beta1, float beta2, float eps);
       // fatal if n_params <= 0 or params mix CPU and CUDA
void   tg_adam_free(TgAdam *opt);       // frees m/v; does not touch params
void   tg_adam_reset(TgAdam *opt);      // zeroes m/v, step = 0

void   tg_adam_zero_grads(TgAdam *opt);                          // tg_zero_grads over params
void   tg_adam_accumulate(TgAdam *opt, Tensor *loss, int n_micro); // scale by 1/n_micro if > 1, backward_accum, free_graph
float  tg_adam_update(TgAdam *opt, float lr, float max_grad_norm);
       // step += 1; clips first if max_grad_norm > 0; returns the pre-clip norm on CPU, 0.0f on CUDA / no clip
```

A step is `tg_adam_zero_grads` → one `tg_adam_accumulate` per micro-step → `tg_adam_update`.
`tg_adam_accumulate` **frees the loss graph**: read the loss value (`tg_scalar_value`) before
calling it, and do not call `tg_free_graph(loss)` afterwards (that is a use-after-free).

### Eval guard (`tg_train.h`)

```c
int  tg_eval_begin(void);           // tg_training = 0; returns the previous value
void tg_eval_end(int prev_training);
typedef struct { double sum; int n; } TgMeter;   // running mean: tg_meter_add / tg_meter_mean
```

### Learning-rate schedules (`tg_sched.h`)

Pure functions of the 1-based step; `warmup_steps` may be 0, `total_steps > 0`. `warm` scales the
whole value (floor included); `progress = clamp((step - 1) / total_steps, 0, 1)`.

```c
float tg_lr_warmup_cosine(int step, int total_steps, int warmup_steps, float base, float floor_lr);
      // (floor + (base - floor) * 0.5 * (1 + cos(pi * progress))) * warm
float tg_lr_warmup_linear(int step, int total_steps, int warmup_steps, float base, float floor_lr);
      // (base + (floor - base) * progress) * warm
```

The harness does not own the schedule: the caller computes `lr` and passes it to `tg_adam_update`.
A constant LR needs no function.

---

## Attention Module

File: `src/tg_attention.c` / `include/ovg/tg_attention.h`

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

File: `src/tg_block.c` / `include/ovg/tg_block.h`

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

File: `src/tg_tokenizer.c` / `include/ovg/tg_tokenizer.h`

Character-level byte vocabulary.

```c
typedef struct { char chars[256]; int ids[256]; int size; } TgVocab;

char    *tg_read_file(const char *path, int *out_len);             // malloc'd; caller frees
TgVocab  tg_vocab_build(const char *text, int len);
TgVocab  tg_vocab_from_chars(const char *chars);                   // ids in ascending ASCII order (matches tg_vocab_build); pins vocab across phases
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

Binary format v3 (little-endian, fixed-width). Config precedes the params so a model can be built
from the file before any tensor exists; the optimizer section follows the params so the
weights-only reader can stop early.

```text
uint32   magic        = 0x00475633   (TG_CHECKPOINT_MAGIC_V3)
uint32   flags        bit 0: optimizer section present; all other bits must be 0
int32    step         completed updates at save (0 for weights-only)
uint32   rng_state    xorshift32 word at save when flags bit 0 is set; 0 otherwise
int32    config_len   >= 0
uint8    config[config_len]
int32    n_params
per param: int32 ndim, int32 shape[TG_MAX_DIMS], float data[numel]
if flags & 1:
  int32  optimizer_kind = 1 (Adam); float beta1, beta2, eps
  per param: float m[numel], float v[numel]
```

```c
int tg_checkpoint_save(const char *path, Tensor **params, int n);   // weights-only: v3, flags = 0
int tg_checkpoint_load(const char *path, Tensor **params, int n);   // reads v2 or v3; params only

int tg_checkpoint_info(const char *path, TgCheckpointInfo *info, void *config_out, int config_cap);
    // header only; copies min(config_len, config_cap) config bytes; -1 on error / missing file
int tg_checkpoint_save_run(const char *path, const TgAdam *opt, const void *config, int config_len);
    // params + m/v/step/betas/eps + current RNG state + config; device buffers synced by the call
int tg_checkpoint_load_run(const char *path, TgAdam *opt, TgLoadMode mode, TgCheckpointInfo *info);
    // TG_LOAD_RESUME (default): restores params, moments, step, RNG state
    // TG_LOAD_INIT_FROM_WEIGHTS: params only; tg_adam_reset(opt); RNG untouched
```

Rules:

- Both writers write `<path>.tmp` then replace `<path>`, so a kill mid-save keeps the previous file.
- All readers validate magic ∈ {v2, v3}, unknown flag bits, `config_len`, and short reads; the
  loaders also validate `n_params` and per-param `ndim`/shape. `load_run` on RESUME additionally
  rejects a wrong `optimizer_kind`, betas/eps that differ from the target `TgAdam` (exact float
  compare — change hyperparameters via `INIT_FROM_WEIGHTS`), and `rng_state == 0`.
- **v2 compatibility** (magic `0x00475632`: `uint32 magic, int32 n, params`): loads as weights-only.
  `load_run(RESUME)` on a v2 file, or on a v3 file without an optimizer section, loads the params,
  resets the optimizer to step 0, leaves the RNG untouched, and prints
  `[ovg] checkpoint <path>: no optimizer state; moments zeroed, step reset to 0`.
- v1 (magic `0x00475643`) is rejected.
- `load_run` failure is not transactional: params `0..k-1` may be overwritten. Treat -1 as fatal.
- `opt->params` must be the same tensors in the same order as at save; the file validates shapes,
  not identity.

---

## CUDA Support

Enabled via `OVG_CUDA=ON`. When enabled:
- `OVG_CUDA_ENABLED` is defined globally (propagates to consumers via `PUBLIC`)
- All ops dispatch to GPU kernels when `t->on_cuda == 1`
- cuBLAS handles matmul: `SgemmStridedBatched` (N-D F32), `cublasSgemm` (2D F32), `cublasGemmEx` (BF16 with `CUBLAS_COMPUTE_32F`)
- `tg_to_cuda(t)` uploads F32 tensor to device; `tg_from_cuda(t)` syncs back
- Public surface (`tg_cuda.h`): `tg_to_cuda`, `tg_from_cuda`, `tg_cuda_free`, `tg_cuda_malloc_floats` / `tg_cuda_free_floats` (device buffers for `tg_adam_step_gpu` moments)
- Internal (`src/tg_cuda_internal.h`, used by `tg_ops.c` / `tg_train.c` / `tg_optim.c` / `tg_checkpoint.c` only): `tg_cuda_alloc(t)` allocates device data+grad with dtype-aware element size (2 bytes for BF16, 4 for F32); `tg_cuda_alloc_cache` / `tg_cuda_upload_cache` for op scratch; `tg_cuda_zero_grad`, `tg_cuda_set_grad_scalar` for backward; `tg_cuda_upload_floats` / `tg_cuda_download_floats` / `tg_cuda_zero_floats` for raw device float buffers (Adam moments)
- Every op's forward and backward has a correct CPU path. BF16 tensors are CUDA-only (assert on CPU).

---

## Build Commands

Default preset: VS2026, CUDA enabled, Release mode, all outputs flattened into `build\`.

```powershell
cmake --preset default             # configure (fresh clone, after deleting build/, or after CMakeLists changes)
cmake --build --preset default     # every subsequent build
.\build\candide.exe                # GPT demo (trains on examples/data/candide.txt); does not run tests
.\build\otto_von_grad_tests.exe    # test suite — 101 tests (89 in the cpu preset), exits 0 on all-pass
```

Non-default presets:

```powershell
cmake --preset debug   && cmake --build --preset debug   # Debug + CUDA
cmake --preset cpu     && cmake --build --preset cpu     # Release, no CUDA
```

## Verification

After any code change, build and run the test binary before declaring the work done:

```powershell
cmake --build --preset default
.\build\otto_von_grad_tests.exe    # expect "101 passed, 0 failed" (cpu preset: "89 passed, 0 failed")
```

Docs-only changes are exempt. After a CMake change, also confirm each layer still builds on its own (`cmake --build --preset default --target ovg_core`, then `ovg_nn`, then `ovg_lm`) and that `../lambda` configures and builds with no edits to its own CMakeLists. For CUDA-specific changes, the default (CUDA) preset is the one that matters — the CPU preset will not exercise the kernels. Tests guarded by `#ifdef OVG_CUDA_ENABLED` are skipped in CPU-only builds.

---

## RNG and Seeding

```c
void     tg_seed(uint32_t seed);       // seed rand() + xorshift32, log seed to stdout
void     tg_seed_from_entropy(void);   // seed from OS entropy, then call tg_seed()
uint32_t tg_rng_xorshift32(void);      // raw xorshift32 (used internally by tg_dropout)
float    tg_rng_uniform(void);         // uniform float in [0, 1)
uint32_t tg_rng_get_state(void);       // the xorshift32 word (saved by tg_checkpoint_save_run)
void     tg_rng_set_state(uint32_t s); // fatal if s == 0
```

**Ordering contract.** `tg_seed()` / `tg_seed_from_entropy()` overwrite the xorshift word, so a
resume (`tg_checkpoint_load_run`) must run **after** them or the seed clobbers the restored state.
Only library randomness (dropout, drop-path, sampling) is restored; application-side `rand()` is not.

---

## Error Handling

```c
#include "ovg_error.h"
ovg_set_fatal_handler(my_handler);  // install pre-exit hook (not thread-safe)
```

`ovg_fatal()` always calls `exit(1)` after the hook returns. Tests use `setjmp`/`longjmp` to capture fatal calls in-process.

---

## Include Style

Public headers live in `include/ovg/`, which is the library's `PUBLIC` include directory, so both
in-tree code and consumers include them by bare name:

```c
#include "tg_ops.h"    // ops + Tensor struct (via tg_tensor.h)
#include "tg_train.h"  // tg_backward, optimizers (+ Tensor struct)
```

`src/` is `PRIVATE`. Two headers live there and are not part of the public surface: `cuda_ops.h`
(kernel dispatch) and `tg_cuda_internal.h` (device plumbing for `tg_ops.c` / `tg_train.c` /
`tg_optim.c` / `tg_checkpoint.c`). A `.c` file that needs
both the public and internal CUDA functions includes `tg_cuda_internal.h`, which pulls in `tg_cuda.h`. A new public function gets its prototype in the matching
`include/ovg/` header; a new internal helper stays in `src/`.

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

Working style:

* Read nearby code before editing; this project values explicit tensor math and inspectability over clever abstractions.
* Keep changes narrow and easy to review. Avoid unrelated refactors, formatting churn, or metadata updates.
* Preserve user work in the git tree. If unrelated files are dirty, leave them alone.
* New public functions need a prototype in the matching header. Include the narrowest header that provides what you need (see Include Style).
* New source files go in the lowest layer that has everything they need (see Library Layers), and get added to that layer's source list in `CMakeLists.txt`.

When modifying code:

* Preserve explicit tensor math — do not hide operations behind abstractions.
* Do not add broadcasting as a convenience fix. If a shape doesn't line up, make it explicit with `tg_reshape` / `tg_expand_dim` at the call site.
* Do not manually free intermediate graph tensors after a training step; call `tg_free_graph(loss)` and let it handle every non-persistent node. Through `TgAdam`, `tg_adam_accumulate` already does this — a trailing `tg_free_graph(loss)` is a use-after-free.
* Create `TgAdam` after moving params to their device; call `tg_seed` before `tg_checkpoint_load_run`.
* If an op allocates auxiliary data for backward (`cache`), make ownership obvious and verify `tg_free_graph` cleans it up.
* Preserve the backward function paired with each op in `tg_ops.c`.
* Do not introduce external ML libraries.
* Do not add silent broadcasting — it changes semantics for all callers.
* When adding a new op, add both the forward function and its `_backward` counterpart.
* When adding a CUDA kernel, ensure the CPU path remains correct and the dispatch logic is symmetric.
* `tg_block_forward` reshapes 2D `[T, C]` input to `[1, T, C]` at the start — the whole block operates in 3D. Attention output is `[B, T, C]` (3D), not `[B*T, C]`.
* `tg_gpt_forward` returns `[B*T, V]` (2D), ready for cross-entropy.
* The checkpoint magic is `0x00475633` (v3). v2 (`0x00475632`) files load weights-only; the v1 magic `0x00475643` is rejected on load.
