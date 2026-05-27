# otto_von_grad

Reverse-mode autograd engine and neural-network infrastructure, written from scratch in C.

---

## Tensor Autograd API

### Tensor lifecycle (`tg_tensor.h`)

```c
Tensor *tg_new(int rows, int cols);
void    tg_free(Tensor *t);
void    tg_fill(Tensor *t, float val);
void    tg_fill_uniform(Tensor *t, float lo, float hi);
void    tg_fill_randn(Tensor *t, float scale);
void    tg_fill_xavier_uniform(Tensor *t);
void    tg_fill_xavier_normal(Tensor *t);
float   tg_scalar_value(Tensor *t);
void    tg_print(const Tensor *t, const char *name);
```

`Tensor.persistent = 1` marks a tensor as exempt from `tg_free_graph` (use for parameters
and pre-allocated inputs that you want to free manually).

### Ops (`tg_ops.h`)

All ops return a new `Tensor*` that participates in the computation graph.

| Op | Shape | Notes |
|---|---|---|
| `tg_add(a, b)` | same → same | element-wise |
| `tg_sub(a, b)` | same → same | element-wise |
| `tg_mul(a, b)` | same → same | element-wise |
| `tg_pow(a, p)` | same → same | element-wise a^p |
| `tg_scale(a, s)` | same → same | element-wise a*s |
| `tg_matmul(a, b)` | [m,k] @ [k,n] → [m,n] | |
| `tg_transpose(a)` | [r,c] → [c,r] | |
| `tg_tanh(a)` | same → same | |
| `tg_relu(a)` | same → same | |
| `tg_gelu(a)` | same → same | tanh-approx GELU |
| `tg_sum(a)` | any → [1,1] | all elements |
| `tg_mean(a)` | any → [1,1] | all elements |
| `tg_mean_rows(a)` | [R,C] → [1,C] | column-wise mean |
| `tg_slice_cols(a, s, e)` | [R,C] → [R,(e-s)] | |
| `tg_concat_cols(parts, n)` | n×[R,C] → [R,n*C] | |
| `tg_row_slice(a, s, e)` | [R,C] → [(e-s),C] | |
| `tg_concat_rows(parts, n)` | n×[R,C] → [n*R,C] | all parts must have equal cols |
| `tg_repeat_rows(a, n)` | [1,C] → [n,C] | input must be exactly 1 row; backward sums grads across rows |
| `tg_repeat_cols(a, n)` | [R,1] → [R,n] | input must be exactly 1 col; backward sums grads across cols |
| `tg_embed(weight, ids, T)` | [V,C], int[T] → [T,C] | gather rows by token id |
| `tg_causal_mask(scores)` | [T,T] → [T,T] | -1e9 on future positions |
| `tg_layer_norm_rows(a, eps)` | same → same | row-wise normalization |
| `tg_layer_norm_rows_affine(a, gamma, beta, eps)` | [R,C],[1,C],[1,C] → [R,C] | row-wise normalization with learned scale/bias |
| `tg_softmax_rows(a)` | same → same | row-wise softmax |
| `tg_cross_entropy(logits, targets)` | [T,V],[T,V] → [1,1] | mean CE, targets one-hot |
| `tg_cross_entropy_no_sync(logits, targets)` | [T,V],[T,V] → [1,1] | CUDA path skips scalar host sync |
| `tg_cross_entropy_sparse(logits, ids, n, smooth)` | [T,V], int[T] → [1,1] | mean CE with integer labels |
| `tg_cross_entropy_sparse_no_sync(logits, ids, n, smooth)` | [T,V], int[T] → [1,1] | sparse CE, CUDA path skips scalar host sync |
| `tg_dropout(a, p)` | same → same | inverted dropout; no-op when `tg_training == 0` or `p == 0` |

### Training (`tg_train.h`)

```c
extern int tg_training;  // 1 = training mode (default), 0 = inference — disables dropout

void tg_backward(Tensor *root);          // zero all grads in graph, then reverse-mode autodiff
void tg_backward_accum(Tensor *root);    // backward without zeroing grads (accumulate)
void tg_zero_grads(Tensor **params, int n);           // zero grad arrays for a param list
void tg_sgd_step(Tensor **params, int n, float lr);  // in-place SGD
void tg_adam_step(Tensor **params, float **m, float **v, int n_params,
                  float lr, int step, float beta1, float beta2, float eps);  // bias-corrected Adam
float tg_clip_grad_norm(Tensor **params, int n_params, float max_norm, float eps);
// Note: in the CUDA path the norm is computed and clipped entirely on device;
// the return value is 0.0f (norm is not read back to the CPU).
void tg_free_graph(Tensor *root);        // free all non-persistent nodes in graph
```

`tg_free_graph` is the correct way to clean up after each training step. It walks the
computation graph in topological order and frees every tensor that isn't marked `persistent`.
Call it on `loss` at the end of each step instead of manually freeing intermediates.

---

## Architecture Modules

### TgSelfAttention (`attention.h`)

Multi-head self-attention. `causal=1` for GPT (masks future positions); `causal=0` for encoders.

```c
TgSelfAttention tg_attention_create(int embed_dim, int n_heads);          // causal
TgSelfAttention tg_attention_create_encoder(int embed_dim, int n_heads);  // non-causal
```

### TgBlock (`tg_block.h`)

Pre-norm transformer block: LayerNorm → Attention → dropout → residual → LayerNorm → FFN → dropout → residual.

```c
TgBlock tg_block_create(int embed_dim, int hidden_dim, int seq_len, int n_heads);
TgBlock tg_block_create_encoder(int embed_dim, int hidden_dim, int seq_len, int n_heads);
```

`TgBlock.dropout` (float, default 0) — set after creation to enable dropout.
`TgBlock.drop_path_rate` (float, default 0) — stochastic depth rate per block; applied to both residual branches during training. Set automatically by `tg_transformer_create_encoder` via a linear depth schedule.
Both are skipped during inference when `tg_training == 0`.

Note: bias tensors `B1` and `B2` are explicitly shaped `[seq_len × dim]` — no broadcasting.

### TgTransformer (`tg_transformer.h`)

Stack of `TgBlock`s.

```c
TgTransformer tg_transformer_create(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads);
TgTransformer tg_transformer_create_encoder(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads, float max_drop_path_rate);
Tensor       *tg_transformer_forward(TgTransformer *t, Tensor *X);
```

### TgGPT (`tg_gpt.h`)

Token + positional embeddings → transformer → output projection. Trained on `data/text/candide.txt`
at character level; generates plausible character-texture output.

```c
typedef struct {
    int vocab_size, embed_dim, hidden_dim, seq_len, n_blocks, n_heads;
} TgGPTConfig;

TgGPT   tg_gpt_create(int vocab_size, int embed_dim, int hidden_dim, int seq_len, int n_blocks, int n_heads);
TgGPT   tg_gpt_create_from_config(const TgGPTConfig *cfg);
Tensor *tg_gpt_forward(TgGPT *g, const int *token_ids);   // int[seq_len] → [T×V] logits
int     tg_gpt_collect_params(TgGPT *g, Tensor **params, int max_params);
// param count = 3 + n_blocks * 12
```

### TgVocab / tokenizer (`tg_tokenizer.h`)

Character-level vocabulary over raw bytes. Every unique byte in the corpus gets a sequential id.

```c
typedef struct { char chars[256]; int ids[256]; int size; } TgVocab;

char    *tg_read_file(const char *path, int *out_len);             // malloc'd; caller frees
TgVocab  tg_vocab_build(const char *text, int len);
int      tg_vocab_encode(const TgVocab *v, char c);                // ovg_fatal on unknown char
char     tg_vocab_decode(const TgVocab *v, int id);                // ovg_fatal on bad id
int     *tg_tokenize(const char *text, int len, const TgVocab *v); // malloc'd; caller frees
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

`tg_sample_topk`: scales logits by `1/temperature`, restricts to the top-`k` candidates, then
samples from their softmax distribution. `top_k=0` uses the full vocab; `top_k=1` is deterministic
argmax. `tg_generate` manages the sliding context window and sets `tg_training=0` for the
duration of generation.

### Checkpoints (`tg_checkpoint.h`)

```c
int tg_checkpoint_save(const char *path, Tensor **params, int n);  // 0 on success, -1 on error
int tg_checkpoint_load(const char *path, Tensor **params, int n);  // silent -1 if file missing
```

Saves all param tensors to a flat binary file (magic header + shapes + float data). Load validates
magic, param count, and per-tensor shapes. A missing file returns `-1` silently (expected on first
run); any other failure writes to stderr.

Checkpoint files are written to `data/checkpoints/` (gitignored).

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
- `ottovongrad` — static library
- `otto_von_grad` — GPT character-level demo: trains on `data/text/candide.txt`, reports train and
  val loss every 200 steps, then generates text using temperature + top-k sampling. Saves a
  checkpoint to `data/checkpoints/model.bin` on exit; resumes from it automatically on the next run.
- `otto_von_grad_tests` — test suite

---

## Testing

```powershell
cmake --build --preset default
.\build\otto_von_grad_tests.exe   # 54 tests; exits 0 on all-pass

# Or via CTest
ctest -C Release --test-dir build   # use -C for Visual Studio multi-config builds
```

See [`tests/README.md`](tests/README.md) for details on the test structure.

---

## Error Handling

All fatal errors in the library go through `ovg_fatal()` rather than calling `exit(1)` inline.
Consumers can install a custom pre-exit hook (e.g. for logging or test capture) before any
library calls:

```c
#include "ovg_error.h"

void my_handler(const char *msg) {
    my_logger_write(msg);
    // ovg_fatal() will still call exit(1) after this returns
}

ovg_set_fatal_handler(my_handler);
```

`ovg_set_fatal_handler` is not thread-safe — call it before spawning any threads.

---

## RNG and Seeding (`tg_rng.h`)

```c
void     tg_seed(uint32_t seed);       // seed rand() + dropout xorshift32, log seed to stdout
void     tg_seed_from_entropy(void);   // seed from OS entropy (rand_s / arc4random / /dev/urandom)
uint32_t tg_rng_xorshift32(void);      // raw xorshift32 step (used internally by tg_dropout)
float    tg_rng_uniform(void);         // uniform float in [0, 1) — wraps xorshift32
```

Call `tg_seed_from_entropy()` once at startup (already done in `main.c`). The chosen seed
is printed so any run can be reproduced:

```
[ovg] rng seed: 0x5f3759df
```

To replay that run, replace the call with `tg_seed(0x5f3759df)`.

---

## Usage as a Dependency

`otto-von-grad` is designed to be consumed as a CMake static library. The `ottovongrad`
target is exported with `PUBLIC` include directories, so consumers just link and include:

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
