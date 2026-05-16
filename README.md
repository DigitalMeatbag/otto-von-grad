# otto_von_grad

Reverse-mode autograd engine and neural-network infrastructure, written from scratch in C.

Contains two historical layers that illustrate the progression from scalar to tensor autodiff,
then from simple MLPs up to a full GPT-style transformer.

---

## Layers

### Scalar autograd (reference)

Files: `value.c / value.h`, `mlp.c / mlp.h`

- Scalar computation graph with topological sort and reverse traversal
- `Value` struct: data, grad, backward function pointer, parent pointers
- Scalar MLP demonstrating XOR learning

This layer is kept as a reference. The tensor layer supersedes it.

### Tensor autograd (active engine)

Files: `tg_tensor.h/c`, `tg_ops.h/c`, `tg_train.h/c`, `tg_mlp.h/c`,
`attention.h/c`, `tg_block.h/c`, `tg_transformer.h/c`, `tg_gpt.h/c`

---

## Tensor Autograd API

### Tensor lifecycle (`tg_tensor.h`)

```c
Tensor *tg_new(int rows, int cols);
void    tg_free(Tensor *t);
void    tg_fill(Tensor *t, float val);
void    tg_fill_randn(Tensor *t, float scale);
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
| `tg_sum(a)` | any → [1,1] | all elements |
| `tg_mean(a)` | any → [1,1] | all elements |
| `tg_mean_rows(a)` | [R,C] → [1,C] | column-wise mean |
| `tg_slice_cols(a, s, e)` | [R,C] → [R,(e-s)] | |
| `tg_concat_cols(parts, n)` | n×[R,C] → [R,n*C] | |
| `tg_causal_mask(scores)` | [T,T] → [T,T] | -1e9 on future positions |
| `tg_layer_norm_rows(a, eps)` | same → same | row-wise normalization |
| `tg_softmax_rows(a)` | same → same | row-wise softmax |
| `tg_cross_entropy(logits, targets)` | [T,V],[T,V] → [1,1] | mean CE, targets one-hot |
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

`TgBlock.dropout` (float, default 0) — set after creation to enable dropout (e.g. `block.dropout = 0.1f`).
Dropout is automatically skipped during inference when `tg_training == 0`.

Note: bias tensors `B1` and `B2` are explicitly shaped `[seq_len × dim]` — no broadcasting.

### TgTransformer (`tg_transformer.h`)

Stack of `TgBlock`s.

```c
TgTransformer tg_transformer_create(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads);
TgTransformer tg_transformer_create_encoder(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads);
Tensor       *tg_transformer_forward(TgTransformer *t, Tensor *X);
```

### TgGPT (`tg_gpt.h`)

Token + positional embeddings → transformer → output projection. Trained on `candide.txt`
at character level; generates plausible character-texture output.

```c
TgGPT   tg_gpt_create(int vocab_size, int embed_dim, int hidden_dim, int seq_len, int n_blocks, int n_heads);
Tensor *tg_gpt_forward(TgGPT *g, Tensor *token_one_hot);  // [T×V] → [T×V] logits
int     tg_gpt_collect_params(TgGPT *g, Tensor **params);
```

---

## Build

```powershell
cd otto-von-grad
cmake -B build
cmake --build build
.\build\Debug\otto_von_grad.exe

# With CUDA
cmake -B build -DOVG_CUDA=ON
cmake --build build
.\build\Debug\otto_von_grad.exe
```

CMake produces two targets:
- `ottovongrad` — static library (consumed by `vexilloscope`)
- `otto_von_grad` — standalone executable

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
