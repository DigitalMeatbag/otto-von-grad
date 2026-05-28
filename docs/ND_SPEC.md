# otto-von-grad v2 — Spec

> **Status:** Complete. This is the authoritative spec for the v2 N-D migration. Implementation order is fixed by the dependency chain: `tg_reshape` → N-D op migration (including `tg_expand_dim`, `tg_cast`) → architecture module migration (`TgSelfAttention` is the gate for the rest). BF16 ships in the same pass.

---

## 1. Strategic Intent

OVG was built as a learning project. That phase is complete.

The next few revisions reposition OVG as a capable, standalone autograd library — a dependency for two active sibling repositories with different structural requirements. Future work on OVG is motivated by the needs of both consumers simultaneously, not by ViT alone.

**Consuming projects:**

| Project | Model family | Key constraint |
|---|---|---|
| **vexilloscope** | ViT encoder — already implemented, >400 flags, hitting scale limits | Encoder attention, patch embeddings, CLS token, large output head |
| **GPT sibling** (unnamed) | Decoder-only language model, character-level or larger | Causal attention, large vocab, long context |

Both share the transformer backbone. Differences are: causal masking (GPT) vs. none (ViT), pooling strategy (row-0 CLS vs. mean), and input type (token ids vs. patch projections).

---

## 2. Hardware Constraints

All training and inference runs on a single machine with no multi-GPU path.

| Component | Spec | Constraint |
|---|---|---|
| GPU | RTX 4070 Super, 12 GB GDDR6X | Hard VRAM ceiling. No second GPU. |
| CPU | Ryzen 9 7950X3D | Not the bottleneck |
| RAM | 64 GB | Not the bottleneck |
| Storage | SSD | Not the bottleneck |

**Implications for v2:**

- 12 GB is the absolute ceiling for model weights + activations + gradient buffers combined. No swapping.
- Running single samples (v1's effective batch size of 1) leaves the 4070 Super severely underutilized. Batching is mandatory, not optional, for acceptable GPU efficiency.
- Memory bandwidth matters more than raw FLOP count on the 4070 Super (504 GB/s). In-place ops and avoiding unnecessary copies are meaningful wins.
- All optimization decisions are made in the context of this single machine. There is no "scale out" path.

---

## 3. What's Wrong with v1

v1's 2D-only `[rows × cols]` tensor constraint is the root cause of most current limitations:

- **No batching.** Batch dimension can't be represented. GPU sits mostly idle during training.
- **Workaround shapes.** Attention scores that are conceptually `[B, H, T, T]` get squeezed into 2D, which hides structure and makes multi-head batching impossible without rewriting the whole attention module.
- **Broadcast ops are materialized.** `tg_repeat_rows` / `tg_repeat_cols` work but are semantically clunky as the permanent solution to what should be strided access.
- **Graph node pressure.** At batch size 1, `TG_MAX_GRAPH = 4096` is comfortable. With batching, intermediate node counts multiply.

These are not bugs — they are correct behaviors within a deliberately narrow design. v2 lifts the constraint.

---

## 4. Resolved Decisions

### 4.1 — N-D Tensor: Option A
Replace the `rows` / `cols` fields with a proper N-D representation.

**Design:**
- Add `int ndim` and `int shape[TG_MAX_DIMS]` where `TG_MAX_DIMS = 4` (covers all current and anticipated shapes: `[B, H, T, D]`).
- `rows` and `cols` are removed. No aliases, no deprecated wrappers.
- All tensor data remains flat (`float *data`). Shape interpretation is the caller's responsibility, same as now.
- `tg_new` gains an N-D factory with signature `tg_new(int ndim, const int shape[])`. C11 compound literals keep call sites concise: `tg_new(4, (int[]){B, H, T, D})`. The 2-argument `tg_new(rows, cols)` form is removed — hard cut, no shim (see §5.4).

**Why Option A over Option B (add `int batch` field):**
- Option B hits a structural ceiling at `[B, H, T, T]` attention scores — that's already 4D and `batch` can't express it.
- Option A pays a larger upfront migration cost but has no second ceiling. One reshape, done.

**TG_MAX_DIMS = 4 rationale:** This is a practical static ceiling — the same kind of constant as `TG_MAX_PARENTS = 8` and `TG_MAX_GRAPH = 8192`. The value 4 is not arbitrary: the deepest shape in scope is `[B, H, T, D]` (batch, heads, sequence, dim) from attention scores, which is exactly 4D. So 4 is the minimum sufficient value for the planned workload, not a principled architectural identity claim. If a future op genuinely required a 5th dimension, `TG_MAX_DIMS` would be bumped to 5, the same way `TG_MAX_GRAPH` would be raised if graph depth exceeded the current limit.

**Minimum ndim:** All tensors have `ndim ≥ 2`. Scalars are `[1, 1]`, bias vectors are `[1, C]`. A `tg_new` call with `ndim < 2` is a fatal error.

**`tg_reshape`:** The primary N-D primitive for changing tensor shape. Signature: `tg_reshape(Tensor *a, int ndim, const int shape[])`. Returns a new tensor with its own allocation; data is copied from `a`. Total element count must be preserved (`product(shape) == product(a->shape)`); violation is a fatal error. A reshape to `ndim < 2` is also a fatal error, consistent with the `ndim ≥ 2` invariant established by `tg_new`. Backward: accumulates `out->grad` (reshaped back to `a`'s shape) into `a->grad`. This is the first op to land in the v2 migration — the attention and LayerNorm broadcast paths both depend on it.

**Tensor struct — final v2 layout:**

```c
typedef struct Tensor {
    int      ndim;
    int      shape[TG_MAX_DIMS];
    TgDtype  dtype;
    float   *data;
    float   *grad;
    int      persistent;
    int      on_cuda;
    /* backward linkage and parent tracking: unchanged from v1 */
} Tensor;
```

`rows` and `cols` are removed. There are no aliases, no deprecated wrappers.

### 4.2 — No Silent Broadcasting
The v1 rule holds: shape mismatches must fail loudly. OVG does not silently broadcast.

The v1 ops `tg_repeat_rows` and `tg_repeat_cols` are removed and replaced by `tg_expand_dim(Tensor *a, int axis, int n)`, which expands `a` along `axis` from size 1 to size `n`. The target axis must have size 1; any other size is a fatal error. This generalizes to arbitrary dimensions with no 2D-specific assumptions. The batch dimension is **not** a special case — a `[1, H, T, D]` tensor is not automatically compatible with a `[B, H, T, D]` tensor. Explicit expansion via `tg_expand_dim` is required. Returns a new tensor with its own allocation; input data is tiled `n` times along `axis`. Backward: sums `out->grad` over `axis`, accumulating into `a->grad` (contracting that axis back to size 1).

To cross an ndim boundary before expanding (e.g., broadcasting `[1, C]` parameters over a `[B, T, C]` input), use `tg_reshape` first: `[1, C]` → `tg_reshape` → `[1, 1, C]` → `tg_expand_dim(axis=0, n=B)` → `[B, 1, C]` → `tg_expand_dim(axis=1, n=T)` → `[B, T, C]`.

This keeps ownership and shape flow visible in the graph, which is essential for debugging in a hand-written autograd engine.

### 4.3 — CUDA Support Stays Optional
`OVG_CUDA=ON` / `OVG_CUDA=OFF` via CMake remains the mechanism. The CPU path must stay correct and symmetric. No CUDA-only op implementations — every op's forward and backward must have a working CPU path. Exception: `TG_DTYPE_BF16` is a CUDA-only opt-in dtype; the `dtype` field and `tg_cast` op exist on both paths, but tensors with `dtype == TG_DTYPE_BF16` assert on CPU. See §5.2.

### 4.4 — C11, No ML Framework Dependencies
OVG is written in C11. No PyTorch, no ONNX, no ML framework dependencies. Vendor math libraries (cuBLAS, standard BLAS) are permitted where they provide meaningful throughput gains without obscuring the underlying logic — see §5.1.

---

## 5. Decisions

### 5.1 — cuBLAS

**Decision: cuBLAS is allowed for matmul. All other ops remain hand-written kernels.**

cuBLAS ships with the CUDA toolkit, which is already a required dependency (`find_package(CUDAToolkit REQUIRED)`). Adding it requires one CMake line (`CUDA::cublas`); nothing new to install.

The "no external libraries" constraint is reframed to match its actual intent: **no ML frameworks, no wrappers around things you don't understand**. Vendor math libraries (cuBLAS, and by the same logic standard CPU BLAS if it ever becomes relevant) are fair game. The autograd logic, graph structure, op definitions, and backward passes all remain hand-written — cuBLAS only handles the raw matrix multiply kernel that runs inside `tg_matmul`.

This also updates the constraint stated in §4.4: the rule is now "no ML framework dependencies," not "no external libraries."

`tg_matmul` dispatches to the appropriate cuBLAS routine based on input ndim: `cublasSgemm` for 2D×2D FP32, `cublasSgemmStridedBatched` for N-D FP32 inputs (e.g., 3D×2D weight projections, 4D×4D attention scores). For a 2D weight projected over a batched input, `cublasSgemmStridedBatched` is called with `strideB = 0` for the weight operand — the standard cuBLAS idiom for broadcasting a single matrix across a batch without replication. When both inputs are BF16, the appropriate variant is `cublasGemmEx` with `CUDA_R_16BF` data type and `CUBLAS_COMPUTE_32F` accumulation — there is no `cublasHgemm` equivalent for BF16; `cublasGemmEx` is always used. cuBLAS handles the precision boundary automatically. The cuBLAS permission covers all of these variants — they are all matmul. Architecture modules always call `tg_matmul`; they do not reach cuBLAS directly.

**`tg_matmul` shape rules.** Inputs contract over inner dimensions: `A[..., M, K] @ B[..., K, N] → [..., M, N]`. The K dimensions must match exactly; mismatch is fatal.

- **Same ndim:** Leading dimensions must match element-wise; mismatch is fatal.
- **Different ndim — second operand lower:** If `ndim(B) < ndim(A)`, B is implicitly treated as having shape `(1, ..., 1, K, N)` and broadcast over A's leading dims. This is the intended path for weight projections: `[B, T, C] @ [C, C_out] → [B, T, C_out]`. Only the second operand may be lower-ndim; first-operand-lower is fatal.

**`tg_matmul` backward.** Let `dOut` be the incoming gradient `[..., M, N]`:
- `dA = dOut @ B^T` → shape `[..., M, K]`.
- `dB`: if `ndim(A) == ndim(B)`, standard `A^T @ dOut`. If `ndim(B) < ndim(A)`, sum `A^T @ dOut` over the broadcast leading dims before accumulating into `B->grad` (contracting back to B's shape).

### 5.2 — Mixed Precision (BF16)

**Decision: mixed precision as an opt-in CUDA-only path (BF16). FP32 remains the default.**

The BF16 path is in scope for v2 and ships alongside the N-D migration — it is not deferred. The `Tensor` struct gains a `dtype` field as part of the N-D struct redesign; stubbing it in now avoids a second struct migration.

The field type is `TgDtype`, an enum with two values: `TG_DTYPE_F32` (default) and `TG_DTYPE_BF16`. The `DTYPE_` segment is required — C enum members are global-scope and `TG_F32` is a collision risk. The CPU path only ever holds `TG_DTYPE_F32`; `TG_DTYPE_BF16` is only valid on CUDA tensors (Ampere / sm_80 or newer) and will assert otherwise. By 2026, Ampere is the de facto baseline for consumer ML hardware; the RTX 4070 Super (sm_89, Ada Lovelace) fully supports it.

**Dtype conversion:** precision changes are explicit graph nodes via `tg_cast(Tensor *a, TgDtype dtype)`. There is no dtype-at-construction argument to `tg_new` — all tensors are allocated as `TG_DTYPE_F32` and cast explicitly when needed. This keeps dtype changes visible in the computation graph. Returns a new tensor with its own allocation. The backward pass through `tg_cast` converts the incoming gradient to the input tensor's dtype. Calling `tg_cast(a, a->dtype)` is a fatal error.

BF16 has the same exponent range as FP32 — gradients do not underflow in the backward pass. Loss scaling is not required. `tg_backward` (seed = 1.0f) is used unchanged for both FP32 and BF16 paths.

Breakdown of what runs at which precision:
- **Activations and weights:** BF16 on the CUDA path, via explicit `tg_cast` after allocation
- **Optimizer state (Adam m/v buffers):** FP32 — BF16 has only 7 mantissa bits; small gradient accumulations quantize to zero at BF16 precision, corrupting the moment estimates
- **Matmul accumulation:** FP32 internally even on Tensor Cores — cuBLAS handles this boundary automatically via `CUBLAS_COMPUTE_32F`

The CPU path stays FP32-only. No BF16 benefit exists there (Tensor Cores are GPU hardware).

### 5.3 — Memory Strategy Under Batching
**Decision: keep per-op `malloc`. Raise `TG_MAX_GRAPH` from 4096 to 8192.**

Batching increases the size of each allocation (larger tensors) but not the number of graph nodes — batch is a dimension in the shape, not additional nodes. The per-step `malloc` count stays the same as today. `TG_MAX_GRAPH` is a node-count ceiling, so the main risk is deeper models pushing against it; 8192 is a comfortable margin for the planned workloads.

Arena or pool allocation is deferred until profiling shows allocation overhead is actually a problem. The N-D struct migration is enough structural change to manage at once.

### 5.4 — API Break Strategy
**Decision: hard cut. No shim layer.**

`tg_new(rows, cols)` is removed. All callers migrate to the N-D API at once. Sibling repos (vexilloscope, GPT sibling) update when OVG lands the change — they are not in production and the coordination cost is acceptable. Compatibility shims are explicitly rejected: they signal the old API is still valid when it isn't, and they linger past their usefulness.

### 5.5 — TG_MAX_GRAPH and TG_MAX_PARENTS
**Decision: `TG_MAX_GRAPH` raised to 8192 (captured in §5.3). `TG_MAX_PARENTS = 8` unchanged.**

Batching doesn't change graph topology — ops take the same number of inputs regardless of batch size, and no current or planned op approaches the 8-parent limit.

The recursive topo sort is safe at current model depths (~200). Converting to iterative is a known future item — tracked here as a non-blocking note — not a prerequisite for the N-D migration.

---

## 6. Principles

These govern tradeoff decisions throughout v2.

**1. Explicit over implicit.** No silent broadcasting. No inferred shapes. If a shape is unexpected, fail loudly. This is non-negotiable — it is what makes a hand-written autograd engine debuggable. `tg_matmul`'s batch broadcast (§5.1) is the single defined exception — it is written and explicit in the spec, not silent.

**2. Correctness before performance.** v2 targets real performance (batching, GPU utilization), but a faster incorrect result is worthless. Every new op ships with a CPU path and tests.

**3. Performance is now explicitly in scope.** v1 deferred performance. v2 cannot — the hardware is fixed and the workloads require real throughput. Performance decisions must be justified by the hardware constraints in §2, not by general principle.

**4. Both consumers.** Every OVG design decision is evaluated against both vexilloscope (encoder, ViT) and the GPT sibling (decoder, language). A change that helps one at the expense of the other requires explicit acknowledgment.

**5. No ML framework dependencies.** Vendor math libraries (cuBLAS) are permitted — see §4.4 and §5.1. ML frameworks (PyTorch, ONNX, etc.) are off the table.

**6. The CPU path stays correct.** CUDA is optional. The CPU fallback must pass all tests and produce correct results. No op implementation is CUDA-only. Exception: `TG_DTYPE_BF16` tensors are invalid on CPU and assert — BF16 is a CUDA-only dtype, not a CUDA-only op.

**7. Ownership is obvious.** `persistent = 1` marks parameters. `tg_free_graph` handles intermediates. This ownership model carries forward; batching must not require callers to track additional allocations.

---

## 7. v1 Features Carrying Forward

Everything in the current v1 API carries forward unless explicitly displaced by a v2 decision:

- All ops in `tg_ops.h` (elementwise, matmul, attention, norm, loss, etc.), with the following exceptions and changes:
  - **Removed:** `tg_repeat_rows`, `tg_repeat_cols` (replaced by `tg_expand_dim` — see §4.2). `tg_concat_cols`, `tg_concat_rows` — hard cut, no N-D replacement. `tg_slice_cols` — hard cut; replaced by `tg_slice` (see New ops below).
  - **Generalized:** `tg_transpose(a)` → `tg_transpose(a, dim0, dim1)` (swap any two axes; old one-argument form is a hard cut, no shim). Backward: `tg_transpose(out->grad, dim0, dim1)` accumulated into `a->grad`. `tg_softmax_rows(a)` → `tg_softmax(a, axis)` (softmax along a specified axis). Backward: as v1 `tg_softmax_rows`, generalized to the specified axis. `tg_causal_mask(a)` — signature unchanged; implementation generalizes from 2D `[T, T]` to N-D, masking the last two dimensions uniformly across all leading dims. Backward: unchanged from v1. `tg_layer_norm` normalizes over `ndim-1` (the feature/channel axis) by design — no `axis` parameter. LayerNorm is defined over the feature dimension and the axis does not vary at call sites; an `axis` parameter would be misleading API surface with no real use case. Backward: unchanged from v1, applied over axis `ndim-1`.
  - **New:** `tg_reshape(Tensor *a, int ndim, const int shape[])` (see §4.1). `tg_expand_dim(Tensor *a, int axis, int n)` (see §4.2). `tg_cast(Tensor *a, TgDtype dtype)` (see §5.2). `tg_scale(Tensor *a, float scalar)` — multiply every element of `a` by `scalar`. Returns a new tensor. Backward: `a->grad += scalar * out->grad`. `tg_slice(Tensor *a, int axis, int start, int len)` — extract a contiguous slice of `len` elements starting at `start` along `axis`. Returns a new tensor with its own allocation; output shape is `a->shape` with `shape[axis]` replaced by `len`. `start + len` must not exceed `a->shape[axis]`; violation is a fatal error. Backward: accumulates `out->grad` into the corresponding slice of `a->grad` (the non-sliced region receives no gradient contribution from this op).
- `TgSelfAttention`, `TgBlock`, `TgTransformer`, `TgGPT` architecture modules — these carry forward in function but require significant internal rewrites. The op migration is a prerequisite; no module can be migrated until all ops it depends on support N-D shapes. Key decisions recorded here:

  **`TgSelfAttention` is the epicenter.** The current per-head loop (`tg_slice_cols` → `tg_transpose` → `tg_matmul` → `tg_causal_mask` → `tg_softmax_rows` → `tg_matmul`, × n_heads, then `tg_concat_cols`) is replaced by a reshape+strided approach. The full forward sequence: reshape `Q/K/V` from `[B, T, C]` to `[B, H, T, D]`; transpose K: `K_t = tg_transpose(K_4d, 2, 3)` → `[B, H, D, T]`; compute `Q @ K_t` via `tg_matmul` → `[B, H, T, T]`; scale by `1/√D` via `tg_scale(q_kt, 1.0f / sqrtf((float)D))`; apply `tg_causal_mask` on `[B, H, T, T]` (GPT only, skip for ViT); apply `tg_softmax(axis=3)`; compute `Attn @ V` via a second `tg_matmul` → `[B, H, T, D]`; reshape back to `[B, T, C]`. This eliminates the `n_heads > TG_MAX_PARENTS` ceiling and is the correct path for Tensor Core utilization on the 4070 Super. It requires `tg_reshape` to land before `tg_attention_forward` can be migrated.

  **Weight broadcast pattern.** All attention projection weights (`Wq`, `Wk`, `Wv`, `Wo`) remain 2D (`[C, C]`). FFN weights are `W1: [C, ffn_dim]` and `W2: [ffn_dim, C]`, where `ffn_dim` is typically `4×C`. (`ffn_dim` is used here to avoid collision with `H`, which denotes head count throughout this document.) Batched matmul against a 3D input (`[B, T, C] @ [C, C]`) is the dominant pattern throughout and must be supported by `tg_matmul` and the cuBLAS path.

  **`TgBlock` LayerNorm parameters** (`gamma`, `beta`) are `[1, C]` and must broadcast over `[B, T, C]` inputs. `tg_layer_norm` normalizes over the last axis (axis `ndim-1`, i.e., the `C` dimension) for any N-D input. This crosses an ndim boundary: use `tg_reshape` to lift to `[1, 1, C]`, then `tg_expand_dim` twice (axis 0 → B, axis 1 → T). See §4.2 for the full sequence. No implicit broadcasting.

  **`tg_gpt_forward` input interface changes.** The current `tg_gpt_forward(TgGPT *g, const int *token_ids)` takes a flat array of length `seq_len`. The batched signature is `tg_gpt_forward(TgGPT *g, const int *token_ids, int batch_size)` where `token_ids` is a flat `[B × T]` row-major int array. `T` (`seq_len`) is a field of `TgGPT` and is not passed separately. This is consistent with tensor data layout and GPU transfer requirements — pointer-of-pointers is not used.

- `tg_backward` / `tg_backward_accum` / `tg_free_graph` training loop
- Adam and SGD optimizers (`tg_adam_step`, `tg_adam_step_gpu`, `tg_sgd_step`)
- `tg_clip_grad_norm` (GPU path stays device-side)
- Checkpoints (`tg_checkpoint_save` / `tg_checkpoint_load`)
- Tokenizer, sampler, RNG
- `ovg_fatal` / `ovg_set_fatal_handler` error handling model
- `tg_training` global flag (dropout, drop path)
- CUDA dispatch pattern: `t->on_cuda == 1` → GPU kernel, else CPU

---

## 8. What v2 Does Not Include

These are explicitly out of scope for v2 unless a future decision reopens them:

- Multi-GPU / distributed training
- Model parallelism
- Gradient checkpointing
- Dynamic computation graphs (v2 is still static-graph-per-step)
- Automatic mixed precision — BF16 support (§5.2) is explicit and opt-in, not automatic
- A Python binding
- ONNX export
- Any external ML framework dependency
