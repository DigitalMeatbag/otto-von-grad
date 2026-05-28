# Remediation — otto-von-grad v2 Code Review

Findings and fix plan from the post-v2 correctness review. Items are listed in execution order (High → Medium → Low → Tests). Each section names the file(s), the problem, and the intended fix so the changes can be reviewed against the plan as they land.

---

## H1 — Dense cross-entropy CUDA forward uses wrong loss formula

**Severity:** High  
**File:** `src/cuda_ops.cu` — `cross_entropy_fwd_k`

**Problem:** Thread 0 computes the per-row loss as:
```c
loss -= row_tgt[j] * logf(row_prb[j] + 1e-7f);
```
The CPU path (`cross_entropy_dense_impl` in `tg_ops.c`) uses the numerically stable log-sum-exp identity:
```c
loss -= tgt[j] * (row[j] - log_sum_exp);
```
For near-zero softmax probabilities the `+1e-7` floor caps the log and produces a materially different loss value than the CPU path. Gradients are unaffected (the backward uses cached probs via `(prob - target)/rows` in both paths), but loss telemetry diverges between CPU and GPU runs for identical inputs.

**Fix (dense kernel only — `cross_entropy_fwd_k`):** After `sum = smem[0]`, all threads already hold `sum` and `mx` as register values. Compute `float log_sum_exp = logf(sum) + mx;` in registers immediately at that point, before the probability-normalization loop — no smem slot needed. Replace the thread-0 serial loss loop with `loss -= row_tgt[j] * (row_in[j] - log_sum_exp)`.

The sparse kernel (`cross_entropy_sparse_fwd_k`) already computes `log_sum_exp` in registers (line 1073) and already uses the correct formula (line 1081) — H1 does not apply there.

This fix also unlocks Step L1 for the dense kernel, since `log_sum_exp` is then in all threads' registers.

---

## H2 — Dead non-affine LayerNorm CUDA code

**Severity:** High  
**Files:** `src/cuda_ops.cu` (~lines 604–683, 778–783), `src/cuda_ops.h` (~lines 89–93)

**Problem:** `layer_norm_rows_fwd_k`, `layer_norm_rows_bwd_k`, and their host launchers `cuda_layer_norm_rows_fwd` / `cuda_layer_norm_rows_bwd` are implemented and declared but never called. The non-affine `tg_layer_norm_rows` op was removed during the v2 N-D migration; only the affine path (`cuda_layer_norm_rows_affine_*`) is used by `tg_layer_norm`. The non-affine backward kernel would also compute incorrect `dx` if wired up accidentally (it doesn't apply gamma to `dxhat`).

**Fix:** Delete the two kernels and their launchers from `cuda_ops.cu`; delete the two prototypes from `cuda_ops.h`.

---

## M1 — No CUDA kernel error checking after launches

**Severity:** Medium  
**File:** `src/cuda_ops.cu`

**Problem:** `CUDA_CHECK` (defined in `tg_cuda.cu`) guards all explicit runtime calls (malloc, memcpy, memset) but not kernel launches. An out-of-bounds access or invalid configuration produces no diagnostic until the next unrelated CUDA call, making failures hard to localise.

**Fix:** Add `CUDA_CHECK(cudaGetLastError())` immediately after every kernel launch in `cuda_ops.cu`. Define the macro locally at the top of `cuda_ops.cu` (duplicating the definition from `tg_cuda.cu`) rather than extracting to a shared header, to keep the change self-contained. Approximately 40 launch sites.

---

## M2 — Integer overflow in shape-product computation

**Severity:** Medium  
**Files:** `src/tg_tensor.c` (`tg_new`), `src/tg_ops.c` (`backward_matmul`)

**Problem:** `int nel = 1; for ... nel *= shape[i]` overflows signed 32-bit for any tensor with more than ~2.1B elements. With BF16 (2 bytes/element) on a 12 GB GPU, a tensor filling the entire device would have 6B elements — beyond INT_MAX. The same overflow is possible in `int batch = tg_numel(A) / (M * K)` in `backward_matmul`.

**Decision:** Add overflow guards in `tg_new` and `backward_matmul` rather than promoting `tg_numel` to `long` throughout. The full promotion would be strictly correct, but the refactor is wide (every loop index, every printf format specifier) and disproportionate to the practical constraint: a BF16 tensor with >2.1B elements is not reachable on the target hardware (RTX 4070 Super, 12 GB). A comment in `tg_new` documents this decision.

**Fix in `tg_new`:**
```c
/* NOTE: nel computed as int. Promoting tg_numel to long would be strictly
   correct but the refactor is wide; >2.1B elements is unreachable on 12 GB
   VRAM in practice. This guard catches overflow before allocation. */
if (nel <= 0)
    ovg_fatal("tg_new: shape product overflows int (%d dims, first dim %d)",
              ndim, shape[0]);
```

**Fix in `backward_matmul`** (after `int batch = tg_numel(A) / (M * K)`):
```c
if (batch <= 0)
    ovg_fatal("backward_matmul: batch computation overflowed int");
```

---

## M3 — Stochastic depth drops the entire batch together, not per-sample

**Severity:** Medium  
**File:** `src/tg_block.c` — `tg_block_forward`

**Problem:** The current implementation samples one `tg_rng_uniform()` value and applies the same `tg_scale(A, 0.0 or inv_keep)` to the entire `[B, T, C]` residual. Canonical stochastic depth (Huang et al., 2016) drops each sample in the batch independently, giving every example its own Bernoulli draw.

**Fix:** Replace the two `tg_scale` stochastic-depth blocks with per-sample masking. Build a `[B, 1, 1]` tensor from B independent Bernoulli draws, expand it to `[B, T, C]` via two `tg_expand_dim` calls, then multiply with `tg_mul`. This keeps the mask in the autograd graph so gradients flow correctly through kept positions and zero through dropped ones.

```c
// Replaces the full body of:
//   if (tg_training && b->drop_path_rate > 0.0f) {
//       float u = tg_rng_uniform();
//       A = tg_scale(A, u < b->drop_path_rate ? 0.0f : 1.0f / (1.0f - b->drop_path_rate));
//   }
// with:
if (tg_training && b->drop_path_rate > 0.0f) {
    int B = A->shape[0], T = A->shape[1], C = A->shape[2];
    int mask_shape[3] = {B, 1, 1};
    Tensor *dp_mask = tg_new(3, mask_shape);
    float inv_keep = 1.0f / (1.0f - b->drop_path_rate);
    float *md = TG_DATAF(dp_mask);
    for (int i = 0; i < B; i++)
        md[i] = (tg_rng_uniform() >= b->drop_path_rate) ? inv_keep : 0.0f;
    Tensor *dp_1t = tg_expand_dim(dp_mask, 1, T);   /* [B, T, 1] */
    Tensor *dp_btc = tg_expand_dim(dp_1t, 2, C);    /* [B, T, C] */
    A = tg_mul(A, dp_btc);
}
```

Apply identically to the `F2` drop-path block.

---

## L1 — Thread 0 accumulates CE loss serially over all vocab columns

**Severity:** Low  
**File:** `src/cuda_ops.cu` — `cross_entropy_fwd_k`, `cross_entropy_sparse_fwd_k`

**Problem:** After the parallel softmax phase, 255 threads go idle while thread 0 loops serially over `cols` (up to 50 000+) to compute the row loss. For large vocabularies this is a significant serial tail.

**Fix:** Once `log_sum_exp` is in registers (from H1 for the dense kernel; already present at line 1073 for the sparse kernel), all threads can participate in the loss accumulation using the same stride-loop + shared-memory tree-reduce pattern already used for the max and sum phases:

```c
// All threads: accumulate stripe
float loss_partial = 0.0f;
for (int j = threadIdx.x; j < cols; j += BLOCK)
    loss_partial -= row_tgt[j] * (row_in[j] - log_sum_exp);
smem[threadIdx.x] = loss_partial;
__syncthreads();
// Tree-reduce smem → smem[0]
for (int s = BLOCK/2; s > 0; s >>= 1) {
    if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
    __syncthreads();
}
if (threadIdx.x == 0) atomicAdd(loss_acc, smem[0] / (float)rows);
```

Apply the same pattern to the sparse CE kernel. The sparse kernel already computes `log_sum_exp` in registers (line 1073) — no H1 prerequisite. `off` is already computed at the kernel's top as `cols > 1 ? smoothing / (float)(cols - 1) : 0.0f`. The per-column target expression is `target = (j == id) ? (1.0f - smoothing) : off`.

---

## T1 — `tg_tanh`, `tg_relu`, `tg_scale`: no dedicated tests

**File:** `tests/test_ops.c`

Add `test_tanh_grad`, `test_relu_grad`, `test_scale_grad`. Each should verify the forward value at a few known points and check the backward against central finite differences (h = 1e-3, tolerance 2e-3).

---

## T2 — `tg_softmax` backward on a non-last axis is untested

**File:** `tests/test_ops.c`

Add `test_softmax_axis0`: apply softmax along axis=0 on a `[3, 4]` tensor; verify column sums equal 1; verify gradient against central finite differences.

---

## T3 — `tg_dropout` has no gradient test in training mode

**File:** `tests/test_ops.c`

Add `test_dropout_grad`: set `tg_training=1`, apply `tg_dropout(a, 0.5f)`, run backward. Verify that positions where the mask is 0 have zero gradient in `a->grad`, and positions where the mask is `inv_keep` have gradient equal to upstream × `inv_keep`.

---

## T4 — LayerNorm backward not numerically verified

**File:** `tests/test_ops.c`

Add `test_layer_norm_backward_finite_diff`: for a `[2, 4]` input with non-trivial gamma/beta, compare analytical `dx`, `dgamma`, `dbeta` against central finite differences (h = 1e-3, tolerance 1e-3).

---

## T5 — `tg_slice` tested only on 2D inputs

**File:** `tests/test_ops.c`

Add `test_slice_3d`: slice a `[2, 4, 3]` tensor along axis=1 (start=1, len=2) to get `[2, 2, 3]`; verify the extracted values and that gradients flow back only to the sliced rows.

---

## Verification

```powershell
cmake --build --preset default
.\build\otto_von_grad_tests.exe
```

Expected outcome: all 61 existing tests pass, plus 7 new tests (T1 adds 3 functions, T2–T5 add 1 each) = **68 tests, 0 failed**.

Spot-checks after the build:
- `test_cross_entropy_value` — CPU loss value unchanged (H1 does not affect CPU path)
- `test_cuda_new_ops` — CUDA CE loss now matches CPU exactly for the same inputs
- `test_drop_path_inference_noop` and `test_drop_path_rate_schedule` — drop-path schedule and inference no-op still correct after M3 refactor
- H2: `grep -r "cuda_layer_norm_rows_fwd\|cuda_layer_norm_rows_bwd" src/` returns no matches — confirms dead launchers are gone
- M1: build succeeds and all tests pass; `CUDA_CHECK(cudaGetLastError())` is passive and fires on the next CUDA call after a bad launch, so no existing test will trigger it deliberately — verify by code inspection that every launch site in `cuda_ops.cu` is followed by a check
- M2: code inspection — confirm the overflow guard appears immediately after the `nel`-computation loop in `tg_new` and after the `batch` computation in `backward_matmul`

**Note:** After the T-series tests land, update the test-count line in `CLAUDE.md` from "48 tests" to "68 tests".
