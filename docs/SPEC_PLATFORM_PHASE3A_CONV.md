# Spec: Platform Phase 3a — Graph Capacity and the Conv Family

> **Status:** Draft 2026-09-20, not yet reviewed. Derived from `docs/FOUNDATION_PLATFORM.md` (Decision Log rows "Conv family scope" and "Graph and dimension limits", both closed 2026-09-20) and from the Phase 2 close-out (`docs/SPEC_PLATFORM_PHASE2_VISION.md`, library at `2740328`, tests 120 CUDA / 105 CPU). `AGENTS.md` describes what exists; this document describes exactly what Phase 3a adds, and is complete when every item in [Acceptance](#acceptance) is checked. Phase 3b (`ovg_diffusion`: UNet assembly, noise schedule, sampler, the first DDPM) is a separate spec that builds only on what this one ships.

---

## Open Questions

None blocking. One fork the foundation document stated as a fact rather than decided — activation layout — was put to the owner at spec time and closed (see [Decisions this spec makes](#decisions-this-spec-makes), first row).

Premises in the foundation document that the numbers contradict, recorded so the reviewer does not re-derive them:

- *"A UNet's node count … is not obviously under the cap."* It is. The 3b sketch below comes to **≈1,070 graph nodes**, about 7.5× under `TG_MAX_GRAPH = 8192`. The measured calibration point: vexilloscope's ViT (6 blocks, 192-dim, drop-path active, training) is **321 nodes**; a 6-block GPT is 281; one encoder block is 44 op nodes + 12 parameters in eval and 52 op nodes in training with drop-path active. Node count depends only on graph structure, never on `B`, `T`, `C`, or image size, so these numbers hold for the real models. The `topo_sort` change is still made — the foundation pre-committed it and the honest reasons stand (a recursive DFS whose depth is the model's depth, a 64 KB `Tensor *[TG_MAX_GRAPH]` array on the stack in three functions, a compile-time ceiling on model depth) — but the completion signal in the foundation's Phase 3 decision ("a UNet-shaped graph exceeds the old 8192-node cap without fataling") cannot be met by the 3b model and is restated here as a synthetic deep-graph test.
- *"`[B, C, H, W]` → `tg_reshape` → `[B, HW, C]`."* Not a reshape in NCHW — it is `reshape [B, C, HW]` then `transpose(1, 2)`, a full copy each way. In the NHWC layout this spec adopts, the flatten *is* a pure reshape.
- *"im2col memory: ~9× the input activation."* Correct for the buffer; the im2col output is a graph node, so it also carries a grad buffer of the same size that lives until `tg_free_graph`. The effective cost is ~18× per conv. The sizing in [The 3b Sketch](#the-3b-sketch-unet-op-set-and-node-count) uses the doubled figure.

---

## Purpose

Give the platform everything a small UNet-style image generator needs *below* the diffusion package, so that Phase 3b can be the model, the schedule, and the sampler and nothing else. Two halves:

1. **Graph capacity.** `topo_sort` becomes iterative with a heap-allocated, growable node list, so no model's depth is bounded by a compile-time constant or by the C stack. `TG_MAX_GRAPH` goes away.
2. **The conv family**, driven by a sketched 32×32 DDPM and nothing wider: `tg_silu`, `tg_im2col` (+ `tg_conv2d` composed over it and `tg_matmul`), `tg_upsample2d`, `tg_group_norm`, and `tg_sinusoidal_embed` in `ovg_core`; `TgConv2d`, `TgConvResBlock`, and `TgSpatialAttention` in `ovg_nn`. Every op has a CPU path, a CUDA path, and a paired backward; every block has a CPU/CUDA parity test.

**Completion signal:** all op and block tests pass on both presets; the four layers build standalone; a synthetic chain of 3× the old cap (24,576 nodes) runs forward, `tg_backward`, and `tg_free_graph` without fataling on either preset; a 3b-shaped UNet assembled from the new blocks in a test forwards, backwards, and frees at batch 2 on CUDA within the node count predicted by the sketch (±10 %); `../lambda` and `../vexilloscope` build unchanged.

---

## Decisions Inherited From the Foundation Document

These are settled and this spec does not reopen them:

1. **Driven by a first concrete model.** The op set is exactly what the sketched 32×32 RGB DDPM needs: `conv2d` (also strided, for downsampling), `group_norm`, `silu`, nearest `upsample2d`, sinusoidal timestep embedding. `conv_transpose2d`, `max_pool2d`, `avg_pool2d`, and `sigmoid` are not built.
2. **`conv2d` is im2col + cuBLAS GEMM**, riding on `tg_matmul`. A direct kernel is a permitted later replacement behind the same `tg_conv2d` signature; cuDNN is not a dependency and stays off the table.
3. **`group_norm` is its own op**, `tg_group_norm(a, gamma, beta, n_groups, eps)`, with per-channel affine. `tg_layer_norm` is untouched.
4. **`TG_MAX_DIMS` stays 4.** Activations are 4D; attention over spatial positions flattens to 3D first.
5. **`topo_sort` becomes iterative with a growable node list** as a prerequisite of this phase.
6. **`ovg_diffusion` is Phase 3b.** Nothing here assumes noise timesteps: the res block takes a generic conditioning vector, and the sinusoidal embedding is a function of integers.
7. **Every op ships with its paired backward, a CPU path, a CUDA path, and tests.** Batching must work from the start — no batch-1 designs.
8. **Performance posture** applies: eager dispatch, one kernel per op, correctness and readability first. im2col's memory cost is accepted and sized below.
9. Layer rules are enforced by grep plus per-target builds. `../lambda` and `../vexilloscope` must keep building with no edits to their CMakeLists.

### Decisions this spec makes

| Question | Decision |
|---|---|
| Activation layout | **NHWC — `[B, H, W, C]`.** Closed by the owner at spec time (2026-09-20). Feature-last is the library's existing convention (`[B, T, C]` sequences; `tg_layer_norm`, `TgLinear`, and `tg_matmul`'s broadcast all act on the last axis). In NHWC the conv composition needs no transpose (`im2col [B, H_out·W_out, k·k·C] @ W [k·k·C, C_out]` is already `[B, H_out·W_out, C_out]`, a reshape from the output), and flattening for attention (`[B, H, W, C] → [B, H·W, C]`) is a pure reshape. NCHW would have cost one full transpose per conv and one per attention block, each way. The foundation's constraint text is corrected in the same change. |
| Is `tg_conv2d` one op or a composition? | **A composition** over one new op. `tg_im2col` is the op (with `col2im` as its backward and its own CUDA kernels); `tg_conv2d` is a helper in `tg_ops.c` that chains `tg_im2col → tg_matmul → tg_reshape (→ bias expand + tg_add)`. The GEMM and its backward come from `tg_matmul`, the best-tested path in the library, including its batched cuBLAS dispatch. Composition does *not* save a kernel — im2col and col2im are needed either way — and the im2col buffer must survive to backward under either design (`dW` needs it), so memory is not the deciding factor. What decides it is that a single-op conv would re-implement `tg_matmul`'s same-ndim/lower-ndim dispatch and backward for no gain. Cost accepted: one extra copy of the output per conv (the reshape) plus 4–5 graph nodes for the bias. |
| Conv weight layout | **2D `[k·k·C_in, C_out]`**, column index `(kh·k + kw)·C_in + c`, with `k` passed explicitly. A 4D `[k, k, C_in, C_out]` weight would be more self-describing but would need a `tg_reshape` node (a weight-sized copy and a grad sum) on every forward. 2D matches `TgLinear` and `tg_fill_xavier_uniform` (which reads `fan_in = shape[0] = k·k·C_in` — the correct conv fan-in). The checkpoint validates `[k·k·C_in, C_out]`, so a `k` mismatch is caught at load by element count, not by name. |
| Bias in `tg_conv2d`? | **Optional by `NULL`.** `b == NULL` skips the expansion. A conv followed by `group_norm` gains nothing from a bias; the res block still carries biases (matching the reference DDPM) and the option lets 3b drop them later without an API change. This is the library's second `NULL`-means-absent parameter after `TgPatchEmbed.Cls`. |
| Padding and stride | General integer `stride ≥ 1` and `pad ≥ 0`, square `k×k` kernels, no dilation, no groups. `H_out = (H + 2·pad − k) / stride + 1` with floor division (PyTorch's rule); fatal if `H_out ≤ 0` or `W_out ≤ 0`. The sketch uses only `(k=3, s=1, p=1)`, `(k=3, s=2, p=1)`, and `(k=1, s=0, p=0)`; the general form costs nothing extra in the kernels. |
| Down / up blocks | **Recipes over `TgConv2d`, not structs.** Down is `TgConv2d(k=3, stride=2, pad=1)`; up is `tg_upsample2d(x, 2)` followed by `TgConv2d(k=3, 1, 1)`. A `TgConvDown` struct would hold exactly a `TgConv2d`; a `TgConvUp` would hold one plus an integer. The foundation's "down/up-sampling blocks" are satisfied by `TgConv2d` and the documented recipes. |
| Conditioning in the res block | **Generic `cond [B, cond_dim]`, optional at create time.** `cond_dim = 0` builds a block with no projection and `forward` then requires `cond == NULL`; `cond_dim > 0` requires `cond` non-`NULL`. The block projects `silu(cond) @ Wc + Bc` and adds it per channel. It does not know what `cond` means — 3b passes the time embedding; an autoencoder passes nothing. |
| `TG_MAX_GRAPH` | **Removed** from `tg_tensor.h`. Nothing in `tests/`, `examples/`, `../lambda`, or `../vexilloscope` references it (verified 2026-09-20). The growable list starts at 1,024 entries and doubles. |
| Iterative `topo_sort` order | **Bit-identical to the recursive post-order.** An explicit stack of `(node, next_child)` visits parents in index order and emits a node after its last parent, exactly as the recursion does. Gradient accumulation order is therefore unchanged and `test_checkpoint_exact_resume` stays meaningful across the change. |
| `tg_sinusoidal_embed` — op or factory? | **A factory for a host leaf.** It returns a fresh non-persistent `[n, dim]` tensor with no parents and no backward, like the drop-path mask in `tg_block_forward`. `tg_free_graph` frees it; a CUDA caller uploads it with `tg_to_cuda` before use (the block that consumes it does this, as `tg_block_forward` does for its mask). It lives in `ovg_core` because it is a function of integers into a tensor with no notion of a model. |
| `group_norm` backward for `gamma`/`beta` on CUDA | Per-channel reductions over `B·H·W` use `atomicAdd`, as `cuda_layer_norm_rows_affine_bwd` already does (`cuda_ops.cu:720-721`). Deterministic order is not promised on CUDA today and this does not change that. |
| BF16 | **F32 only** for every new op and block. `tg_conv2d` inherits `tg_matmul`'s BF16 path in principle, but `tg_im2col`, `tg_group_norm`, `tg_silu`, and `tg_upsample2d` are F32, and nothing in 3b calls for BF16 conv. Stated as a non-goal so it does not leak in through the matmul. |

---

## Definitions

- **NHWC.** Image activations are `[B, H, W, C]`: batch, rows, columns, channels; channels are the fastest-varying index. A pixel's channel vector is contiguous, like a token's feature vector.
- **im2col.** The unrolled-window matrix of a conv input: row `(b, y_out, x_out)`, column `(kh, kw, c)`, value `x[b, y_out·s − p + kh, x_out·s − p + kw, c]` or 0 when that index is outside the image. Shape `[B, H_out·W_out, k·k·C]`.
- **col2im.** The adjoint of im2col: scatter-add every column entry back to the input pixel it was read from. It is `tg_im2col`'s backward.
- **Group norm.** For each `(b, g)`, normalise the `H·W·(C/G)` values of channels `[g·C/G, (g+1)·C/G)` to zero mean and unit variance, then apply per-channel `gamma[c]`, `beta[c]`. `G = 1` is layer norm over `(H, W, C)`; `G = C` is instance norm.
- **Conditioning vector.** A `[B, cond_dim]` tensor the res block projects and adds per channel. In 3b it is the time embedding; the block does not care.
- **Sketch.** The 3b model as this spec sizes it. It fixes the op set and the node-count and memory estimates; 3b may change hyperparameters within the same op set without reopening this spec.

---

## The 3b Sketch: UNet Op Set and Node Count

Written first, as the foundation requires, so that 3a builds nothing 3b does not use. The model is the DDPM CIFAR-10 shape scaled down one notch:

| Item | Value |
|---|---|
| Input | `[B, 32, 32, 3]`, values in `[−1, 1]` |
| Base channels `C0` | 64 (DDPM used 128; see memory below) |
| Channel multipliers | `(1, 2, 2)` → 64 @ 32², 128 @ 16², 128 @ 8² |
| Res blocks per level | 2 in the encoder, 3 in the decoder (one per skip) |
| Attention | one level, 16×16 (256 tokens), 4 heads, in every res position at that level and in the middle |
| Time embedding | `tg_sinusoidal_embed(t, B, C0) → Linear(C0, 4·C0) → silu → Linear(4·C0, 4·C0)`; `cond_dim = 4·C0 = 256` |
| Group norm | 32 groups, eps 1e-5 |
| Dropout | 0.1 inside each res block |
| Down / up | strided 3×3 conv / nearest ×2 + 3×3 conv |
| Output | `group_norm → silu → 3×3 conv (C0 → 3)` |
| Loss | MSE between predicted and true noise: `tg_mean(tg_pow(tg_sub(pred, noise), 2))` |

Structure: `conv_in (3 → 64)`; encoder levels `[res, res, (attn), down]` ×3 (no down at the last); middle `res, attn, res`; decoder levels `[concat-skip, res, (attn)] ×3, up` ×3 (no up at the first); output head. Nine skips: after `conv_in`, after each encoder res block, after each down.

**Nodes per component**, under this spec's compositions (a *node* is any tensor `topo_sort` reaches: op outputs, parameters, and leaves):

| Component | Op nodes | Params |
|---|---|---|
| `tg_conv2d` with bias | 8 (im2col, matmul, reshape; bias reshape, 3 expands, add) | 2 |
| `tg_conv2d`, no bias | 3 | 1 |
| `tg_group_norm` / `tg_silu` / `tg_dropout` / `tg_upsample2d` / `tg_concat` / `tg_add` | 1 each | 2 for GN |
| Res block, `C_in == C_out` | 32 (GN, silu, conv 8; cond: silu, matmul, bias 3, reshape, 2 expands, add = 10; GN, silu, dropout, conv 8; add) | 10 |
| Res block, `C_in ≠ C_out` (1×1 skip conv) | 40 | 12 |
| Spatial attention block | 21 (GN, reshape, `TgSelfAttention` 17, reshape, add) | 6 |
| Down (strided conv) | 8 | 2 |
| Up (upsample + conv) | 9 | 2 |
| Time MLP | 10 (sinusoid leaf, matmul + bias 3, silu, matmul + bias 3) | 4 |

**Total for the sketch:** encoder 258, middle 85, decoder 450 (9 res blocks all with skip convs because of the concat, 3 attention blocks, 9 concats, 2 ups), `conv_in` 8, output head 10, time MLP 10, loss 3 → **≈ 824 op nodes**; parameters ≈ 244; leaves 3 (input, noise, sinusoid) → **≈ 1,070 nodes**, longest dependency chain ≈ 600. Against the 321-node ViT calibration (measured with a scratch consumer that walks `parents[]` the way `topo_sort` does), the estimate is credible to ±10 %. `TG_MAX_GRAPH = 8192` was never at risk from this model; the change is made for the reasons in [Open Questions](#open-questions).

**im2col memory**, per image, F32, summing every conv's `[H_out·W_out, k·k·C_in]` buffer in the sketch: level 0 (32², 1,024 rows) ≈ 10.7 M floats, level 1 (16²) ≈ 4.3 M, level 2 (8²) ≈ 1.3 M → **≈ 16 M floats ≈ 65 MB per image for the buffers, ≈ 130 MB with their grads.** Activations, GN caches, and attention scores add well under half that again. On a 12 GB card: **batch 32 fits with room (~5 GB), batch 64 is at the edge.** Doubling `C0` to 128 roughly doubles every column count and halves the batch. 3b should start at `C0 = 64`, batch 32, and treat `C0 = 128` as an experiment, not the default. This is why "no batch-1 designs" is a constraint: every op here takes `B` in its first axis and every kernel loops over it.

---

## Deliverables

### 1. Graph capacity — `src/tg_train.c`, `tg_tensor.h`

Remove `#define TG_MAX_GRAPH 8192` from `tg_tensor.h`. In `tg_train.c`:

```c
/* Module-static growable node list shared by tg_backward, tg_backward_accum,
   and tg_free_graph (none of which re-enter another). Grows by doubling from
   1024 entries; never shrinks; freed at exit by the OS. Not thread-safe, like
   the rest of the library. */
static Tensor **g_topo     = NULL;
static int      g_topo_cap = 0;

/* Iterative DFS with an explicit (node, next_child) stack. Emits the same
   post-order as the former recursive version: parents in index order, a node
   after its last parent. Returns the node count; the list is g_topo[0..n). */
static int topo_sort(Tensor *root);
static void clear_visited(int n);
```

The three stack arrays `Tensor *topo[TG_MAX_GRAPH]` at today's `tg_train.c:148` (`tg_free_graph`), `:164` (`tg_backward`), and `:211` (`tg_backward_accum`) all become `int n = topo_sort(root);` followed by loops over `g_topo`. **All three call sites change in the same commit**; an implementer who converts two leaves a silent inconsistency. The explicit stack is a second growable array (`(Tensor *, int)` pairs) with the same doubling rule, since its depth can equal `n`. The `visited` flag protocol is unchanged: set during the walk, cleared by `clear_visited` before any backward or free runs, so a fatal mid-walk cannot leave a graph half-marked without also ending the process.

No public signature changes. `tg_free_graph` inside a `backward_fn` is not supported today and remains unsupported.

### 2. Ops — `tg_ops.h` / `src/tg_ops.c` (`ovg_core`)

Every op below follows the existing shape: `make_op` (or the manual three-parent wiring `tg_layer_norm` uses at `tg_ops.c:806-810`, including its hand-written mixed-device check that `make_op` would otherwise perform), a `backward_*` defined immediately above its forward, a CPU loop, and a `#ifdef OVG_CUDA_ENABLED` dispatch to `cuda_ops.cu`. All are F32-only and fatal on BF16 input.

```c
/* Element-wise x * sigmoid(x).  Backward: sigmoid(x) * (1 + x * (1 - sigmoid(x))). */
Tensor *tg_silu(Tensor *a);

/* im2col over NHWC.  a: [B, H, W, C]; square k×k window, stride, zero padding pad.
   H_out = (H + 2*pad - k) / stride + 1 (floor), likewise W_out; fatal if either <= 0,
   if k <= 0, stride <= 0, pad < 0, or a is not 4D.
   Returns [B, H_out * W_out, k * k * C]; row = y_out * W_out + x_out,
   column = (kh * k + kw) * C + c.  k, stride, pad are stored as three floats in the
   host `cache` on both paths (backward reads them host-side before dispatching; the
   CUDA kernels take the geometry as arguments and need no device cache).
   Backward (col2im): scatter-add each column entry to the input element it came from;
   padding positions receive nothing. */
Tensor *tg_im2col(Tensor *a, int k, int stride, int pad);

/* Composition, not an op node.  a: [B, H, W, C_in]; W: [k*k*C_in, C_out]; b: [1, C_out] or NULL.
   cols = tg_im2col(a, k, stride, pad)                     [B, H_out*W_out, k*k*C_in]
   y    = tg_matmul(cols, W)                               [B, H_out*W_out, C_out]   (3D @ 2D broadcast)
   y    = tg_reshape(y, 4, {B, H_out, W_out, C_out})
   if b: y = tg_add(y, expand_dim(expand_dim(expand_dim(reshape(b, {1,1,1,C_out}), 0, B), 1, H_out), 2, W_out))
   Fatal if W->shape[0] != k*k*C_in, or b is not [1, C_out].  Upgrade path: a direct
   kernel replaces this body behind the same signature. */
Tensor *tg_conv2d(Tensor *a, Tensor *W, Tensor *b, int k, int stride, int pad);

/* Nearest-neighbour upsample by an integer factor >= 2 on NHWC:
   [B, H, W, C] -> [B, H*f, W*f, C]; out[b, y, x, c] = a[b, y/f, x/f, c].
   Backward: each input element receives the sum of its f×f block. */
Tensor *tg_upsample2d(Tensor *a, int factor);

/* Group norm over NHWC.  a: [B, H, W, C]; C % n_groups == 0; gamma, beta: [1, C]
   (exactly C elements, same rule as tg_layer_norm).  Each (b, g) is normalised over
   its H*W*(C/n_groups) values; affine is per channel.  Three parents (a, gamma, beta).
   cache: mean[B*G], inv_std[B*G], xhat[B*H*W*C] — the layer_norm layout with a
   per-(b,g) statistic instead of per-row.  Backward mirrors backward_layer_norm with
   the group as the reduction set. */
Tensor *tg_group_norm(Tensor *a, Tensor *gamma, Tensor *beta, int n_groups, float eps);

/* Sinusoidal embedding of integer steps.  Returns a fresh NON-PERSISTENT host leaf
   [n, dim] with no parents and no backward; dim must be even.
   out[i, j]         = sin(steps[i] * f_j)   for j <  dim/2
   out[i, j + dim/2] = cos(steps[i] * f_j)   for j <  dim/2
   f_j = exp(-ln(max_period) * j / (dim/2)).      (max_period = 10000 is the usual choice)
   tg_free_graph frees it with the rest of the graph.  A CUDA caller uploads it with
   tg_to_cuda before feeding it to an op on the device. */
Tensor *tg_sinusoidal_embed(const int *steps, int n, int dim, float max_period);
```

**CUDA kernels** added to `cuda_ops.h` / `cuda_ops.cu`, in the file's existing sections:

```c
void cuda_silu_fwd(const float *a, float *out, int n);
void cuda_silu_bwd(const float *a, const float *g, float *da, int n);

// NHWC im2col.  One thread per output element of cols.
void cuda_im2col_fwd(const float *a, float *cols, int B, int H, int W, int C,
                     int k, int stride, int pad, int H_out, int W_out);
// col2im as a GATHER: one thread per input element (b, y, x, c) sums the <= k*k
// column entries that read it.  No atomics, deterministic.
void cuda_col2im_bwd(const float *gcols, float *da, int B, int H, int W, int C,
                     int k, int stride, int pad, int H_out, int W_out);

void cuda_upsample2d_fwd(const float *a, float *out, int B, int H, int W, int C, int f);
void cuda_upsample2d_bwd(const float *g, float *da, int B, int H, int W, int C, int f);  // gather over f×f

// One block per (b, g); shared-memory reductions.  cache layout: mean[B*G], inv_std[B*G], xhat[B*H*W*C].
void cuda_group_norm_fwd(const float *a, const float *gamma, const float *beta, float eps,
                         float *out, float *cache, int B, int HW, int C, int G);
void cuda_group_norm_bwd(const float *xhat, const float *inv_std, const float *dy, const float *gamma,
                         float *dx, float *dgamma, float *dbeta, int B, int HW, int C, int G);
```

`col2im` as a gather rather than a scatter is deliberate: the scatter form needs `atomicAdd` on `da` and is non-deterministic; the gather form does the same arithmetic with a bounded inner loop over `(kh, kw)` and writes each `da` element once. The CPU path can use either; use the gather so the two agree in summation order.

### 3. Conv layers and blocks — `tg_conv.h` / `src/tg_conv.c` (`ovg_nn`)

```c
#include "tg_ops.h"

/* A conv2d layer: W [k*k*c_in, c_out] Xavier-uniform (fan_in = k*k*c_in), B [1, c_out] zeros;
   both persistent.  Fatal on non-positive dims, stride <= 0, pad < 0. */
typedef struct {
    Tensor *W, *B;
    int c_in, c_out, k, stride, pad;
} TgConv2d;

TgConv2d  tg_conv2d_create(int c_in, int c_out, int k, int stride, int pad);
void      tg_conv2d_free(TgConv2d *l);
Tensor   *tg_conv2d_forward(TgConv2d *l, Tensor *x);          /* [B, H, W, c_in] -> [B, H_out, W_out, c_out]; = tg_conv2d(x, W, B, k, stride, pad) */
int       tg_conv2d_collect_params(TgConv2d *l, Tensor **params, int max_params);   /* W, B -> 2 */

/* Pre-activation residual block (GN -> silu -> conv -> +cond -> GN -> silu -> dropout -> conv, + skip).
   c_in -> c_out; 3x3 convs, stride 1, pad 1; 1x1 skip conv when c_in != c_out, identity otherwise.
   cond_dim = 0 builds no projection (Wc, Bc NULL) and forward requires cond == NULL. */
typedef struct {
    Tensor *gamma1, *beta1;   /* GN before conv1: [1, c_in] */
    TgConv2d conv1;           /* c_in -> c_out */
    Tensor *Wc, *Bc;          /* cond projection [cond_dim, c_out], [1, c_out]; NULL when cond_dim == 0 */
    Tensor *gamma2, *beta2;   /* GN before conv2: [1, c_out] */
    TgConv2d conv2;           /* c_out -> c_out; W ZERO-initialised so the block starts as its skip path */
    TgConv2d skip;            /* 1x1, c_in -> c_out; W/B NULL when c_in == c_out */
    int   c_in, c_out, cond_dim, n_groups;
    float dropout;            /* 0 by default; applied after the second silu when tg_training */
} TgConvResBlock;

TgConvResBlock tg_conv_res_block_create(int c_in, int c_out, int cond_dim, int n_groups);
               /* gamma = 1, beta = 0; conv1 and skip via tg_conv2d_create (Xavier); Wc Xavier, Bc zeros;
                  conv2 via tg_conv2d_create and then conv2.W and conv2.B OVERWRITTEN WITH ZEROS.
                  Fatal if c_in % n_groups or c_out % n_groups != 0, or any dim <= 0, or cond_dim < 0. */
void           tg_conv_res_block_free(TgConvResBlock *b);
Tensor        *tg_conv_res_block_forward(TgConvResBlock *b, Tensor *x, Tensor *cond);
               /* x [B, H, W, c_in], cond [B, cond_dim] or NULL -> [B, H, W, c_out] */
int            tg_conv_res_block_collect_params(TgConvResBlock *b, Tensor **params, int max_params);
               /* order: gamma1, beta1, conv1.W, conv1.B, [Wc, Bc], gamma2, beta2, conv2.W, conv2.B, [skip.W, skip.B]
                  -> 8 + 2*(cond_dim > 0) + 2*(c_in != c_out).  This order is the checkpoint contract. */
```

Forward, in NHWC throughout:

```text
h = conv1(silu(group_norm(x, gamma1, beta1, G)))                      [B, H, W, c_out]
if cond:
  c = silu(cond) @ Wc + expand_dim(Bc, 0, B)                           [B, c_out]
  h = h + expand_dim(expand_dim(reshape(c, [B, 1, 1, c_out]), 1, H), 2, W)
h = silu(group_norm(h, gamma2, beta2, G))
h = dropout(h, p)  if p > 0                                            (pass-through when !tg_training)
h = conv2(h)
return h + (skip.W ? skip(x) : x)
```

`conv2.W` is zero-initialised (its bias too), the standard DDPM trick so a fresh block is the identity on its skip path and deep stacks train from step 0. `tg_fill_xavier_uniform` on `[k·k·c_in, c_out]` uses `fan_in = k·k·c_in`, which is the correct conv fan-in — no special init helper is needed.

Down and up are recipes, not structs:

```c
TgConv2d down = tg_conv2d_create(c, c, 3, /*stride*/2, /*pad*/1);   /* [B, H, W, c] -> [B, H/2, W/2, c]; H, W even */
TgConv2d up   = tg_conv2d_create(c, c, 3, 1, 1);
Tensor *y = tg_conv2d_forward(&up, tg_upsample2d(x, 2));            /* [B, H, W, c] -> [B, 2H, 2W, c] */
```

### 4. Spatial attention — `tg_spatial_attention.h` / `src/tg_spatial_attention.c` (`ovg_nn`)

```c
#include "tg_attention.h"

/* Self-attention over the H*W positions of an NHWC map, residual, GN pre-norm.
   Reuses TgSelfAttention (encoder / non-causal).  channels % n_heads == 0, channels % n_groups == 0. */
typedef struct {
    Tensor *gamma, *beta;     /* [1, channels] */
    TgSelfAttention attn;     /* embed_dim = channels, causal = 0 */
    int channels, n_groups;
} TgSpatialAttention;

TgSpatialAttention tg_spatial_attention_create(int channels, int n_heads, int n_groups);
void               tg_spatial_attention_free(TgSpatialAttention *s);
Tensor            *tg_spatial_attention_forward(TgSpatialAttention *s, Tensor *x);   /* [B, H, W, C] -> [B, H, W, C] */
int                tg_spatial_attention_collect_params(TgSpatialAttention *s, Tensor **params, int max_params);
                   /* gamma, beta, Wq, Wk, Wv, Wo -> 6 */
```

Forward: `h = group_norm(x)` → `tg_reshape(h, 3, {B, H·W, C})` → `tg_attention_forward(&attn, h)` → `tg_reshape(·, 4, {B, H, W, C})` → `tg_add(x, ·)`. Because the layout is NHWC, both reshapes are copies of contiguous memory with no permutation; the attention block sees `[B, T, C]` exactly as a transformer does. Attention scores are `[B, n_heads, H·W, H·W]` — at 16×16 that is `B·4·65,536` floats, the reason the sketch keeps attention to one level.

### 5. Build — `CMakeLists.txt`

```cmake
set(OVG_NN_SOURCES
    ...
    src/tg_conv.c
    src/tg_spatial_attention.c
)
add_executable(otto_von_grad_tests ... tests/test_conv.c)
```

No new target. `ovg_core` gains only ops (in `tg_ops.c` and `cuda_ops.cu`); `ovg_nn` gains two files; `ovg_lm` and `ovg_vision` are untouched.

**Layer check** (grep + per-target build, as today):

- `cmake --build --preset default --target ovg_core`, then `ovg_nn`, `ovg_lm`, `ovg_vision` — each succeeds on its own.
- `grep -l "tg_gpt.h\|tg_tokenizer.h\|tg_sample.h\|tg_patch_embed.h\|tg_pool.h" src/tg_conv.c src/tg_spatial_attention.c` → nothing (nn reaches neither modality package).
- `grep -l "tg_conv.h\|tg_spatial_attention.h" src/tg_*.c` → only the two files themselves (nothing in core, lm, or vision includes them).
- `grep -rn "TG_MAX_GRAPH" src include tests examples ../lambda/src ../vexilloscope/src` → nothing.

---

## The UNet Recipe, As Written Against This Phase

For reference and for the 3b spec. This is the sketch as code; 3b wraps it in a struct, a schedule, and a sampler.

```c
/* channels per level: ch[0] = C0, ch[l] = C0 * mult[l]; cond_dim = 4 * C0 */
TgConv2d           conv_in = tg_conv2d_create(3, ch[0], 3, 1, 1);
TgConvResBlock     enc[L][2], mid[2], dec[L][3];        /* c_in of dec blocks = ch[l] + skip channels */
TgSpatialAttention enc_attn[2], mid_attn, dec_attn[3];  /* at the attention level only */
TgConv2d           down[L-1], up[L-1];
Tensor            *out_gamma, *out_beta;  TgConv2d conv_out = tg_conv2d_create(ch[0], 3, 3, 1, 1);
Tensor            *Wt1, *Bt1, *Wt2, *Bt2;                 /* time MLP */

Tensor *temb = tg_sinusoidal_embed(t, B, C0, 10000.0f);  if (on_cuda) tg_to_cuda(temb);
temb = silu(temb @ Wt1 + Bt1) @ Wt2 + Bt2;                /* [B, 4*C0] */

Tensor *h = tg_conv2d_forward(&conv_in, x);  skips[n_skip++] = h;
for l in levels:
    for r in 0..1: h = res(enc[l][r], h, temb); if l == attn_level: h = attn(h); skips[n_skip++] = h;
    if l < L-1:    h = tg_conv2d_forward(&down[l], h);    skips[n_skip++] = h;
h = res(mid[0], h, temb); h = attn(mid_attn, h); h = res(mid[1], h, temb);
for l in reversed(levels):
    for r in 0..2: h = res(dec[l][r], tg_concat(h, skips[--n_skip], 3), temb); if l == attn_level: h = attn(h);
    if l > 0:      h = tg_conv2d_forward(&up[l-1], tg_upsample2d(h, 2));
Tensor *pred = tg_conv2d_forward(&conv_out, tg_silu(tg_group_norm(h, out_gamma, out_beta, 32, 1e-5f)));
Tensor *loss = tg_mean(tg_pow(tg_sub(pred, noise), 2.0f));
```

Skip concat is along axis 3 (channels) — `tg_concat` already supports any axis. The decoder block's `c_in` is the sum, which is why every decoder res block has a 1×1 skip conv in the node count.

---

## Consumer Migrations

None. No consumer uses anything this phase changes; `TG_MAX_GRAPH` is referenced by neither. `../lambda` and `../vexilloscope` must configure and build with no edits, and `examples/candide.c` is untouched. The first consumer of the conv family is `ovg_diffusion` in 3b.

---

## Tests

New file `tests/test_conv.c` with `run_conv_tests`, called in `test_main.c` after `=== linear ===` (section `=== conv ===`); additions to `tests/test_ops.c` and `tests/test_train.c`. Tolerances are absolute on floats of order 1 unless stated. Hand-loop references are written per test, not shared, so a bug in a helper cannot hide a bug in an op.

### `test_ops.c` additions

| Test | Checks |
|---|---|
| `test_silu` | `[2, 3]` with values `{−2, −1, 0, 0.5, 1, 3}`; forward within 1e-6 of `x/(1+e^−x)`; `tg_backward(tg_sum)`: grad within 1e-6 of `σ(x)(1 + x(1 − σ(x)))`; the gradient at 0 is exactly 0.5. |
| `test_im2col_identity_1x1` | `[2, 3, 3, 4]` ramp; `tg_im2col(a, 1, 1, 0)` is `[2, 9, 4]` and equals `a` reshaped, bitwise; backward of `tg_sum` gives `a->grad == 1` everywhere. |
| `test_im2col_3x3_pad1` | `[1, 4, 4, 2]` ramp; `tg_im2col(a, 3, 1, 1)` is `[1, 16, 18]`; every entry equals the hand-indexed value (`0` where the window leaves the image) — checked exhaustively; backward of `tg_sum`: `a->grad[y, x, c]` equals the number of 3×3 windows containing `(y, x)` (4 at corners, 6 on edges, 9 inside). |
| `test_im2col_stride2` | `[1, 5, 5, 1]`, `k=3, s=2, p=1` → `H_out = 3`; values hand-checked at all 9 rows; `[1, 4, 4, 1]`, `k=3, s=2, p=0` → `H_out = 1` (floor); `k=5, s=1, p=0` on `[1, 4, 4, 1]` → fatal. |
| `test_conv2d_vs_direct` | `[2, 5, 5, 3]` random, `W [27, 4]` random, `b [1, 4]` random; `tg_conv2d(a, W, b, 3, 1, 1)` is `[2, 5, 5, 4]` and matches a direct seven-nested-loop conv within 1e-5; `tg_backward(tg_sum)`: `W->grad`, `b->grad` (`= 2·25` each), and `a->grad` match the direct loop's analytic gradients within 1e-4. |
| `test_conv2d_no_bias` | Same with `b = NULL`; output equals the biased output minus `b`, within 1e-6; graph has no bias nodes (`tg_free_graph` then a second forward succeeds). |
| `test_conv2d_bad_shapes_fatal` | `W [26, 4]` (wrong `k·k·C_in`), `b [4, 1]`, a 3D input → fatal each. |
| `test_upsample2d` | `[1, 2, 2, 3]` ramp, factor 2 → `[1, 4, 4, 3]`, each 2×2 block equals its source pixel; factor 3 → `[1, 6, 6, 3]`; backward of `tg_sum`: `a->grad == f²` everywhere; factor 1 → fatal. |
| `test_concat_axis3_4d` | Two `[2, 3, 3, 4]` ramps concatenated along axis 3 → `[2, 3, 3, 8]`; every value hand-checked (`inner = 1` in the `outer/axis/inner` decomposition — the case Phase 2's CLS-prepend test at axis 1 on 3D does not reach); backward of `tg_sum` gives grad 1 in both parents; a channel mismatch on axis 2 → fatal. The decoder's skip concat is this call. |
| `test_group_norm_vs_layer_norm` | `[2, 3, 3, 8]` random; `tg_group_norm(a, γ, β, 1, eps)` equals `tg_layer_norm(tg_reshape(a, {2, 72}), γ₇₂, β₇₂, eps)` reshaped back, within 1e-5, when `γ`/`β` are constant across channels (both reduce over the same 72 values). |
| `test_group_norm_stats` | `G = 4` on `[2, 3, 3, 8]`; for every `(b, g)`, the 18 output values (with `γ = 1, β = 0`) have mean within 1e-5 of 0 and variance within 1e-4 of 1; with `γ[c] = c, β[c] = 10c` the output is `c·x̂ + 10c`. |
| `test_group_norm_backward` | Finite-difference check: `[1, 2, 2, 4]`, `G = 2`, loss `tg_sum(tg_mul(out, R))` for a random `R`; every `a`, `γ`, `β` grad within 1e-3 of a central difference with `h = 1e-3` (the same method `test_ops.c` uses for layer norm). |
| `test_group_norm_bad_args_fatal` | `C % G != 0`; `γ [1, 7]`; 3D input → fatal each. |
| `test_sinusoidal_embed` | `steps = {0, 1, 1000}`, `dim = 8`, `max_period = 10000`; row 0 is `{0,0,0,0, 1,1,1,1}` exactly; `out[1][0] = sin(1)`, `out[1][4] = cos(1)`, `out[2][3] = sin(1000·10000^(−3/4))` within 1e-5; the tensor is non-persistent with `n_parents == 0` and `backward_fn == NULL`; odd `dim` → fatal; `tg_free_graph` on it does not crash. |
| `test_conv_ops_cuda_parity` *(CUDA)* | `tg_silu`, `tg_im2col` (3×3 p1 and 3×3 s2 p1), `tg_conv2d` (with and without bias), `tg_upsample2d`, `tg_group_norm`, `tg_concat` on axis 3 of 4D, on device copies of the same inputs as above; forward within 1e-5 and every grad within 1e-4 of the host run. One test, seven sub-checks, each reported by name on failure. |

### `test_train.c` additions

| Test | Checks |
|---|---|
| `test_deep_chain_graph` | `x [2, 2]` persistent; 24,576 chained `tg_scale(·, 1.0f)` nodes (3× the removed cap); `tg_backward(tg_sum)` runs, `x->grad == 1` everywhere; `tg_free_graph` runs; a second chain of 100 nodes then works (the list was retained, not corrupted). CPU only — the point is the list and the stack, not the kernels. |
| `test_topo_order_unchanged` | A small DAG with shared parents (`d = add(mul(a, b), mul(b, c))`); `tg_backward` gives the analytic grads exactly, and `tg_free_graph` frees every intermediate exactly once (no double free under a debug allocator; checked by building it twice). This is the regression guard for the iterative rewrite emitting a valid post-order. |

### `test_conv.c`

Fixture: `B = 2, H = W = 4, c_in = 4, c_out = 8, n_groups = 2, cond_dim = 6`; random inputs from `tg_seed(42)` (set by `test_main`).

| Test | Checks |
|---|---|
| `test_conv2d_layer` | `tg_conv2d_create(4, 8, 3, 1, 1)`: `W [36, 8]`, `B [1, 8]` zeros, both persistent; forward `[2, 4, 4, 4] → [2, 4, 4, 8]` equals `tg_conv2d(x, W, B, 3, 1, 1)` bitwise; `collect_params` → 2; stride-2 layer gives `[2, 2, 2, 8]`. |
| `test_res_block_identity_at_init` | `create(4, 4, 0, 2)` (no cond, no skip conv): because `conv2.W == 0` and `conv2.B == 0`, `forward(x, NULL)` equals `x` bitwise; `create(4, 8, 0, 2)` equals `skip(x)` bitwise. |
| `test_res_block_forward_vs_manual` | `create(4, 8, 6, 2)` with `conv2.W` refilled random; forward equals the same composition written out by hand with the public ops (`tg_group_norm`, `tg_silu`, `tg_conv2d`, …) within 1e-6; `forward(x, NULL)` with `cond_dim = 6` → fatal; `forward(x, cond)` with `cond_dim = 0` → fatal. |
| `test_res_block_backward_and_free` | Two backward passes on `create(4, 8, 6, 2)`. **At init** (`conv2.W == 0`): `tg_backward(tg_sum(forward(x, cond)))` leaves exactly `conv2.W`, `conv2.B`, `skip.W`, `skip.B` with non-zero grads and the other eight parameters with grads that are exactly 0 — the zero-init property, pinned. **After refilling `conv2.W` random** (as the forward test does): every one of the 12 parameters has a finite, non-zero grad, `Bc->grad == 2·16` (`B·H·W`) exactly, and `conv2.B->grad == 2·16` exactly; `tg_free_graph` then a second forward succeeds; the block's parameters and `x`/`cond` (persistent) survive. |
| `test_res_block_collect_params` | Order and count for the four `(cond, skip)` combinations: 8, 10, 10, 12; identity of every pointer; `max_params` too small → fatal. |
| `test_res_block_dropout_eval` | `dropout = 0.5`: two forwards in training differ; under `tg_eval_begin`, forward equals the no-dropout block bitwise. |
| `test_spatial_attention` | `create(8, 2, 2)` on `[2, 4, 4, 8]`: output shape `[2, 4, 4, 8]`; equals `x + reshape(attn(reshape(gn(x), [2, 16, 8])), [2, 4, 4, 8])` computed by hand with the public ops within 1e-6; batch row 1 equals the batch-1 run on `x[1]` within 1e-5; `collect_params` → 6; `channels % n_heads != 0` → fatal. |
| `test_unet_sketch_end_to_end` | The recipe above at `C0 = 8, mult = (1, 2, 2), 8×8 input, attention at the 4×4 level, 2 heads, 4 groups, B = 2`: forward gives `[2, 8, 8, 3]`; `tg_backward` of the MSE loss leaves every parameter with a finite grad; **the node count (walked from the loss via `parents[]`) is within ±10 % of the sketch's per-component table evaluated at this config**, printed on failure; `tg_free_graph` then a second step succeeds. This is the acceptance check for the sketch arithmetic. |
| `test_conv_blocks_cuda_parity` *(CUDA)* | Res block (with cond and skip) and spatial attention on device copies; forward within 1e-5 and every parameter grad within 1e-4 of the host run; `tg_sinusoidal_embed` uploaded with `tg_to_cuda` feeds a device matmul without fataling. |
| `test_unet_sketch_cuda` *(CUDA)* | The end-to-end sketch on the device at the same config; loss within 1e-4 of the host run after one forward; backward and free succeed. |

**Expected count:** `test_ops.c` +15 (1 CUDA-guarded) + `test_train.c` +2 + `test_conv.c` +10 (2 CUDA-guarded) → **120 + 27 = 147 (CUDA build) / 105 + 24 = 129 (CPU build)**. The implementation sets the final number; `AGENTS.md` is updated to match.

---

## Documentation Updates (same change)

`AGENTS.md` (this repo):

- Opening paragraph: `ovg_nn` gains "conv layer, conv residual block, spatial attention"; `ovg_core` gains "conv-family ops (NHWC im2col/conv2d, group norm, silu, upsample, sinusoidal embedding)".
- Repository Structure: `tg_conv.h`, `tg_spatial_attention.h` under `[nn]`; `tg_conv.c`, `tg_spatial_attention.c` under `[nn → ovg_nn]`; `tests/test_conv.c`; `tg_train.c` line gains "iterative topo sort, growable node list".
- Tensor Struct: remove `TG_MAX_GRAPH = 8192` from the constants comment; add "graph size is unbounded (heap-allocated node list)".
- Tensor Autograd Ops: new subsection **Conv family (NHWC)** listing `tg_silu`, `tg_im2col`, `tg_conv2d` (composition, bias optional), `tg_upsample2d`, `tg_group_norm`, `tg_sinusoidal_embed` (factory; host leaf), with the layout sentence and the `H_out` formula.
- New section **Conv Blocks** (`tg_conv.h`, `tg_spatial_attention.h`): the three structs, forward compositions, collect orders, the down/up recipes, the zero-init of `conv2`.
- Build Commands / Verification: new counts.
- Known Constraints: replace the `topo_sort` bullet ("recursive … if `TG_MAX_GRAPH` needs raising …") with "graph capacity is dynamic; the node list and DFS stack grow by doubling". Add "**Conv ops are NHWC and F32.** `[B, H, W, C]` only; BF16 conv is not supported."
- Important Guidance: "Image tensors are NHWC. Flatten to `[B, H·W, C]` with `tg_reshape` for attention; never transpose to channels-first." and "`tg_sinusoidal_embed` returns a non-persistent host leaf: upload it yourself on CUDA, and do not free it by hand."

`docs/FOUNDATION_PLATFORM.md` — at spec time (this commit), because they are decisions and measurements, not implementation state:

- Conv Family → Constraints: the `[B, C, H, W]` bullet becomes NHWC `[B, H, W, C]`, with the flatten as a pure reshape; the im2col bullet notes the grad-buffer doubling.
- Conv Family → Follow-ups: the sketch item is closed, pointing at this spec's [sketch](#the-3b-sketch-unet-op-set-and-node-count) (≈1,070 nodes, ≈130 MB/image im2col with grads, batch 32 at `C0 = 64`).
- Conv Family → Decision: the 3a completion signal's last clause becomes "a synthetic graph of 3× the old cap runs without fataling; the sketch's node count is confirmed by a test"; a sentence records the NHWC decision and the composition decision.
- Graph and Dimension Limits → Follow-ups: closed with the measurements (ViT 321, GPT 281, UNet sketch ≈1,070).
- Brownfield Baseline: nothing until implementation; then a **Phase 3a** line with the counts.

On completion: `AGENTS.md` as above; the foundation's Brownfield Baseline Phase 3a line; this document's Status line and Acceptance boxes.

---

## Non-Goals

- `ovg_diffusion`, a `TgUNet` struct, the noise schedule, the sampler, any training or sampling of an actual DDPM (Phase 3b). The end-to-end sketch test builds the graph; it does not train it.
- `conv_transpose2d`, `max_pool2d`, `avg_pool2d`, `sigmoid`, dilation, grouped or depthwise conv, non-square kernels, asymmetric padding.
- A direct (non-im2col) conv kernel; cuDNN.
- BF16 for any new op or block.
- NCHW, or any layout conversion helper.
- A convolutional patch embedding for `ovg_vision` (it would be `TgConv2d` with `k = stride = patch`; a vision phase can add it).
- `tg_transformer_collect_params` or any collect helper above the block level; 3b assembles its own parameter list as `TgGPT` and vexilloscope do.
- Consumer changes of any kind.
- Namespaced includes (Phase 4).

---

## Risks and Notes for the Implementer

- **Three call sites, one list.** `tg_free_graph`, `tg_backward`, `tg_backward_accum` each own a `Tensor *topo[TG_MAX_GRAPH]` today. Convert all three to the shared growable list in one commit, and delete the macro in the same commit so a missed site fails to compile rather than silently keeping a bound.
- **Post-order must not change.** The iterative DFS emits a node after its *last* parent, visiting parents in index order — the recursive version's order exactly. A stack-of-nodes rewrite that emits on push (pre-order reversed) is also a valid topological order but changes float accumulation order; `test_topo_order_unchanged` catches the invalid orders, not the reordered valid ones, so keep to the specified traversal.
- **`make_op` wires two parents.** `tg_group_norm` has three; copy the manual wiring at `tg_ops.c:806-810` and, because `make_op` is bypassed, its mixed-device check too. `tg_conv2d` is a composition, so it needs neither.
- **im2col output is a real node with a real grad.** `tg_new` allocates `grad` for every tensor; the `[B, H_out·W_out, k·k·C]` buffer therefore costs 2× its size until `tg_free_graph`. The sketch's memory figure includes this. Do not try to make the im2col node grad-less; `backward_matmul` writes into it.
- **col2im as a gather.** One thread (or one CPU loop body) per *input* element, iterating the ≤ `k·k` `(kh, kw)` pairs that map to a valid `(y_out, x_out)`: `y_out = (y + pad − kh) / stride` when `(y + pad − kh) % stride == 0` and `0 ≤ y_out < H_out`. No atomics, deterministic, and the CPU and CUDA paths sum in the same order.
- **Group-norm cache on CUDA** has three parts (`mean`, `inv_std`, `xhat`); allocate `2·B·G + B·H·W·C` floats with `tg_cuda_alloc_cache`, mirroring `tg_layer_norm`'s `rows + rows·C`. The backward needs `inv_std` and `xhat`; `mean` is kept for symmetry with a future fused kernel and may be dropped if the implementer prefers — say which in the code comment.
- **Zero-init of `conv2` is a create-time fact, not a forward-time one.** Tests that need a non-trivial `conv2` refill it. Document in the header that a warm-started checkpoint overwrites it anyway.
- **`tg_sinusoidal_embed` on CUDA.** The leaf is created on the host; any block that consumes it on a device graph must `tg_to_cuda` it first, or `make_op` fatals with "mixed CUDA/CPU parents". In the recipe this happens once per step, right after creation. `tg_to_cuda` on a non-persistent tensor is fine — `tg_free_graph` calls `tg_cuda_free` before `tg_free`.
- **Attention memory at 16×16.** Scores are `[B, heads, 256, 256]` per attention block, twice (scaled and softmax'd) plus their grads: at `B = 32`, 4 heads, six blocks that is ~800 MB. The sketch keeps it to one level for this reason; 3b should not add attention at 32×32 without re-sizing.
- **Node-count test tolerance.** The `±10 %` in `test_unet_sketch_end_to_end` guards the *sketch arithmetic*, not the implementation: if the composition changes (say, bias becomes a single fused node), update the table in this document, not the tolerance.
- **MSVC stack.** With the recursion gone, the only deep recursion left in the library is none; the deep-chain test would have overflowed a 1 MB stack at ~50k frames of the old `topo_sort`, so it also guards against the recursion coming back.

---

## Acceptance

Library (this repo):

- [ ] `TG_MAX_GRAPH` is gone from `tg_tensor.h`; `topo_sort` is iterative over a growable heap list used by all three of `tg_backward`, `tg_backward_accum`, `tg_free_graph`; `test_deep_chain_graph` passes on both presets.
- [ ] `tg_silu`, `tg_im2col`, `tg_conv2d`, `tg_upsample2d`, `tg_group_norm`, `tg_sinusoidal_embed` exist in `tg_ops.h` with the specified contracts; each op has its `backward_*` adjacent, a CPU path, and a CUDA path; the CUDA kernels listed in [Ops](#2-ops--tg_opsh--srctg_opsc-ovg_core) exist in `cuda_ops.h` / `.cu`.
- [ ] `tg_conv.h` / `tg_conv.c` (`TgConv2d`, `TgConvResBlock`) and `tg_spatial_attention.h` / `.c` (`TgSpatialAttention`) exist in `ovg_nn` with the specified APIs, forward compositions, and collect orders; `conv2` is zero-initialised.
- [ ] `cmake --preset default && cmake --build --preset default` clean; `otto_von_grad_tests.exe` reports all pass at the new count (expected 147); `cmake --preset cpu && cmake --build --preset cpu` passes its count (expected 129).
- [ ] `ovg_core`, `ovg_nn`, `ovg_lm`, `ovg_vision` each build standalone via `--target`; the layer greps in [Build](#5-build--cmakeliststxt) return nothing unexpected.
- [ ] `test_unet_sketch_end_to_end` passes with the measured node count recorded in this document's sketch section (replace "≈" with the number).
- [ ] `examples/candide.c` untouched and builds; `../lambda` and `../vexilloscope` configure and build with no edits to their CMakeLists.

Documentation:

- [ ] `AGENTS.md` updated as listed.
- [ ] `docs/FOUNDATION_PLATFORM.md`: the spec-time edits (constraints, follow-ups, decision text) landed with this spec; the Brownfield Baseline Phase 3a line lands with the implementation.
- [ ] This document's Status line reads Implemented, with the library commit and final test counts.
