#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#ifdef OVG_CUDA_ENABLED

#ifdef __cplusplus
extern "C" {
#endif

// ── Matmul ───────────────────────────────────────────────────────────────────
//
// Forward:   C  = A @ B        A:[m×K]  B:[K×n]  C:[m×n]
// Backward:  dA += dC @ B^T   dA:[m×K]  (atomicAdd — safe for shared weights)
//            dB += A^T @ dC   dB:[K×n]  (atomicAdd)

void cuda_matmul_forward (const float *A, const float *B, float *C,
                          int m, int K, int n);
void cuda_matmul_bwd_dA  (const float *dC, const float *B, float *dA,
                          int m, int K, int n);
void cuda_matmul_bwd_dB  (const float *A, const float *dC, float *dB,
                          int m, int K, int n);

// ── Element-wise binary ───────────────────────────────────────────────────────

void cuda_add_fwd(const float *a, const float *b, float *out, int n);
void cuda_add_bwd(const float *g, float *da, float *db, int n);

void cuda_sub_fwd(const float *a, const float *b, float *out, int n);
void cuda_sub_bwd(const float *g, float *da, float *db, int n);

void cuda_mul_fwd(const float *a, const float *b, float *out, int n);
void cuda_mul_bwd(const float *a, const float *b, const float *g,
                  float *da, float *db, int n);

// ── Element-wise unary ────────────────────────────────────────────────────────

void cuda_scale_fwd(const float *a, float s, float *out, int n);
void cuda_scale_bwd(const float *g, float s, float *da, int n);

void cuda_pow_fwd(const float *a, float p, float *out, int n);
void cuda_pow_bwd(const float *a, const float *g, float p, float *da, int n);

void cuda_relu_fwd(const float *a, float *out, int n);
void cuda_relu_bwd(const float *a, const float *g, float *da, int n);

void cuda_tanh_fwd(const float *a, float *out, int n);
void cuda_tanh_bwd(const float *out_data, const float *g, float *da, int n);

void cuda_gelu_fwd(const float *a, float *out, int n);
void cuda_gelu_bwd(const float *a, const float *g, float *da, int n);

// ── Transpose ────────────────────────────────────────────────────────────────

void cuda_transpose_fwd(const float *a, float *out, int rows, int cols);
// da += transpose(g); accumulates into da (does not overwrite)
void cuda_transpose_bwd(const float *g, float *da, int out_rows, int out_cols);

// ── Reductions ────────────────────────────────────────────────────────────────
// out must be zero-initialised before cuda_sum_fwd (use tg_cuda_alloc).

void cuda_sum_fwd (const float *a, float *out, int n);
void cuda_sum_bwd (const float *g, float *da, int n);           // broadcast scalar g[0]
void cuda_mean_bwd(const float *g, float *da, int n);           // broadcast g[0]/n

// mean_rows: [R×C] → [1×C]  (out must be zero-initialised)
void cuda_mean_rows_fwd(const float *a, float *out, int rows, int cols);
void cuda_mean_rows_bwd(const float *g, float *da, int rows, int cols);

// ── Row-wise ops ──────────────────────────────────────────────────────────────
// One CUDA block per row; shared-memory intra-block reductions.

void cuda_softmax_rows_fwd(const float *a, float *out, int rows, int cols);
void cuda_softmax_rows_bwd(const float *out_data, const float *g, float *da,
                           int rows, int cols);

// inv_std: R-float device buffer (output of fwd, input of bwd)
void cuda_layer_norm_rows_fwd(const float *a, float eps, float *out,
                               float *inv_std, int rows, int cols);
void cuda_layer_norm_rows_bwd(const float *y, const float *dy,
                               const float *inv_std, float *dx,
                               int rows, int cols);
// cache layout: inv_std[rows], xhat[rows*cols]
void cuda_layer_norm_rows_affine_fwd(const float *a, const float *gamma, const float *beta,
                                      float eps, float *out, float *cache,
                                      int rows, int cols);
void cuda_layer_norm_rows_affine_bwd(const float *xhat, const float *dy,
                                      const float *gamma, const float *inv_std,
                                      float *dx, float *dgamma, float *dbeta,
                                      int rows, int cols);

// ── Special ops ───────────────────────────────────────────────────────────────

// causal mask: [T×T] — writes -1e9 to future (j>i) positions
void cuda_causal_mask_fwd(const float *a, float *out, int seq_len);
void cuda_causal_mask_bwd(const float *g, float *da, int seq_len);

// slice_cols: a[rows × a_cols] → out[rows × out_cols] starting at col_start
void cuda_slice_cols_fwd(const float *a, float *out,
                         int rows, int a_cols, int col_start, int out_cols);
void cuda_slice_cols_bwd(const float *g, float *da,
                         int rows, int a_cols, int col_start, int out_cols);

// concat_cols (one part at a time): copy src[rows × src_cols] into
//   dst[rows × total_cols] at column offset col_offset.
void cuda_concat_cols_fwd(const float *src, float *dst,
                          int rows, int src_cols, int total_cols, int col_offset);
void cuda_concat_cols_bwd(const float *g, float *da,
                          int rows, int src_cols, int total_cols, int col_offset);

// slice_rows: a[a_rows × cols] → out[out_rows × cols] starting at row_start
void cuda_slice_rows_fwd(const float *a, float *out,
                         int a_rows, int cols, int row_start, int out_rows);
void cuda_slice_rows_bwd(const float *g, float *da,
                         int a_rows, int cols, int row_start, int out_rows);

// concat_rows (one part at a time): copy src[src_rows × cols] into
//   dst[total_rows × cols] at row offset row_offset.
void cuda_concat_rows_fwd(const float *src, float *dst,
                          int src_rows, int cols, int total_rows, int row_offset);
void cuda_concat_rows_bwd(const float *g, float *da,
                          int src_rows, int cols, int total_rows, int row_offset);

// repeat_rows: a[1 × cols] → out[n_rows × cols]
// Backward sums output grad rows into da[1 × cols] via atomicAdd.
void cuda_repeat_rows_fwd(const float *a, float *out, int n_rows, int cols);
void cuda_repeat_rows_bwd(const float *g, float *da, int n_rows, int cols);

// repeat_cols: a[rows × 1] → out[rows × n_cols]
// Backward sums output grad cols into da[rows × 1] via atomicAdd.
void cuda_repeat_cols_fwd(const float *a, float *out, int rows, int n_cols);
void cuda_repeat_cols_bwd(const float *g, float *da, int rows, int n_cols);

// cross_entropy: logits[R×C] + targets[R×C] → probs_cache[R×C], loss_out[1]
// loss_out must be zero-initialised.  After call, copy loss_out to host.
void cuda_cross_entropy_fwd(const float *logits, const float *targets,
                             float *probs_cache, float *loss_out,
                             int rows, int cols);
// backward: d_logits[i] += g * (probs[i] - targets[i]) / rows
void cuda_cross_entropy_bwd(const float *probs, const float *targets,
                             const float *g, float *d_logits,
                             int rows, int cols);
// sparse cache layout: probs[rows*cols], ids[rows]
void cuda_cross_entropy_sparse_fwd(const float *logits, const float *ids,
                                   float smoothing, float *probs, float *loss_out,
                                   int rows, int cols);
void cuda_cross_entropy_sparse_bwd(const float *probs, const float *ids,
                                   const float *g, float smoothing, float *d_logits,
                                   int rows, int cols);

// embed: weight[V×C], ids[T] (float-encoded ints in cuda_cache) → out[T×C]
// Backward uses atomicAdd — safe when ids contains duplicate token indices.
void cuda_embed_fwd(const float *weight, const float *ids, float *out, int T, int C);
void cuda_embed_bwd(const float *g, const float *ids, float *dw, int T, int C);

// dropout: apply pre-computed mask (generated on CPU, uploaded to cuda_cache)
void cuda_dropout_fwd(const float *a, const float *mask, float *out, int n);
void cuda_dropout_bwd(const float *mask, const float *g, float *da, int n);

// ── BF16 conversion ──────────────────────────────────────────────────────────
// void* pointers carry __nv_bfloat16 data (2 bytes/element).

void cuda_cast_f32_to_bf16(const float *in, void *out_bf16, int n);
void cuda_cast_bf16_to_f32(const void *in_bf16, float *out, int n);

// BF16 batched GEMM: A, B are BF16 (void*), C is F32.
// Same op_A/op_B convention as cuda_batched_sgemm.
void cuda_batched_sgemm_bf16(char op_A, char op_B,
                              const void *A, const void *B, float *C,
                              int batch, int M, int K, int N,
                              long strideA, long strideB, long strideC,
                              float alpha, float beta);

// ── Batched matmul (same-ndim path, cuBLAS SgemmStridedBatched) ──────────────
// C[b] = alpha * op(A[b]) @ op(B[b]) + beta * C[b]
// op_A/op_B: 'N' = no-transpose, 'T' = transpose
// Strides are in float elements, not bytes.

void cuda_batched_sgemm(char op_A, char op_B,
                        const float *A, const float *B, float *C,
                        int batch, int M, int K, int N,
                        long strideA, long strideB, long strideC,
                        float alpha, float beta);

// ── expand_dim: tile a[..., 1, ...] n times along axis → out[..., n, ...]
// Backward: sum output grad over axis into input grad.

void cuda_expand_dim_fwd(const float *a, float *out,
                         int outer, int inner, int n);
void cuda_expand_dim_bwd(const float *g, float *da,
                         int outer, int inner, int n);

// ── slice: extract contiguous slice of len elements at start along axis.
// outer = product of dims before axis, inner = product of dims after axis.
// Backward: scatter g back into da at the same offset.

void cuda_slice_fwd(const float *a, float *out,
                    int outer, int a_axis, int inner, int start, int len);
void cuda_slice_bwd(const float *g, float *da,
                    int outer, int a_axis, int inner, int start, int len);

// ── Adam optimizer step ───────────────────────────────────────────────────────
// Updates param in-place using m, v moment buffers (all on device).

void cuda_adam_step(float *param, float *m, float *v, const float *grad,
                    int n, float lr, float bc1, float bc2,
                    float beta1, float beta2, float eps);
void cuda_grad_sumsq(const float *grad, float *sum_out, int n);
void cuda_scale_grad(float *grad, float scale, int n);
void cuda_clip_scale_grad(const float *gpu_sumsq, float max_norm, float eps,
                           float *grad, int n);

#ifdef __cplusplus
}
#endif

#endif // OVG_CUDA_ENABLED
#endif // CUDA_OPS_H
