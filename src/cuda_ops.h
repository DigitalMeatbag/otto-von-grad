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

// cross_entropy: logits[R×C] + targets[R×C] → probs_cache[R×C], loss_out[1]
// loss_out must be zero-initialised.  After call, copy loss_out to host.
void cuda_cross_entropy_fwd(const float *logits, const float *targets,
                             float *probs_cache, float *loss_out,
                             int rows, int cols);
// backward: d_logits[i] += g * (probs[i] - targets[i]) / rows
void cuda_cross_entropy_bwd(const float *probs, const float *targets,
                             const float *g, float *d_logits,
                             int rows, int cols);

// dropout: apply pre-computed mask (generated on CPU, uploaded to cuda_cache)
void cuda_dropout_fwd(const float *a, const float *mask, float *out, int n);
void cuda_dropout_bwd(const float *mask, const float *g, float *da, int n);

// ── Adam optimizer step ───────────────────────────────────────────────────────
// Updates param in-place using m, v moment buffers (all on device).

void cuda_adam_step(float *param, float *m, float *v, const float *grad,
                    int n, float lr, float bc1, float bc2,
                    float beta1, float beta2, float eps);

#ifdef __cplusplus
}
#endif

#endif // OVG_CUDA_ENABLED
#endif // CUDA_OPS_H
