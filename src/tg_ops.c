#include "tg_ops.h"
#include "tg_train.h"
#include "tg_rng.h"
#include "ovg_error.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef OVG_CUDA_ENABLED
#include "tg_cuda.h"
#include "cuda_ops.h"
#endif

// ── Op factory ───────────────────────────────────────────────────────────────

static Tensor *make_op(int ndim, const int shape[],
                       Tensor *p0, Tensor *p1, TgBackwardFn fn) {
    Tensor *out = tg_new(ndim, shape);
    out->parents[0]  = p0;
    out->parents[1]  = p1;
    out->n_parents   = p1 ? 2 : 1;
    out->backward_fn = fn;
#ifdef OVG_CUDA_ENABLED
    if (p0 && p1 && p0->on_cuda != p1->on_cuda)
        ovg_fatal("make_op: mixed CUDA/CPU parents (p0.on_cuda=%d, p1.on_cuda=%d)",
                  p0->on_cuda, p1->on_cuda);
    if ((p0 && p0->on_cuda) || (p1 && p1->on_cuda))
        tg_cuda_alloc(out);
#endif
    return out;
}

static float gelu_value(float x) {
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(k * (x + c * x3)));
}

static float gelu_grad(float x) {
    const float k = 0.7978845608028654f;
    const float c = 0.044715f;
    float x2 = x * x;
    float u = k * (x + c * x * x2);
    float t = tanhf(u);
    float du = k * (1.0f + 3.0f * c * x2);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
}

// ── N-D transpose helper ──────────────────────────────────────────────────────

/* Permute src by swapping axes dim0/dim1; accumulate (do_add=1) or assign into dst. */
static void transpose_nd(const float *src, float *dst, int do_add,
                         int ndim, const int src_shape[],
                         int dim0, int dim1) {
    int n = 1;
    for (int i = 0; i < ndim; i++) n *= src_shape[i];

    int src_strides[TG_MAX_DIMS], dst_strides[TG_MAX_DIMS];
    int dst_shape[TG_MAX_DIMS];
    for (int i = 0; i < ndim; i++) dst_shape[i] = src_shape[i];
    { int t = dst_shape[dim0]; dst_shape[dim0] = dst_shape[dim1]; dst_shape[dim1] = t; }

    src_strides[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--)
        src_strides[i] = src_strides[i+1] * src_shape[i+1];
    dst_strides[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--)
        dst_strides[i] = dst_strides[i+1] * dst_shape[i+1];

    int mi[TG_MAX_DIMS];
    for (int k = 0; k < n; k++) {
        int tmp = k;
        for (int i = ndim-1; i >= 0; i--) { mi[i] = tmp % src_shape[i]; tmp /= src_shape[i]; }
        int t = mi[dim0]; mi[dim0] = mi[dim1]; mi[dim1] = t;
        int j = 0;
        for (int i = 0; i < ndim; i++) j += dst_strides[i] * mi[i];
        if (do_add) dst[j] += src[k];
        else        dst[j]  = src[k];
        /* restore mi for next iteration (restore the swap) */
        t = mi[dim0]; mi[dim0] = mi[dim1]; mi[dim1] = t;
    }
}

// ── Backward functions ────────────────────────────────────────────────────────

static void backward_add(Tensor *self) {
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    int n = tg_numel(self);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_add_bwd(self->cuda_grad, a->cuda_grad, b->cuda_grad, n); return; }
#endif
    for (int i = 0; i < n; i++) {
        a->grad[i] += self->grad[i];
        b->grad[i] += self->grad[i];
    }
}

static void backward_sub(Tensor *self) {
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    int n = tg_numel(self);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_sub_bwd(self->cuda_grad, a->cuda_grad, b->cuda_grad, n); return; }
#endif
    for (int i = 0; i < n; i++) {
        a->grad[i] += self->grad[i];
        b->grad[i] -= self->grad[i];
    }
}

static void backward_mul(Tensor *self) {
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    int n = tg_numel(self);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_mul_bwd(a->cuda_data, b->cuda_data, self->cuda_grad, a->cuda_grad, b->cuda_grad, n);
        return;
    }
#endif
    float *ad = TG_DATAF(a), *bd = TG_DATAF(b);
    for (int i = 0; i < n; i++) {
        a->grad[i] += bd[i] * self->grad[i];
        b->grad[i] += ad[i] * self->grad[i];
    }
}

static void backward_pow(Tensor *self) {
    Tensor *a = self->parents[0];
    float p = self->aux;
    int n = tg_numel(self);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_pow_bwd(a->cuda_data, self->cuda_grad, p, a->cuda_grad, n); return; }
#endif
    float *ad = TG_DATAF(a);
    for (int i = 0; i < n; i++)
        a->grad[i] += p * powf(ad[i], p - 1.0f) * self->grad[i];
}

static void backward_matmul(Tensor *self) {
    Tensor *A = self->parents[0];
    Tensor *B = self->parents[1];
    int M = A->shape[A->ndim - 2];
    int K = A->shape[A->ndim - 1];
    int N = B->shape[B->ndim - 1];
    int batch = tg_numel(A) / (M * K);
    int same_ndim = (A->ndim == B->ndim);

#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        int a_bf16 = (A->dtype == TG_DTYPE_BF16);
        /* For BF16 inputs, cast to F32 temp buffers for gradient computation.
           Gradients (a->grad, b->grad) are always F32. */
        float *A_data = a_bf16 ? tg_cuda_malloc_floats(tg_numel(A)) : A->cuda_data;
        float *B_data = a_bf16 ? tg_cuda_malloc_floats(tg_numel(B)) : B->cuda_data;
        if (a_bf16) {
            cuda_cast_bf16_to_f32(A->cuda_data, A_data, tg_numel(A));
            cuda_cast_bf16_to_f32(B->cuda_data, B_data, tg_numel(B));
        }

        if (same_ndim) {
            long sA = M * K, sB = K * N, sC = M * N;
            cuda_batched_sgemm('N', 'T', self->cuda_grad, B_data, A->cuda_grad,
                               batch, M, N, K, sC, sB, sA, 1.0f, 1.0f);
            cuda_batched_sgemm('T', 'N', A_data, self->cuda_grad, B->cuda_grad,
                               batch, K, M, N, sA, sC, sB, 1.0f, 1.0f);
        } else {
            long flat_m = (long)batch * M;
            cuda_matmul_bwd_dA(self->cuda_grad, B_data, A->cuda_grad,
                               (int)flat_m, K, N);
            cuda_matmul_bwd_dB(A_data, self->cuda_grad, B->cuda_grad,
                               (int)flat_m, K, N);
        }

        if (a_bf16) { tg_cuda_free_floats(A_data); tg_cuda_free_floats(B_data); }
        return;
    }
#endif

    float *Ad = TG_DATAF(A), *Bd = TG_DATAF(B), *dC = self->grad;
    if (same_ndim) {
        long sA = M * K, sB = K * N, sC = M * N;
        for (int b = 0; b < batch; b++) {
            const float *dout = dC + b * sC;
            float *da = A->grad  + b * sA;
            float *db = B->grad  + b * sB;
            const float *a = Ad + b * sA;
            const float *bm = Bd + b * sB;
            for (int i = 0; i < M; i++)
                for (int ki = 0; ki < K; ki++) {
                    float sum = 0.0f;
                    for (int j = 0; j < N; j++) sum += dout[i*N+j] * bm[ki*N+j];
                    da[i*K+ki] += sum;
                }
            for (int i = 0; i < M; i++)
                for (int ki = 0; ki < K; ki++) {
                    float aik = a[i*K+ki];
                    for (int j = 0; j < N; j++) db[ki*N+j] += aik * dout[i*N+j];
                }
        }
    } else {
        /* Broadcast B: treat as flat [batch*M, K] @ [K, N] */
        long flat_m = (long)batch * M;
        for (long i = 0; i < flat_m; i++)
            for (int ki = 0; ki < K; ki++) {
                float sum = 0.0f;
                for (int j = 0; j < N; j++) sum += dC[i*N+j] * Bd[ki*N+j];
                A->grad[i*K+ki] += sum;
            }
        for (long i = 0; i < flat_m; i++)
            for (int ki = 0; ki < K; ki++) {
                float aik = Ad[i*K+ki];
                for (int j = 0; j < N; j++) B->grad[ki*N+j] += aik * dC[i*N+j];
            }
    }
}

static void backward_tanh(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = tg_numel(self);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_tanh_bwd(self->cuda_data, self->cuda_grad, a->cuda_grad, n); return; }
#endif
    float *sd = TG_DATAF(self);
    for (int i = 0; i < n; i++) {
        float t = sd[i];
        a->grad[i] += (1.0f - t * t) * self->grad[i];
    }
}

static void backward_relu(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = tg_numel(self);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_relu_bwd(a->cuda_data, self->cuda_grad, a->cuda_grad, n); return; }
#endif
    float *ad = TG_DATAF(a);
    for (int i = 0; i < n; i++)
        a->grad[i] += self->grad[i] * (ad[i] > 0.0f ? 1.0f : 0.0f);
}

static void backward_gelu(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = tg_numel(self);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_gelu_bwd(a->cuda_data, self->cuda_grad, a->cuda_grad, n); return; }
#endif
    float *ad = TG_DATAF(a);
    for (int i = 0; i < n; i++)
        a->grad[i] += gelu_grad(ad[i]) * self->grad[i];
}

static void backward_sum(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = tg_numel(a);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_sum_bwd(self->cuda_grad, a->cuda_grad, n); return; }
#endif
    float g = self->grad[0];
    for (int i = 0; i < n; i++) a->grad[i] += g;
}

static void backward_mean(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = tg_numel(a);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_mean_bwd(self->cuda_grad, a->cuda_grad, n); return; }
#endif
    float g = self->grad[0] / (float)n;
    for (int i = 0; i < n; i++) a->grad[i] += g;
}

static void backward_transpose(Tensor *self) {
    Tensor *a = self->parents[0];
    int dim0 = (int)self->cache[0];
    int dim1 = (int)self->cache[1];

#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda && self->ndim == 2 && dim0 == 0 && dim1 == 1) {
        cuda_transpose_bwd(self->cuda_grad, a->cuda_grad, self->shape[0], self->shape[1]);
        return;
    }
#endif
    /* Backward of transpose is the same permutation applied to the gradient */
    transpose_nd(self->grad, a->grad, 1/*accumulate*/,
                 self->ndim, self->shape, dim0, dim1);
}

static void backward_scale(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = tg_numel(a);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_scale_bwd(self->cuda_grad, self->aux, a->cuda_grad, n); return; }
#endif
    for (int i = 0; i < n; i++)
        a->grad[i] += self->aux * self->grad[i];
}

static void backward_causal_mask(Tensor *self) {
    Tensor *scores = self->parents[0];
    int seq_len = self->shape[self->ndim - 1];
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_causal_mask_bwd(self->cuda_grad, scores->cuda_grad, seq_len); return; }
#endif
    int n = tg_numel(self);
    for (int k = 0; k < n; k += seq_len * seq_len) {
        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < seq_len; j++)
                if (j <= i)
                    scores->grad[k + i * seq_len + j] += self->grad[k + i * seq_len + j];
    }
}

static void backward_layer_norm(Tensor *self) {
    Tensor *a     = self->parents[0];
    Tensor *gamma = self->parents[1];
    Tensor *beta  = self->parents[2];
    int cols    = self->shape[self->ndim - 1];
    int rows    = tg_numel(self) / cols;

#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_layer_norm_rows_affine_bwd(self->cuda_cache + rows, self->cuda_grad,
                                        gamma->cuda_data, self->cuda_cache,
                                        a->cuda_grad, gamma->cuda_grad, beta->cuda_grad,
                                        rows, cols);
        return;
    }
#endif

    float *inv_std = self->cache;
    float *xhat    = self->cache + rows;
    float *gamma_d = TG_DATAF(gamma);

    for (int r = 0; r < rows; r++) {
        float *dy = self->grad + r * cols;
        float *dx = a->grad    + r * cols;
        float *xh = xhat       + r * cols;

        float mean_dxhat = 0.0f, mean_dxhat_xhat = 0.0f;
        for (int j = 0; j < cols; j++) {
            float g = dy[j];
            gamma->grad[j] += g * xh[j];
            beta->grad[j]  += g;
            float dxhat = g * gamma_d[j];
            mean_dxhat       += dxhat;
            mean_dxhat_xhat  += dxhat * xh[j];
        }
        mean_dxhat      /= (float)cols;
        mean_dxhat_xhat /= (float)cols;

        for (int j = 0; j < cols; j++) {
            float dxhat = dy[j] * gamma_d[j];
            dx[j] += inv_std[r] * (dxhat - mean_dxhat - xh[j] * mean_dxhat_xhat);
        }
    }
}

static void backward_softmax(Tensor *self) {
    Tensor *a = self->parents[0];
    int axis = (int)self->aux;
    int ndim = self->ndim;

    /* outer = product of dims before axis; inner = product of dims after axis */
    int outer = 1, axis_sz = self->shape[axis], inner = 1;
    for (int i = 0; i < axis; i++) outer *= self->shape[i];
    for (int i = axis + 1; i < ndim; i++) inner *= self->shape[i];

#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda && ndim == 2 && axis == 1) {
        cuda_softmax_rows_bwd(self->cuda_data, self->cuda_grad, a->cuda_grad,
                              outer, axis_sz);
        return;
    }
#endif
    float *sd = TG_DATAF(self);
    for (int o = 0; o < outer; o++) {
        for (int s = 0; s < inner; s++) {
            float dot = 0.0f;
            for (int i = 0; i < axis_sz; i++) {
                int idx = (o * axis_sz + i) * inner + s;
                dot += self->grad[idx] * sd[idx];
            }
            for (int i = 0; i < axis_sz; i++) {
                int idx = (o * axis_sz + i) * inner + s;
                a->grad[idx] += sd[idx] * (self->grad[idx] - dot);
            }
        }
    }
}

static void backward_cross_entropy(Tensor *self) {
    Tensor *logits  = self->parents[0];
    Tensor *targets = self->parents[1];
    int rows = logits->shape[0], cols = logits->shape[1];
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_cross_entropy_bwd(self->cuda_cache, targets->cuda_data,
                               self->cuda_grad, logits->cuda_grad, rows, cols);
        return;
    }
#endif
    float *probs = self->cache;
    float scale = self->grad[0] / (float)rows;
    float *td = TG_DATAF(targets);
    for (int i = 0; i < rows * cols; i++)
        logits->grad[i] += scale * (probs[i] - td[i]);
}

static void backward_cross_entropy_sparse(Tensor *self) {
    Tensor *logits = self->parents[0];
    int rows = logits->shape[0], cols = logits->shape[1];
    float smoothing = self->aux;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_cross_entropy_sparse_bwd(self->cuda_cache, self->cuda_cache + rows * cols,
                                      self->cuda_grad, smoothing, logits->cuda_grad,
                                      rows, cols);
        return;
    }
#endif
    float *probs = self->cache;
    float *ids = self->cache + rows * cols;
    float scale = self->grad[0] / (float)rows;
    float off = cols > 1 ? smoothing / (float)(cols - 1) : 0.0f;
    for (int r = 0; r < rows; r++) {
        int id = (int)ids[r];
        for (int j = 0; j < cols; j++) {
            float target = (j == id) ? (1.0f - smoothing) : off;
            logits->grad[r * cols + j] += scale * (probs[r * cols + j] - target);
        }
    }
}

static void backward_mean_rows(Tensor *self) {
    Tensor *a = self->parents[0];
    int rows = a->shape[0], cols = a->shape[1];
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_mean_rows_bwd(self->cuda_grad, a->cuda_grad, rows, cols);
        return;
    }
#endif
    float inv_r = 1.0f / (float)rows;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            a->grad[i * cols + j] += self->grad[j] * inv_r;
}

static void backward_embed(Tensor *self) {
    Tensor *weight = self->parents[0];
    int T = self->shape[0], C = self->shape[1];
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_embed_bwd(self->cuda_grad, self->cuda_cache, weight->cuda_grad, T, C);
        return;
    }
#endif
    for (int t = 0; t < T; t++) {
        int id = (int)self->cache[t];
        for (int c = 0; c < C; c++)
            weight->grad[id * C + c] += self->grad[t * C + c];
    }
}

static void backward_dropout(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = tg_numel(self);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_dropout_bwd(self->cuda_cache, self->cuda_grad, a->cuda_grad, n);
        return;
    }
#endif
    for (int i = 0; i < n; i++)
        a->grad[i] += self->cache[i] * self->grad[i];
}

// ── Ops ───────────────────────────────────────────────────────────────────────

Tensor *tg_add(Tensor *a, Tensor *b) {
    if (a->ndim != b->ndim)
        ovg_fatal("tg_add: ndim mismatch %d vs %d", a->ndim, b->ndim);
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i])
            ovg_fatal("tg_add: shape mismatch at dim %d", i);
    Tensor *out = make_op(a->ndim, a->shape, a, b, backward_add);
    int n = tg_numel(a);
    float *ad = TG_DATAF(a), *bd = TG_DATAF(b), *od = TG_DATAF(out);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_add_fwd(a->cuda_data, b->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) od[i] = ad[i] + bd[i];
    return out;
}

Tensor *tg_sub(Tensor *a, Tensor *b) {
    if (a->ndim != b->ndim)
        ovg_fatal("tg_sub: ndim mismatch %d vs %d", a->ndim, b->ndim);
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i])
            ovg_fatal("tg_sub: shape mismatch at dim %d", i);
    Tensor *out = make_op(a->ndim, a->shape, a, b, backward_sub);
    int n = tg_numel(a);
    float *ad = TG_DATAF(a), *bd = TG_DATAF(b), *od = TG_DATAF(out);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_sub_fwd(a->cuda_data, b->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) od[i] = ad[i] - bd[i];
    return out;
}

Tensor *tg_mul(Tensor *a, Tensor *b) {
    if (a->ndim != b->ndim)
        ovg_fatal("tg_mul: ndim mismatch %d vs %d", a->ndim, b->ndim);
    for (int i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i])
            ovg_fatal("tg_mul: shape mismatch at dim %d", i);
    Tensor *out = make_op(a->ndim, a->shape, a, b, backward_mul);
    int n = tg_numel(a);
    float *ad = TG_DATAF(a), *bd = TG_DATAF(b), *od = TG_DATAF(out);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_mul_fwd(a->cuda_data, b->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) od[i] = ad[i] * bd[i];
    return out;
}

Tensor *tg_pow(Tensor *a, float p) {
    Tensor *out = make_op(a->ndim, a->shape, a, NULL, backward_pow);
    out->aux = p;
    int n = tg_numel(a);
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_pow_fwd(a->cuda_data, p, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) od[i] = powf(ad[i], p);
    return out;
}

Tensor *tg_matmul(Tensor *a, Tensor *b) {
    if (b->ndim < 2)
        ovg_fatal("tg_matmul: b->ndim must be >= 2, got %d", b->ndim);
    if (b->ndim > a->ndim)
        ovg_fatal("tg_matmul: b->ndim (%d) > a->ndim (%d); only second operand may be lower-ndim",
                  b->ndim, a->ndim);

    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int N = b->shape[b->ndim - 1];
    if (b->shape[b->ndim - 2] != K)
        ovg_fatal("tg_matmul: inner dim mismatch: A[...,%d,%d] @ B[...,%d,%d]",
                  M, K, b->shape[b->ndim-2], N);

    /* Validate same-ndim leading dims */
    if (b->ndim == a->ndim) {
        for (int i = 0; i < a->ndim - 2; i++)
            if (a->shape[i] != b->shape[i])
                ovg_fatal("tg_matmul: leading dim mismatch at axis %d: %d vs %d",
                          i, a->shape[i], b->shape[i]);
    }

    /* Build output shape: a->shape[0..ndim-3] ++ [M, N] */
    int out_shape[TG_MAX_DIMS];
    for (int i = 0; i < a->ndim - 2; i++) out_shape[i] = a->shape[i];
    out_shape[a->ndim - 2] = M;
    out_shape[a->ndim - 1] = N;

    Tensor *out = make_op(a->ndim, out_shape, a, b, backward_matmul);
    int batch = tg_numel(a) / (M * K);

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        int a_bf16 = (a->dtype == TG_DTYPE_BF16);
        int b_bf16 = (b->dtype == TG_DTYPE_BF16);
        if (a_bf16 != b_bf16)
            ovg_fatal("tg_matmul: mixed dtype (one BF16, one F32) not supported");
        if (a_bf16) {
            /* BF16 × BF16 → F32 via cublasGemmEx */
            if (a->ndim == b->ndim) {
                long sA = M * K, sB = K * N, sC = M * N;
                cuda_batched_sgemm_bf16('N', 'N', a->cuda_data, b->cuda_data, out->cuda_data,
                                        batch, M, K, N, sA, sB, sC, 1.0f, 0.0f);
            } else {
                cuda_batched_sgemm_bf16('N', 'N', a->cuda_data, b->cuda_data, out->cuda_data,
                                        1, batch * M, K, N, (long)batch*M*K, 0, (long)batch*M*N,
                                        1.0f, 0.0f);
            }
        } else {
            if (a->ndim == b->ndim) {
                long sA = M * K, sB = K * N, sC = M * N;
                cuda_batched_sgemm('N', 'N', a->cuda_data, b->cuda_data, out->cuda_data,
                                   batch, M, K, N, sA, sB, sC, 1.0f, 0.0f);
            } else {
                cuda_matmul_forward(a->cuda_data, b->cuda_data, out->cuda_data,
                                    batch * M, K, N);
            }
        }
        return out;
    }
#endif

    float *ad = TG_DATAF(a), *bd = TG_DATAF(b), *od = TG_DATAF(out);
    if (a->ndim == b->ndim) {
        long sA = M * K, sB = K * N, sC = M * N;
        for (int bi = 0; bi < batch; bi++) {
            const float *ab = ad + bi * sA;
            const float *bb = bd + bi * sB;
            float *cb = od + bi * sC;
            for (int i = 0; i < M; i++)
                for (int ki = 0; ki < K; ki++) {
                    float aik = ab[i*K+ki];
                    for (int j = 0; j < N; j++) cb[i*N+j] += aik * bb[ki*N+j];
                }
        }
    } else {
        /* Broadcast B: flat loop [batch*M, K] @ [K, N] */
        long flat_m = (long)batch * M;
        for (long i = 0; i < flat_m; i++)
            for (int ki = 0; ki < K; ki++) {
                float aik = ad[i*K+ki];
                for (int j = 0; j < N; j++) od[i*N+j] += aik * bd[ki*N+j];
            }
    }
    return out;
}

Tensor *tg_tanh(Tensor *a) {
    Tensor *out = make_op(a->ndim, a->shape, a, NULL, backward_tanh);
    int n = tg_numel(a);
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_tanh_fwd(a->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) od[i] = tanhf(ad[i]);
    return out;
}

Tensor *tg_relu(Tensor *a) {
    Tensor *out = make_op(a->ndim, a->shape, a, NULL, backward_relu);
    int n = tg_numel(a);
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_relu_fwd(a->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) od[i] = ad[i] > 0.0f ? ad[i] : 0.0f;
    return out;
}

Tensor *tg_gelu(Tensor *a) {
    Tensor *out = make_op(a->ndim, a->shape, a, NULL, backward_gelu);
    int n = tg_numel(a);
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_gelu_fwd(a->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) od[i] = gelu_value(ad[i]);
    return out;
}

Tensor *tg_sum(Tensor *a) {
    int s[2] = {1, 1};
    Tensor *out = make_op(2, s, a, NULL, backward_sum);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_sum_fwd(a->cuda_data, out->cuda_data, tg_numel(a)); return out; }
#endif
    int n = tg_numel(a);
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += ad[i];
    od[0] = sum;
    return out;
}

Tensor *tg_mean(Tensor *a) {
    int s[2] = {1, 1};
    Tensor *out = make_op(2, s, a, NULL, backward_mean);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        int n = tg_numel(a);
        cuda_sum_fwd(a->cuda_data, out->cuda_data, n);
        cuda_scale_fwd(out->cuda_data, 1.0f / (float)n, out->cuda_data, 1);
        return out;
    }
#endif
    int n = tg_numel(a);
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += ad[i];
    od[0] = sum / (float)n;
    return out;
}

Tensor *tg_transpose(Tensor *a, int dim0, int dim1) {
    if (dim0 < 0 || dim0 >= a->ndim || dim1 < 0 || dim1 >= a->ndim)
        ovg_fatal("tg_transpose: dims (%d, %d) out of range for ndim=%d",
                  dim0, dim1, a->ndim);
    if (dim0 == dim1)
        ovg_fatal("tg_transpose: dim0 == dim1 == %d", dim0);

    int out_shape[TG_MAX_DIMS];
    for (int i = 0; i < a->ndim; i++) out_shape[i] = a->shape[i];
    { int t = out_shape[dim0]; out_shape[dim0] = out_shape[dim1]; out_shape[dim1] = t; }

    Tensor *out = make_op(a->ndim, out_shape, a, NULL, backward_transpose);
    out->cache = malloc(2 * sizeof(float));
    if (!out->cache) ovg_fatal("tg_transpose: out of memory");
    out->cache[0] = (float)dim0;
    out->cache[1] = (float)dim1;

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda && a->ndim == 2 && dim0 == 0 && dim1 == 1) {
        cuda_transpose_fwd(a->cuda_data, out->cuda_data, a->shape[0], a->shape[1]);
        return out;
    }
#endif
    transpose_nd(TG_DATAF(a), TG_DATAF(out), 0/*assign*/,
                 a->ndim, a->shape, dim0, dim1);
    return out;
}

Tensor *tg_scale(Tensor *a, float s) {
    Tensor *out = make_op(a->ndim, a->shape, a, NULL, backward_scale);
    out->aux = s;
    int n = tg_numel(a);
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_scale_fwd(a->cuda_data, s, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) od[i] = ad[i] * s;
    return out;
}

Tensor *tg_causal_mask(Tensor *scores) {
    int ndim = scores->ndim;
    int seq_len = scores->shape[ndim - 1];
    if (scores->shape[ndim - 2] != seq_len)
        ovg_fatal("tg_causal_mask: last two dims must be equal, got [%dx%d]",
                  scores->shape[ndim - 2], seq_len);
    Tensor *out = make_op(ndim, scores->shape, scores, NULL, backward_causal_mask);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        /* CUDA kernel handles 2D [T×T]; for N-D call it per-slice */
        if (ndim == 2) {
            cuda_causal_mask_fwd(scores->cuda_data, out->cuda_data, seq_len);
            return out;
        }
    }
#endif
    int n = tg_numel(scores);
    float *sd = TG_DATAF(scores), *od = TG_DATAF(out);
    for (int k = 0; k < n; k += seq_len * seq_len) {
        for (int i = 0; i < seq_len; i++)
            for (int j = 0; j < seq_len; j++)
                od[k + i * seq_len + j] = (j <= i)
                    ? sd[k + i * seq_len + j] : -1.0e9f;
    }
    return out;
}

Tensor *tg_layer_norm(Tensor *a, Tensor *gamma, Tensor *beta, float eps) {
    if (!a || !gamma || !beta)
        ovg_fatal("tg_layer_norm: NULL argument");
    int C = a->shape[a->ndim - 1];
    if (gamma->shape[gamma->ndim - 1] != C || beta->shape[beta->ndim - 1] != C)
        ovg_fatal("tg_layer_norm: gamma/beta last dim must equal a->shape[ndim-1]=%d (got %d and %d)",
                  C, gamma->shape[gamma->ndim-1], beta->shape[beta->ndim-1]);
#ifdef OVG_CUDA_ENABLED
    if (a->on_cuda != gamma->on_cuda || a->on_cuda != beta->on_cuda)
        ovg_fatal("tg_layer_norm: mixed CUDA/CPU parents");
#endif

    int rows = tg_numel(a) / C;
    Tensor *out = tg_new(a->ndim, a->shape);
    out->parents[0] = a;
    out->parents[1] = gamma;
    out->parents[2] = beta;
    out->n_parents = 3;
    out->backward_fn = backward_layer_norm;
    out->aux = eps;

#ifdef OVG_CUDA_ENABLED
    if (a->on_cuda) {
        tg_cuda_alloc(out);
        tg_cuda_alloc_cache(out, rows + rows * C);
        cuda_layer_norm_rows_affine_fwd(a->cuda_data, gamma->cuda_data, beta->cuda_data,
                                        eps, out->cuda_data, out->cuda_cache,
                                        rows, C);
        return out;
    }
#endif

    out->cache = malloc((size_t)(rows + rows * C) * sizeof(float));
    if (!out->cache) ovg_fatal("tg_layer_norm: out of memory");
    float *inv_std = out->cache;
    float *xhat    = out->cache + rows;
    float *ad = TG_DATAF(a), *gd = TG_DATAF(gamma), *bd = TG_DATAF(beta);
    float *od = TG_DATAF(out);

    for (int r = 0; r < rows; r++) {
        float *x = ad + r * C;
        float sum = 0.0f, sum2 = 0.0f;
        for (int j = 0; j < C; j++) { sum += x[j]; sum2 += x[j] * x[j]; }
        float mean = sum / (float)C;
        float var  = sum2 / (float)C - mean * mean;
        float is   = 1.0f / sqrtf(var + eps);
        inv_std[r] = is;
        for (int j = 0; j < C; j++) {
            float norm = (x[j] - mean) * is;
            xhat[r * C + j] = norm;
            od[r * C + j]   = norm * gd[j] + bd[j];
        }
    }
    return out;
}

Tensor *tg_softmax(Tensor *a, int axis) {
    if (axis < 0 || axis >= a->ndim)
        ovg_fatal("tg_softmax: axis=%d out of range for ndim=%d", axis, a->ndim);

    Tensor *out = make_op(a->ndim, a->shape, a, NULL, backward_softmax);
    out->aux = (float)axis;

    int outer = 1, axis_sz = a->shape[axis], inner = 1;
    for (int i = 0; i < axis; i++) outer *= a->shape[i];
    for (int i = axis + 1; i < a->ndim; i++) inner *= a->shape[i];

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda && a->ndim == 2 && axis == 1) {
        cuda_softmax_rows_fwd(a->cuda_data, out->cuda_data, outer, axis_sz);
        return out;
    }
#endif
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
    for (int o = 0; o < outer; o++) {
        for (int s = 0; s < inner; s++) {
            float mx = ad[(o * axis_sz) * inner + s];
            for (int i = 1; i < axis_sz; i++) {
                float v = ad[(o * axis_sz + i) * inner + s];
                if (v > mx) mx = v;
            }
            float sum = 0.0f;
            for (int i = 0; i < axis_sz; i++) {
                int idx = (o * axis_sz + i) * inner + s;
                od[idx] = expf(ad[idx] - mx);
                sum += od[idx];
            }
            for (int i = 0; i < axis_sz; i++)
                od[(o * axis_sz + i) * inner + s] /= sum;
        }
    }
    return out;
}

static Tensor *cross_entropy_dense_impl(Tensor *logits, Tensor *targets, int sync_scalar) {
    if (logits->ndim != 2 || targets->ndim != 2)
        ovg_fatal("tg_cross_entropy: expected 2D inputs");
    if (logits->shape[0] != targets->shape[0] || logits->shape[1] != targets->shape[1])
        ovg_fatal("tg_cross_entropy: shape mismatch [%dx%d] vs [%dx%d]",
                  logits->shape[0], logits->shape[1], targets->shape[0], targets->shape[1]);
    int rows = logits->shape[0], cols = logits->shape[1];
    int s[2] = {1, 1};
    Tensor *out = make_op(2, s, logits, targets, backward_cross_entropy);

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        tg_cuda_alloc_cache(out, rows * cols);
        cuda_cross_entropy_fwd(logits->cuda_data, targets->cuda_data,
                               out->cuda_cache, out->cuda_data, rows, cols);
        if (sync_scalar) tg_from_cuda(out);
        return out;
    }
#endif

    out->cache = malloc(rows * cols * sizeof(float));
    if (!out->cache) ovg_fatal("tg_cross_entropy: out of memory");
    float *logd = TG_DATAF(logits), *tgtd = TG_DATAF(targets), *od = TG_DATAF(out);
    float loss = 0.0f;
    for (int r = 0; r < rows; r++) {
        float *row  = logd + r * cols;
        float *prob = out->cache + r * cols;
        const float *tgt = tgtd + r * cols;
        float mx = row[0];
        for (int j = 1; j < cols; j++) if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) { prob[j] = expf(row[j] - mx); sum += prob[j]; }
        float log_sum_exp = logf(sum) + mx;
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) {
            prob[j] *= inv_sum;
            loss -= tgt[j] * (row[j] - log_sum_exp);
        }
    }
    od[0] = loss / (float)rows;
    return out;
}

Tensor *tg_cross_entropy(Tensor *logits, Tensor *targets) {
    return cross_entropy_dense_impl(logits, targets, 1);
}

Tensor *tg_cross_entropy_no_sync(Tensor *logits, Tensor *targets) {
    return cross_entropy_dense_impl(logits, targets, 0);
}

static Tensor *cross_entropy_sparse_impl(Tensor *logits, const int *class_ids,
                                         int n_ids, float label_smoothing,
                                         int sync_scalar) {
    if (!logits || !class_ids)
        ovg_fatal("tg_cross_entropy_sparse: NULL argument");
    if (logits->ndim != 2)
        ovg_fatal("tg_cross_entropy_sparse: expected 2D logits");
    if (n_ids != logits->shape[0])
        ovg_fatal("tg_cross_entropy_sparse: n_ids=%d, expected rows=%d",
                  n_ids, logits->shape[0]);
    if (label_smoothing < 0.0f || label_smoothing >= 1.0f)
        ovg_fatal("tg_cross_entropy_sparse: label_smoothing %.6f out of range [0,1)",
                  label_smoothing);
    int rows = logits->shape[0], cols = logits->shape[1];
    if (cols < 2 && label_smoothing > 0.0f)
        ovg_fatal("tg_cross_entropy_sparse: smoothing requires at least 2 classes");
    for (int r = 0; r < rows; r++) {
        if (class_ids[r] < 0 || class_ids[r] >= cols)
            ovg_fatal("tg_cross_entropy_sparse: class id %d out of range [0, %d)",
                      class_ids[r], cols);
    }

    int s[2] = {1, 1};
    Tensor *out = make_op(2, s, logits, NULL, backward_cross_entropy_sparse);
    out->aux = label_smoothing;

    float *ids = malloc((size_t)rows * sizeof(float));
    if (!ids) ovg_fatal("tg_cross_entropy_sparse: out of memory");
    for (int r = 0; r < rows; r++) ids[r] = (float)class_ids[r];

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        float *host_cache = calloc((size_t)(rows * cols + rows), sizeof(float));
        if (!host_cache) ovg_fatal("tg_cross_entropy_sparse: out of memory");
        for (int r = 0; r < rows; r++) host_cache[rows * cols + r] = ids[r];
        tg_cuda_alloc_cache(out, rows * cols + rows);
        tg_cuda_upload_cache(out, host_cache, rows * cols + rows);
        cuda_cross_entropy_sparse_fwd(logits->cuda_data, out->cuda_cache + rows * cols,
                                      label_smoothing, out->cuda_cache,
                                      out->cuda_data, rows, cols);
        free(host_cache); free(ids);
        if (sync_scalar) tg_from_cuda(out);
        return out;
    }
#endif

    out->cache = malloc((size_t)(rows * cols + rows) * sizeof(float));
    if (!out->cache) ovg_fatal("tg_cross_entropy_sparse: out of memory");
    float *probs = out->cache;
    float *cache_ids = out->cache + rows * cols;
    for (int r = 0; r < rows; r++) cache_ids[r] = ids[r];
    free(ids);

    float *logd = TG_DATAF(logits), *od = TG_DATAF(out);
    float loss = 0.0f;
    float off  = cols > 1 ? label_smoothing / (float)(cols - 1) : 0.0f;
    for (int r = 0; r < rows; r++) {
        float *row  = logd + r * cols;
        float *prob = probs + r * cols;
        int id = class_ids[r];
        float mx = row[0];
        for (int j = 1; j < cols; j++) if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) { prob[j] = expf(row[j] - mx); sum += prob[j]; }
        float log_sum_exp = logf(sum) + mx;
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) {
            prob[j] *= inv_sum;
            float target = (j == id) ? (1.0f - label_smoothing) : off;
            loss -= target * (row[j] - log_sum_exp);
        }
    }
    od[0] = loss / (float)rows;
    return out;
}

Tensor *tg_cross_entropy_sparse(Tensor *logits, const int *class_ids,
                                int n_ids, float label_smoothing) {
    return cross_entropy_sparse_impl(logits, class_ids, n_ids, label_smoothing, 1);
}

Tensor *tg_cross_entropy_sparse_no_sync(Tensor *logits, const int *class_ids,
                                        int n_ids, float label_smoothing) {
    return cross_entropy_sparse_impl(logits, class_ids, n_ids, label_smoothing, 0);
}

Tensor *tg_mean_rows(Tensor *a) {
    if (a->ndim != 2)
        ovg_fatal("tg_mean_rows: expected 2D tensor");
    int rows = a->shape[0], cols = a->shape[1];
    int out_shape[2] = {1, cols};
    Tensor *out = make_op(2, out_shape, a, NULL, backward_mean_rows);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_mean_rows_fwd(a->cuda_data, out->cuda_data, rows, cols); return out; }
#endif
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
    for (int j = 0; j < cols; j++) {
        float s = 0.0f;
        for (int i = 0; i < rows; i++) s += ad[i * cols + j];
        od[j] = s / (float)rows;
    }
    return out;
}

Tensor *tg_dropout(Tensor *a, float p) {
    Tensor *out = make_op(a->ndim, a->shape, a, NULL, backward_dropout);
    int n = tg_numel(a);

    float *mask = malloc((size_t)n * sizeof(float));
    if (!mask) ovg_fatal("tg_dropout: out of memory");

    if (!tg_training || p <= 0.0f) {
        for (int i = 0; i < n; i++) mask[i] = 1.0f;
    } else {
        float inv_keep = 1.0f / (1.0f - p);
        const float scale = 1.0f / 4294967296.0f;
        for (int i = 0; i < n; i++)
            mask[i] = (tg_rng_xorshift32() * scale) >= p ? inv_keep : 0.0f;
    }

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        tg_cuda_alloc_cache(out, n);
        tg_cuda_upload_cache(out, mask, n);
        cuda_dropout_fwd(a->cuda_data, out->cuda_cache, out->cuda_data, n);
        free(mask);
        return out;
    }
#endif

    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
    out->cache = mask;
    for (int i = 0; i < n; i++) od[i] = ad[i] * mask[i];
    return out;
}

Tensor *tg_embed(Tensor *weight, const int *token_ids, int seq_len) {
    if (weight->ndim != 2)
        ovg_fatal("tg_embed: weight must be 2D [V x C]");
    int V = weight->shape[0], C = weight->shape[1];
    for (int t = 0; t < seq_len; t++) {
        if (token_ids[t] < 0 || token_ids[t] >= V)
            ovg_fatal("tg_embed: token id %d out of range [0, %d)", token_ids[t], V);
    }

    int out_shape[2] = {seq_len, C};
    Tensor *out = make_op(2, out_shape, weight, NULL, backward_embed);

    out->cache = malloc((size_t)seq_len * sizeof(float));
    if (!out->cache) ovg_fatal("tg_embed: out of memory");
    for (int t = 0; t < seq_len; t++) out->cache[t] = (float)token_ids[t];

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        tg_cuda_alloc_cache(out, seq_len);
        tg_cuda_upload_cache(out, out->cache, seq_len);
        cuda_embed_fwd(weight->cuda_data, out->cuda_cache, out->cuda_data, seq_len, C);
        return out;
    }
#endif

    float *wd = TG_DATAF(weight), *od = TG_DATAF(out);
    for (int t = 0; t < seq_len; t++) {
        int id = token_ids[t];
        for (int c = 0; c < C; c++)
            od[t * C + c] = wd[id * C + c];
    }
    return out;
}

// ── Phase 2+ stubs ────────────────────────────────────────────────────────────

static void backward_reshape(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = tg_numel(self);
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_scale_bwd(self->cuda_grad, 1.0f, a->cuda_grad, n); return; }
#endif
    for (int i = 0; i < n; i++) a->grad[i] += self->grad[i];
}

Tensor *tg_reshape(Tensor *a, int ndim, const int shape[]) {
    if (ndim < 2 || ndim > TG_MAX_DIMS)
        ovg_fatal("tg_reshape: ndim=%d out of range [2, %d]", ndim, TG_MAX_DIMS);
    int nel = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0)
            ovg_fatal("tg_reshape: shape[%d]=%d must be positive", i, shape[i]);
        nel *= shape[i];
    }
    if (nel != tg_numel(a))
        ovg_fatal("tg_reshape: element count mismatch: input has %d, new shape has %d",
                  tg_numel(a), nel);

    Tensor *out = make_op(ndim, shape, a, NULL, backward_reshape);

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        cuda_scale_fwd(a->cuda_data, 1.0f, out->cuda_data, nel);
        return out;
    }
#endif
    memcpy(TG_DATAF(out), TG_DATAF(a), (size_t)nel * sizeof(float));
    return out;
}

// ── tg_expand_dim ────────────────────────────────────────────────────────────

static void backward_expand_dim(Tensor *self) {
    Tensor *a    = self->parents[0];
    int axis     = (int)self->aux;
    int ndim     = self->ndim;
    int n        = self->shape[axis];
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= self->shape[i];
    for (int i = axis + 1; i < ndim; i++) inner *= self->shape[i];

#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_expand_dim_bwd(self->cuda_grad, a->cuda_grad, outer, inner, n);
        return;
    }
#endif
    for (int o = 0; o < outer; o++)
        for (int s = 0; s < inner; s++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++)
                sum += self->grad[(o * n + i) * inner + s];
            a->grad[o * inner + s] += sum;
        }
}

Tensor *tg_expand_dim(Tensor *a, int axis, int n) {
    if (axis < 0 || axis >= a->ndim)
        ovg_fatal("tg_expand_dim: axis=%d out of range for ndim=%d", axis, a->ndim);
    if (a->shape[axis] != 1)
        ovg_fatal("tg_expand_dim: shape[%d]=%d must be 1 to expand", axis, a->shape[axis]);
    if (n < 1)
        ovg_fatal("tg_expand_dim: n=%d must be >= 1", n);

    int out_shape[TG_MAX_DIMS];
    for (int i = 0; i < a->ndim; i++) out_shape[i] = a->shape[i];
    out_shape[axis] = n;

    Tensor *out = make_op(a->ndim, out_shape, a, NULL, backward_expand_dim);
    out->aux = (float)axis;

    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= a->shape[i];
    for (int i = axis + 1; i < a->ndim; i++) inner *= a->shape[i];

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        cuda_expand_dim_fwd(a->cuda_data, out->cuda_data, outer, inner, n);
        return out;
    }
#endif
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
    for (int o = 0; o < outer; o++)
        for (int i = 0; i < n; i++)
            for (int s = 0; s < inner; s++)
                od[(o * n + i) * inner + s] = ad[o * inner + s];
    return out;
}

// ── tg_cast ──────────────────────────────────────────────────────────────────

static void backward_cast(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = tg_numel(self);
    /* grad is always FP32; no dtype conversion needed */
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_scale_bwd(self->cuda_grad, 1.0f, a->cuda_grad, n); return; }
#endif
    for (int i = 0; i < n; i++) a->grad[i] += self->grad[i];
}

Tensor *tg_cast(Tensor *a, TgDtype dtype) {
    if (dtype == a->dtype)
        ovg_fatal("tg_cast: source and target dtype are the same");
    if (dtype == TG_DTYPE_BF16 && !a->on_cuda)
        ovg_fatal("tg_cast: TG_DTYPE_BF16 is only valid on CUDA tensors");
    if (a->dtype == TG_DTYPE_BF16 && !a->on_cuda)
        ovg_fatal("tg_cast: source BF16 tensor must be on CUDA");

    int nel = tg_numel(a);
    /* Allocate output tensor with new dtype */
    Tensor *out = tg_new(a->ndim, a->shape);  /* F32 alloc; we override dtype below */
    out->dtype = dtype;
    out->parents[0]  = a;
    out->n_parents   = 1;
    out->backward_fn = backward_cast;

#ifdef OVG_CUDA_ENABLED
    if (a->on_cuda) {
        /* Allocate CUDA buffers with dtype-aware element size */
        tg_cuda_alloc(out);
        if (dtype == TG_DTYPE_BF16) {
            cuda_cast_f32_to_bf16(a->cuda_data, out->cuda_data, nel);
        } else {
            /* BF16 → F32 */
            cuda_cast_bf16_to_f32(a->cuda_data, out->cuda_data, nel);
        }
        return out;
    }
#endif
    ovg_fatal("tg_cast: BF16 is CUDA-only; CPU path unreachable");
    return NULL;
}

// ── tg_slice ─────────────────────────────────────────────────────────────────

static void backward_slice(Tensor *self) {
    Tensor *a  = self->parents[0];
    int axis   = (int)self->cache[0];
    int start  = (int)self->cache[1];
    int len    = self->shape[axis];
    int ndim   = self->ndim;
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= self->shape[i];
    for (int i = axis + 1; i < ndim; i++) inner *= self->shape[i];
    int a_axis = a->shape[axis];

#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_slice_bwd(self->cuda_grad, a->cuda_grad, outer, a_axis, inner, start, len);
        return;
    }
#endif
    for (int o = 0; o < outer; o++)
        for (int i = 0; i < len; i++)
            for (int s = 0; s < inner; s++)
                a->grad[(o * a_axis + (start + i)) * inner + s] +=
                    self->grad[(o * len + i) * inner + s];
}

Tensor *tg_slice(Tensor *a, int axis, int start, int len) {
    if (axis < 0 || axis >= a->ndim)
        ovg_fatal("tg_slice: axis=%d out of range for ndim=%d", axis, a->ndim);
    if (start < 0 || len < 1 || start + len > a->shape[axis])
        ovg_fatal("tg_slice: invalid range [%d, %d) for axis %d size %d",
                  start, start + len, axis, a->shape[axis]);

    int out_shape[TG_MAX_DIMS];
    for (int i = 0; i < a->ndim; i++) out_shape[i] = a->shape[i];
    out_shape[axis] = len;

    Tensor *out = make_op(a->ndim, out_shape, a, NULL, backward_slice);
    out->cache = malloc(2 * sizeof(float));
    if (!out->cache) ovg_fatal("tg_slice: out of memory");
    out->cache[0] = (float)axis;
    out->cache[1] = (float)start;

    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= a->shape[i];
    for (int i = axis + 1; i < a->ndim; i++) inner *= a->shape[i];
    int a_axis = a->shape[axis];

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        cuda_slice_fwd(a->cuda_data, out->cuda_data, outer, a_axis, inner, start, len);
        return out;
    }
#endif
    float *ad = TG_DATAF(a), *od = TG_DATAF(out);
    for (int o = 0; o < outer; o++)
        for (int i = 0; i < len; i++)
            for (int s = 0; s < inner; s++)
                od[(o * len + i) * inner + s] = ad[(o * a_axis + (start + i)) * inner + s];
    return out;
}
