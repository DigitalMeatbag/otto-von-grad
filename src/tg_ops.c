#include "tg_ops.h"
#include "tg_train.h"
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

static Tensor *make_op(int rows, int cols, Tensor *p0, Tensor *p1, TgBackwardFn fn) {
    Tensor *out = tg_new(rows, cols);
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

// ── Backward functions ────────────────────────────────────────────────────────

static void backward_add(Tensor *self) {
    Tensor *a = self->parents[0];
    Tensor *b = self->parents[1];
    int n = self->rows * self->cols;
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
    int n = self->rows * self->cols;
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
    int n = self->rows * self->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_mul_bwd(a->cuda_data, b->cuda_data, self->cuda_grad, a->cuda_grad, b->cuda_grad, n);
        return;
    }
#endif
    for (int i = 0; i < n; i++) {
        a->grad[i] += b->data[i] * self->grad[i];
        b->grad[i] += a->data[i] * self->grad[i];
    }
}

static void backward_pow(Tensor *self) {
    Tensor *a = self->parents[0];
    float p = self->aux;
    int n = self->rows * self->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_pow_bwd(a->cuda_data, self->cuda_grad, p, a->cuda_grad, n); return; }
#endif
    for (int i = 0; i < n; i++)
        a->grad[i] += p * powf(a->data[i], p - 1.0f) * self->grad[i];
}

static void backward_matmul(Tensor *self) {
    Tensor *A = self->parents[0];  // [m, K]
    Tensor *B = self->parents[1];  // [K, n]
    int m = A->rows, K = A->cols, n = B->cols;

#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_matmul_bwd_dA(self->cuda_grad, B->cuda_data, A->cuda_grad, m, K, n);
        cuda_matmul_bwd_dB(A->cuda_data, self->cuda_grad, B->cuda_grad, m, K, n);
        return;
    }
#endif

    // dA += dC @ B^T  — inner loop over j keeps grad and B sequential
    for (int i = 0; i < m; i++)
        for (int ki = 0; ki < K; ki++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++)
                sum += self->grad[i*n+j] * B->data[ki*n+j];
            A->grad[i*K+ki] += sum;
        }

    // dB += A^T @ dC  — inner loop over j keeps dC and B->grad sequential
    for (int i = 0; i < m; i++)
        for (int ki = 0; ki < K; ki++) {
            float aik = A->data[i*K+ki];
            for (int j = 0; j < n; j++)
                B->grad[ki*n+j] += aik * self->grad[i*n+j];
        }
}

static void backward_tanh(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = self->rows * self->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_tanh_bwd(self->cuda_data, self->cuda_grad, a->cuda_grad, n); return; }
#endif
    for (int i = 0; i < n; i++) {
        float t = self->data[i];
        a->grad[i] += (1.0f - t * t) * self->grad[i];
    }
}

static void backward_relu(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = self->rows * self->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_relu_bwd(a->cuda_data, self->cuda_grad, a->cuda_grad, n); return; }
#endif
    for (int i = 0; i < n; i++)
        a->grad[i] += self->grad[i] * (a->data[i] > 0.0f ? 1.0f : 0.0f);
}

static void backward_sum(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = a->rows * a->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_sum_bwd(self->cuda_grad, a->cuda_grad, n); return; }
#endif
    float g = self->grad[0];
    for (int i = 0; i < n; i++) a->grad[i] += g;
}

static void backward_mean(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = a->rows * a->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_mean_bwd(self->cuda_grad, a->cuda_grad, n); return; }
#endif
    float g = self->grad[0] / (float)n;
    for (int i = 0; i < n; i++) a->grad[i] += g;
}

static void backward_transpose(Tensor *self) {
    Tensor *a = self->parents[0];
    int r = a->rows, c = a->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        // self is [c × r]; grad_a[j*c+i] += self_grad[i*r+j]
        // which is a transpose of self->cuda_grad into a->cuda_grad
        cuda_transpose_bwd(self->cuda_grad, a->cuda_grad, self->rows, self->cols);
        return;
    }
#endif
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            a->grad[i*c+j] += self->grad[j*r+i];
}

static void backward_scale(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = a->rows * a->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_scale_bwd(self->cuda_grad, self->aux, a->cuda_grad, n); return; }
#endif
    for (int i = 0; i < n; i++)
        a->grad[i] += self->aux * self->grad[i];
}

static void backward_causal_mask(Tensor *self) {
    Tensor *scores = self->parents[0];
    int seq_len = self->rows;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) { cuda_causal_mask_bwd(self->cuda_grad, scores->cuda_grad, seq_len); return; }
#endif
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < seq_len; j++)
            if (j <= i)
                scores->grad[i * seq_len + j] += self->grad[i * seq_len + j];
}

static void backward_layer_norm_rows(Tensor *self) {
    Tensor *a = self->parents[0];
    int rows = self->rows, cols = self->cols;

#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_layer_norm_rows_bwd(self->cuda_data, self->cuda_grad,
                                 self->cuda_cache, a->cuda_grad, rows, cols);
        return;
    }
#endif

    for (int r = 0; r < rows; r++) {
        float *y  = self->data + r * cols;
        float *dy = self->grad + r * cols;
        float *dx = a->grad    + r * cols;
        float inv_std = self->cache[r];

        float mean_dy = 0.0f;
        float mean_dy_y = 0.0f;
        for (int j = 0; j < cols; j++) {
            mean_dy += dy[j];
            mean_dy_y += dy[j] * y[j];
        }
        mean_dy /= (float)cols;
        mean_dy_y /= (float)cols;

        for (int j = 0; j < cols; j++)
            dx[j] += inv_std * (dy[j] - mean_dy - y[j] * mean_dy_y);
    }
}

static void backward_softmax_rows(Tensor *self) {
    Tensor *a = self->parents[0];
    int rows = self->rows, cols = self->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_softmax_rows_bwd(self->cuda_data, self->cuda_grad, a->cuda_grad, rows, cols);
        return;
    }
#endif
    for (int r = 0; r < rows; r++) {
        float *s  = self->data + r * cols;
        float *dy = self->grad + r * cols;
        float *dx = a->grad   + r * cols;
        float dot = 0.0f;
        for (int j = 0; j < cols; j++) dot += dy[j] * s[j];
        for (int i = 0; i < cols; i++) dx[i] += s[i] * (dy[i] - dot);
    }
}

static void backward_cross_entropy(Tensor *self) {
    Tensor *logits  = self->parents[0];
    Tensor *targets = self->parents[1];
    int rows = logits->rows, cols = logits->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_cross_entropy_bwd(self->cuda_cache, targets->cuda_data,
                               self->cuda_grad, logits->cuda_grad, rows, cols);
        return;
    }
#endif
    float *probs = self->cache;
    float scale = self->grad[0] / (float)rows;
    for (int i = 0; i < rows * cols; i++)
        logits->grad[i] += scale * (probs[i] - targets->data[i]);
}

static void backward_slice_cols(Tensor *self) {
    Tensor *a = self->parents[0];
    int col_start = (int)self->aux;
    int rows = self->rows, w = self->cols, a_cols = a->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_slice_cols_bwd(self->cuda_grad, a->cuda_grad, rows, a_cols, col_start, w);
        return;
    }
#endif
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < w; j++)
            a->grad[i * a_cols + col_start + j] += self->grad[i * w + j];
}

static void backward_concat_cols(Tensor *self) {
    int rows = self->rows, total_cols = self->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        int col_offset = 0;
        for (int p = 0; p < self->n_parents; p++) {
            Tensor *part = self->parents[p];
            cuda_concat_cols_bwd(self->cuda_grad, part->cuda_grad,
                                 rows, part->cols, total_cols, col_offset);
            col_offset += part->cols;
        }
        return;
    }
#endif
    int col_offset = 0;
    for (int p = 0; p < self->n_parents; p++) {
        Tensor *part = self->parents[p];
        int w = part->cols;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < w; j++)
                part->grad[i * w + j] += self->grad[i * total_cols + col_offset + j];
        col_offset += w;
    }
}

static void backward_mean_rows(Tensor *self) {
    Tensor *a = self->parents[0];
    int rows = a->rows, cols = a->cols;
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
    int T = self->rows, C = self->cols;
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

// ── Ops ───────────────────────────────────────────────────────────────────────

Tensor *tg_add(Tensor *a, Tensor *b) {
    if (a->rows != b->rows || a->cols != b->cols)
        ovg_fatal("tg_add: shape mismatch [%dx%d] + [%dx%d]",
                  a->rows, a->cols, b->rows, b->cols);
    Tensor *out = make_op(a->rows, a->cols, a, b, backward_add);
    int n = a->rows * a->cols;
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_add_fwd(a->cuda_data, b->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] + b->data[i];
    return out;
}

Tensor *tg_sub(Tensor *a, Tensor *b) {
    if (a->rows != b->rows || a->cols != b->cols)
        ovg_fatal("tg_sub: shape mismatch [%dx%d] - [%dx%d]",
                  a->rows, a->cols, b->rows, b->cols);
    Tensor *out = make_op(a->rows, a->cols, a, b, backward_sub);
    int n = a->rows * a->cols;
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_sub_fwd(a->cuda_data, b->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] - b->data[i];
    return out;
}

Tensor *tg_mul(Tensor *a, Tensor *b) {
    if (a->rows != b->rows || a->cols != b->cols)
        ovg_fatal("tg_mul: shape mismatch [%dx%d] * [%dx%d]",
                  a->rows, a->cols, b->rows, b->cols);
    Tensor *out = make_op(a->rows, a->cols, a, b, backward_mul);
    int n = a->rows * a->cols;
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_mul_fwd(a->cuda_data, b->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] * b->data[i];
    return out;
}

Tensor *tg_pow(Tensor *a, float p) {
    Tensor *out = make_op(a->rows, a->cols, a, NULL, backward_pow);
    out->aux = p;
    int n = a->rows * a->cols;
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_pow_fwd(a->cuda_data, p, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) out->data[i] = powf(a->data[i], p);
    return out;
}

Tensor *tg_matmul(Tensor *a, Tensor *b) {
    if (a->cols != b->rows)
        ovg_fatal("tg_matmul: shape mismatch [%dx%d] @ [%dx%d]",
                  a->rows, a->cols, b->rows, b->cols);
    int m = a->rows, K = a->cols, n = b->cols;
    Tensor *out = make_op(m, n, a, b, backward_matmul);

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        cuda_matmul_forward(a->cuda_data, b->cuda_data, out->cuda_data, m, K, n);
        return out;
    }
#endif

    // i,k,j loop order: both out and b accessed sequentially in j (cache-friendly)
    for (int i = 0; i < m; i++)
        for (int ki = 0; ki < K; ki++) {
            float aik = a->data[i*K+ki];
            for (int j = 0; j < n; j++)
                out->data[i*n+j] += aik * b->data[ki*n+j];
        }
    return out;
}

Tensor *tg_tanh(Tensor *a) {
    Tensor *out = make_op(a->rows, a->cols, a, NULL, backward_tanh);
    int n = a->rows * a->cols;
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_tanh_fwd(a->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) out->data[i] = tanhf(a->data[i]);
    return out;
}

Tensor *tg_relu(Tensor *a) {
    Tensor *out = make_op(a->rows, a->cols, a, NULL, backward_relu);
    int n = a->rows * a->cols;
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_relu_fwd(a->cuda_data, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
    return out;
}

Tensor *tg_sum(Tensor *a) {
    Tensor *out = make_op(1, 1, a, NULL, backward_sum);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_sum_fwd(a->cuda_data, out->cuda_data, a->rows * a->cols); return out; }
#endif
    int n = a->rows * a->cols;
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a->data[i];
    out->data[0] = s;
    return out;
}

Tensor *tg_mean(Tensor *a) {
    Tensor *out = make_op(1, 1, a, NULL, backward_mean);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        // Reuse sum_fwd then scale
        int n = a->rows * a->cols;
        cuda_sum_fwd(a->cuda_data, out->cuda_data, n);
        cuda_scale_fwd(out->cuda_data, 1.0f / (float)n, out->cuda_data, 1);
        return out;
    }
#endif
    int n = a->rows * a->cols;
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a->data[i];
    out->data[0] = s / (float)n;
    return out;
}

Tensor *tg_transpose(Tensor *a) {
    Tensor *out = make_op(a->cols, a->rows, a, NULL, backward_transpose);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_transpose_fwd(a->cuda_data, out->cuda_data, a->rows, a->cols); return out; }
#endif
    // Tiled transpose to keep writes within L1 cache during block iterations
    int R = a->rows, C = a->cols;
    static const int tblock = 32;
    for (int i = 0; i < R; i += tblock)
        for (int j = 0; j < C; j += tblock)
            for (int ii = i; ii < i + tblock && ii < R; ii++)
                for (int jj = j; jj < j + tblock && jj < C; jj++)
                    out->data[jj * R + ii] = a->data[ii * C + jj];
    return out;
}

Tensor *tg_scale(Tensor *a, float s) {
    Tensor *out = make_op(a->rows, a->cols, a, NULL, backward_scale);
    out->aux = s;
    int n = a->rows * a->cols;
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_scale_fwd(a->cuda_data, s, out->cuda_data, n); return out; }
#endif
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] * s;
    return out;
}

Tensor *tg_causal_mask(Tensor *scores) {
    if (scores->rows != scores->cols)
        ovg_fatal("tg_causal_mask: expected square scores matrix, got [%dx%d]",
                  scores->rows, scores->cols);
    int seq_len = scores->rows;
    Tensor *out = make_op(seq_len, seq_len, scores, NULL, backward_causal_mask);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_causal_mask_fwd(scores->cuda_data, out->cuda_data, seq_len); return out; }
#endif
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < seq_len; j++)
            out->data[i * seq_len + j] = (j <= i) ? scores->data[i * seq_len + j] : -1.0e9f;
    return out;
}

Tensor *tg_layer_norm_rows(Tensor *a, float eps) {
    int rows = a->rows, cols = a->cols;
    Tensor *out = make_op(rows, cols, a, NULL, backward_layer_norm_rows);
    out->aux = eps;

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        tg_cuda_alloc_cache(out, rows);   // inv_std: one float per row
        cuda_layer_norm_rows_fwd(a->cuda_data, eps, out->cuda_data,
                                 out->cuda_cache, rows, cols);
        return out;
    }
#endif

    out->cache = malloc(rows * sizeof(float));
    if (!out->cache) ovg_fatal("tg_layer_norm_rows: out of memory");

    for (int r = 0; r < rows; r++) {
        float *x = a->data + r * cols;
        float *y = out->data + r * cols;

        // Single pass: accumulate sum and sum-of-squares simultaneously
        float sum = 0.0f, sum2 = 0.0f;
        for (int j = 0; j < cols; j++) { sum += x[j]; sum2 += x[j] * x[j]; }
        float inv_cols = 1.0f / (float)cols;
        float mean = sum * inv_cols;
        float var  = sum2 * inv_cols - mean * mean;

        float inv_std = 1.0f / sqrtf(var + eps);
        out->cache[r] = inv_std;

        for (int j = 0; j < cols; j++)
            y[j] = (x[j] - mean) * inv_std;
    }
    return out;
}

Tensor *tg_softmax_rows(Tensor *a) {
    int rows = a->rows, cols = a->cols;
    Tensor *out = make_op(rows, cols, a, NULL, backward_softmax_rows);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_softmax_rows_fwd(a->cuda_data, out->cuda_data, rows, cols); return out; }
#endif
    for (int r = 0; r < rows; r++) {
        float *in  = a->data   + r * cols;
        float *res = out->data + r * cols;
        float mx = in[0];
        for (int j = 1; j < cols; j++) if (in[j] > mx) mx = in[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) { res[j] = expf(in[j] - mx); sum += res[j]; }
        for (int j = 0; j < cols; j++) res[j] /= sum;
    }
    return out;
}

Tensor *tg_cross_entropy(Tensor *logits, Tensor *targets) {
    if (logits->rows != targets->rows || logits->cols != targets->cols)
        ovg_fatal("tg_cross_entropy: shape mismatch [%dx%d] vs [%dx%d]",
                  logits->rows, logits->cols, targets->rows, targets->cols);
    int rows = logits->rows, cols = logits->cols;
    Tensor *out = make_op(1, 1, logits, targets, backward_cross_entropy);

#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        tg_cuda_alloc_cache(out, rows * cols);   // probs: R×C floats
        cuda_cross_entropy_fwd(logits->cuda_data, targets->cuda_data,
                               out->cuda_cache, out->cuda_data, rows, cols);
// Copy scalar loss to CPU so the training loop can print it
        tg_from_cuda(out);   // sync scalar loss to host for printing; on_cuda stays 1 so backward uses cuda_cache
        return out;
    }
#endif

    out->cache = malloc(rows * cols * sizeof(float));
    if (!out->cache) ovg_fatal("tg_cross_entropy: out of memory");
    float loss = 0.0f;
    for (int r = 0; r < rows; r++) {
        float *row    = logits->data + r * cols;
        float *prob   = out->cache   + r * cols;
        const float *tgt = targets->data + r * cols;
        float mx = row[0];
        for (int j = 1; j < cols; j++) if (row[j] > mx) mx = row[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) { prob[j] = expf(row[j] - mx); sum += prob[j]; }
        // log-sum-exp computed once per row; normalize probs and accumulate loss together
        float log_sum_exp = logf(sum) + mx;
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < cols; j++) {
            prob[j] *= inv_sum;
            loss -= tgt[j] * (row[j] - log_sum_exp);
        }
    }
    out->data[0] = loss / (float)rows;
    return out;
}

Tensor *tg_slice_cols(Tensor *a, int col_start, int col_end) {
    if (col_start < 0 || col_end > a->cols || col_start >= col_end)
        ovg_fatal("tg_slice_cols: invalid range [%d,%d) for cols=%d",
                  col_start, col_end, a->cols);
    int w = col_end - col_start;
    Tensor *out = make_op(a->rows, w, a, NULL, backward_slice_cols);
    out->aux = (float)col_start;
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) {
        cuda_slice_cols_fwd(a->cuda_data, out->cuda_data, a->rows, a->cols, col_start, w);
        return out;
    }
#endif
    for (int i = 0; i < a->rows; i++)
        for (int j = 0; j < w; j++)
            out->data[i * w + j] = a->data[i * a->cols + col_start + j];
    return out;
}

Tensor *tg_concat_cols(Tensor **parts, int n_parts) {
    if (n_parts < 1 || n_parts > TG_MAX_PARENTS)
        ovg_fatal("tg_concat_cols: n_parts=%d out of range [1,%d]", n_parts, TG_MAX_PARENTS);
    int rows = parts[0]->rows;
    int total_cols = 0;
    for (int i = 0; i < n_parts; i++) {
        if (parts[i]->rows != rows)
            ovg_fatal("tg_concat_cols: row mismatch at part %d: expected %d, got %d",
                      i, rows, parts[i]->rows);
        total_cols += parts[i]->cols;
    }

    Tensor *out = tg_new(rows, total_cols);
    for (int i = 0; i < n_parts; i++) out->parents[i] = parts[i];
    out->n_parents   = n_parts;
    out->backward_fn = backward_concat_cols;

#ifdef OVG_CUDA_ENABLED
    {
        int any_gpu = 0, all_gpu = 1;
        for (int i = 0; i < n_parts; i++) {
            if (parts[i]->on_cuda) any_gpu = 1;
            else                   all_gpu = 0;
        }
        if (any_gpu && !all_gpu)
            ovg_fatal("tg_concat_cols: mixed CUDA/CPU parts");
        if (any_gpu) {
            tg_cuda_alloc(out);
            int col_offset = 0;
            for (int p = 0; p < n_parts; p++) {
                cuda_concat_cols_fwd(parts[p]->cuda_data, out->cuda_data,
                                     rows, parts[p]->cols, total_cols, col_offset);
                col_offset += parts[p]->cols;
            }
            return out;
        }
    }
#endif

    int col_offset = 0;
    for (int p = 0; p < n_parts; p++) {
        int w = parts[p]->cols;
        for (int i = 0; i < rows; i++)
            memcpy(out->data + i * total_cols + col_offset,
                   parts[p]->data + i * w,
                   (size_t)w * sizeof(float));
        col_offset += w;
    }
    return out;
}

Tensor *tg_mean_rows(Tensor *a) {
    int rows = a->rows, cols = a->cols;
    Tensor *out = make_op(1, cols, a, NULL, backward_mean_rows);
#ifdef OVG_CUDA_ENABLED
    if (out->on_cuda) { cuda_mean_rows_fwd(a->cuda_data, out->cuda_data, rows, cols); return out; }
#endif
    for (int j = 0; j < cols; j++) {
        float s = 0.0f;
        for (int i = 0; i < rows; i++) s += a->data[i * cols + j];
        out->data[j] = s / (float)rows;
    }
    return out;
}

static void backward_dropout(Tensor *self) {
    Tensor *a = self->parents[0];
    int n = self->rows * self->cols;
#ifdef OVG_CUDA_ENABLED
    if (self->on_cuda) {
        cuda_dropout_bwd(self->cuda_cache, self->cuda_grad, a->cuda_grad, n);
        return;
    }
#endif
    for (int i = 0; i < n; i++)
        a->grad[i] += self->cache[i] * self->grad[i];
}

// xorshift32: fast non-cryptographic PRNG; much faster than rand() with no lock contention
// Fixed seed: dropout masks are deterministic across runs (intentional for reproducibility).
// To vary masks, seed s_dropout_rng from an external source before the first tg_dropout call.
static uint32_t s_dropout_rng = 0xdeadbeef;
static inline uint32_t xorshift32(void) {
    s_dropout_rng ^= s_dropout_rng << 13;
    s_dropout_rng ^= s_dropout_rng >> 17;
    s_dropout_rng ^= s_dropout_rng << 5;
    return s_dropout_rng;
}

Tensor *tg_dropout(Tensor *a, float p) {
    Tensor *out = make_op(a->rows, a->cols, a, NULL, backward_dropout);
    int n = a->rows * a->cols;

    // Generate CPU mask regardless of GPU/CPU path
    float *mask = malloc((size_t)n * sizeof(float));
    if (!mask) ovg_fatal("tg_dropout: out of memory");

    if (!tg_training || p <= 0.0f) {
        for (int i = 0; i < n; i++) mask[i] = 1.0f;
    } else {
        float inv_keep = 1.0f / (1.0f - p);
        // Scale factor converts uint32 to [0, 1); compare against p threshold
        const float scale = 1.0f / 4294967296.0f;
        for (int i = 0; i < n; i++)
            mask[i] = (xorshift32() * scale) >= p ? inv_keep : 0.0f;
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

    out->cache = mask;
    for (int i = 0; i < n; i++) out->data[i] = a->data[i] * mask[i];
    return out;
}

Tensor *tg_embed(Tensor *weight, const int *token_ids, int seq_len) {
    int V = weight->rows, C = weight->cols;
    for (int t = 0; t < seq_len; t++) {
        if (token_ids[t] < 0 || token_ids[t] >= V)
            ovg_fatal("tg_embed: token id %d out of range [0, %d)", token_ids[t], V);
    }

    Tensor *out = make_op(seq_len, C, weight, NULL, backward_embed);

    // Cache ids as floats — CPU backward reads self->cache, CUDA backward reads cuda_cache
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

    for (int t = 0; t < seq_len; t++) {
        int id = token_ids[t];
        for (int c = 0; c < C; c++)
            out->data[t * C + c] = weight->data[id * C + c];
    }
    return out;
}
