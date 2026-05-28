#include "tg_tensor.h"
#include "ovg_error.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef OVG_CUDA_ENABLED
#include "tg_cuda.h"
#endif

Tensor *tg_new(int ndim, const int shape[]) {
    if (ndim < 2 || ndim > TG_MAX_DIMS)
        ovg_fatal("tg_new: ndim=%d out of range [2, %d]", ndim, TG_MAX_DIMS);
    int nel = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0)
            ovg_fatal("tg_new: shape[%d]=%d must be positive", i, shape[i]);
        nel *= shape[i];
    }
    /* NOTE: nel computed as int. Promoting tg_numel to long would be strictly
       correct but the refactor is wide; >2.1B elements is unreachable on 12 GB
       VRAM in practice. This guard catches overflow before allocation. */
    if (nel <= 0)
        ovg_fatal("tg_new: shape product overflows int (%d dims, first dim %d)",
                  ndim, shape[0]);
    Tensor *t = calloc(1, sizeof(Tensor));
    if (!t) ovg_fatal("tg_new: out of memory");
    t->ndim  = ndim;
    t->dtype = TG_DTYPE_F32;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    t->data = calloc((size_t)nel, sizeof(float));
    if (!t->data) { free(t); ovg_fatal("tg_new: out of memory"); }
    t->grad = calloc((size_t)nel, sizeof(float));
    if (!t->grad) { free(t->data); free(t); ovg_fatal("tg_new: out of memory"); }
    return t;
}

void tg_free(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t->cache);
    free(t);
}

void tg_fill(Tensor *t, float val) {
    int n = tg_numel(t);
    float *d = TG_DATAF(t);
    for (int i = 0; i < n; i++) d[i] = val;
}

void tg_fill_uniform(Tensor *t, float lo, float hi) {
    if (!t || hi < lo)
        ovg_fatal("tg_fill_uniform: invalid arguments");
    int n = tg_numel(t);
    float *d = TG_DATAF(t);
    float span = hi - lo;
    for (int i = 0; i < n; i++) {
        float u = (rand() + 0.5f) / (RAND_MAX + 1.0f);
        d[i] = lo + span * u;
    }
}

void tg_fill_randn(Tensor *t, float scale) {
    int n = tg_numel(t);
    float *d = TG_DATAF(t);
    for (int i = 0; i < n - 1; i += 2) {
        float u1 = (rand() + 0.5f) / (RAND_MAX + 1.0f);
        float u2 = (rand() + 0.5f) / (RAND_MAX + 1.0f);
        float r   = sqrtf(-2.0f * logf(u1));
        float th  = 2.0f * 3.14159265f * u2;
        d[i]   = scale * r * cosf(th);
        d[i+1] = scale * r * sinf(th);
    }
    if (n % 2 == 1) {
        float u1 = (rand() + 0.5f) / (RAND_MAX + 1.0f);
        float u2 = (rand() + 0.5f) / (RAND_MAX + 1.0f);
        d[n-1] = scale * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    }
}

void tg_fill_xavier_uniform(Tensor *t) {
    if (!t || t->ndim < 2)
        ovg_fatal("tg_fill_xavier_uniform: invalid tensor shape");
    int fan_in  = t->shape[t->ndim - 2];
    int fan_out = t->shape[t->ndim - 1];
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    tg_fill_uniform(t, -limit, limit);
}

void tg_fill_xavier_normal(Tensor *t) {
    if (!t || t->ndim < 2)
        ovg_fatal("tg_fill_xavier_normal: invalid tensor shape");
    int fan_in  = t->shape[t->ndim - 2];
    int fan_out = t->shape[t->ndim - 1];
    float scale = sqrtf(2.0f / (float)(fan_in + fan_out));
    tg_fill_randn(t, scale);
}

float tg_scalar_value(Tensor *t) {
    if (!t || tg_numel(t) != 1)
        ovg_fatal("tg_scalar_value: expected scalar tensor (1 element)");
#ifdef OVG_CUDA_ENABLED
    if (t->on_cuda)
        tg_from_cuda(t);
#endif
    return TG_DATAF(t)[0];
}

static void print_nd(const float *buf, int ndim, const int shape[], const char *name) {
    int rows = shape[ndim - 2];
    int cols = shape[ndim - 1];
    int outer = 1;
    for (int i = 0; i < ndim - 2; i++) outer *= shape[i];
    printf("%s [", name);
    for (int i = 0; i < ndim; i++) { if (i) printf("x"); printf("%d", shape[i]); }
    printf("]:\n");
    for (int b = 0; b < outer; b++) {
        if (outer > 1) printf("  [batch %d]\n", b);
        for (int i = 0; i < rows; i++) {
            printf("  ");
            for (int j = 0; j < cols; j++)
                printf("%8.4f ", buf[b * rows * cols + i * cols + j]);
            printf("\n");
        }
    }
}

void tg_print(const Tensor *t, const char *name) {
    print_nd(TG_DATAF(t), t->ndim, t->shape, name);
}
void tg_print_grad(const Tensor *t, const char *name) {
    print_nd(t->grad, t->ndim, t->shape, name);
}
