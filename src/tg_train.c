#include "tg_train.h"
#include "ovg_error.h"

#include <math.h>
#include <stdlib.h>

#ifdef OVG_CUDA_ENABLED
#include "tg_cuda.h"
#include "cuda_ops.h"
#endif

int tg_training = 1;

/* Depth-first topo sort using the tensor's own visited flag (O(n), not O(n²)). */
static void topo_sort(Tensor *t, Tensor **topo, int *n) {
    if (t->visited) return;
    if (*n >= TG_MAX_GRAPH)
        ovg_fatal("topo_sort: graph exceeds TG_MAX_GRAPH=%d; increase TG_MAX_GRAPH",
                  TG_MAX_GRAPH);
    t->visited = 1;
    for (int i = 0; i < t->n_parents; i++)
        topo_sort(t->parents[i], topo, n);
    topo[(*n)++] = t;
}

static void clear_visited(Tensor **topo, int n) {
    for (int i = 0; i < n; i++) topo[i]->visited = 0;
}

void tg_sgd_step(Tensor **params, int n_params, float lr) {
#ifdef OVG_CUDA_ENABLED
    for (int p = 0; p < n_params; p++) {
        if (params[p]->on_cuda)
            ovg_fatal("tg_sgd_step: CUDA tensors not supported; use tg_adam_step_gpu");
    }
#endif
    for (int p = 0; p < n_params; p++) {
        Tensor *t = params[p];
        int n = t->rows * t->cols;
        for (int i = 0; i < n; i++)
            t->data[i] -= lr * t->grad[i];
    }
}

void tg_adam_step(Tensor **params, float **m, float **v, int n_params,
                  float lr, int step, float beta1, float beta2, float eps) {
#ifdef OVG_CUDA_ENABLED
    for (int p = 0; p < n_params; p++) {
        if (params[p]->on_cuda)
            ovg_fatal("tg_adam_step: CUDA tensors not supported; use tg_adam_step_gpu");
    }
#endif
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    for (int p = 0; p < n_params; p++) {
        Tensor *t = params[p];
        int n = t->rows * t->cols;
        for (int i = 0; i < n; i++) {
            float g = t->grad[i];
            m[p][i] = beta1 * m[p][i] + (1.0f - beta1) * g;
            v[p][i] = beta2 * v[p][i] + (1.0f - beta2) * g * g;
            t->data[i] -= lr * (m[p][i] / bc1) / (sqrtf(v[p][i] / bc2) + eps);
        }
    }
}

float tg_clip_grad_norm(Tensor **params, int n_params, float max_norm, float eps) {
    if (!params || n_params < 0 || max_norm <= 0.0f || eps < 0.0f)
        ovg_fatal("tg_clip_grad_norm: invalid arguments");

    float sumsq = 0.0f;
#ifdef OVG_CUDA_ENABLED
    int any_gpu = 0, any_cpu = 0;
    float *gpu_sumsq = NULL;
    for (int p = 0; p < n_params; p++) {
        if (params[p]->on_cuda) any_gpu = 1;
        else any_cpu = 1;
    }
    if (any_gpu && any_cpu)
        ovg_fatal("tg_clip_grad_norm: mixed CUDA/CPU params");
    if (any_gpu) {
        gpu_sumsq = tg_cuda_malloc_floats(1);
        tg_cuda_zero_float(gpu_sumsq);
        for (int p = 0; p < n_params; p++) {
            Tensor *t = params[p];
            cuda_grad_sumsq(t->cuda_grad, gpu_sumsq, t->rows * t->cols);
        }
        /* Clip and scale entirely on GPU — no CPU readback. */
        for (int p = 0; p < n_params; p++) {
            Tensor *t = params[p];
            cuda_clip_scale_grad(gpu_sumsq, max_norm, eps,
                                 t->cuda_grad, t->rows * t->cols);
        }
        tg_cuda_free_floats(gpu_sumsq);
        return 0.0f;  /* norm not read back to CPU in CUDA path */
    } else {
#endif
    for (int p = 0; p < n_params; p++) {
        Tensor *t = params[p];
        int n = t->rows * t->cols;
        for (int i = 0; i < n; i++)
            sumsq += t->grad[i] * t->grad[i];
    }
#ifdef OVG_CUDA_ENABLED
    }
#endif

    float norm = sqrtf(sumsq);
    if (norm > max_norm) {
        float scale = max_norm / (norm + eps);
        for (int p = 0; p < n_params; p++) {
            Tensor *t = params[p];
            int n = t->rows * t->cols;
            for (int i = 0; i < n; i++)
                t->grad[i] *= scale;
        }
    }
    return norm;
}

#ifdef OVG_CUDA_ENABLED
void tg_adam_step_gpu(Tensor **params, float **m_gpu, float **v_gpu, int n_params,
                      float lr, int step, float beta1, float beta2, float eps) {
    float bc1 = 1.0f - powf(beta1, (float)step);
    float bc2 = 1.0f - powf(beta2, (float)step);
    for (int p = 0; p < n_params; p++) {
        Tensor *t = params[p];
        int n = t->rows * t->cols;
        cuda_adam_step(t->cuda_data, m_gpu[p], v_gpu[p], t->cuda_grad,
                       n, lr, bc1, bc2, beta1, beta2, eps);
    }
}
#endif

void tg_free_graph(Tensor *root) {
    Tensor *topo[TG_MAX_GRAPH];
    int n = 0;
    topo_sort(root, topo, &n);
    clear_visited(topo, n);

    for (int i = 0; i < n; i++) {
        if (!topo[i]->persistent) {
#ifdef OVG_CUDA_ENABLED
            tg_cuda_free(topo[i]);
#endif
            tg_free(topo[i]);
        }
    }
}

void tg_backward(Tensor *root) {
    Tensor *topo[TG_MAX_GRAPH];
    int n = 0;
    topo_sort(root, topo, &n);
    clear_visited(topo, n);

    for (int i = 0; i < n; i++) {
        int sz = topo[i]->rows * topo[i]->cols;
#ifdef OVG_CUDA_ENABLED
        if (topo[i]->on_cuda) {
            tg_cuda_zero_grad(topo[i]);
        } else {
#endif
        for (int j = 0; j < sz; j++) topo[i]->grad[j] = 0.0f;
#ifdef OVG_CUDA_ENABLED
        }
#endif
    }

#ifdef OVG_CUDA_ENABLED
    if (root->on_cuda) {
        tg_cuda_set_grad_scalar(root, 1.0f);
    } else {
#endif
    for (int i = 0; i < root->rows * root->cols; i++) root->grad[i] = 1.0f;
#ifdef OVG_CUDA_ENABLED
    }
#endif

    for (int i = n - 1; i >= 0; i--)
        if (topo[i]->backward_fn) topo[i]->backward_fn(topo[i]);
}

void tg_zero_grads(Tensor **params, int n_params) {
    for (int p = 0; p < n_params; p++) {
        Tensor *t = params[p];
        int sz = t->rows * t->cols;
#ifdef OVG_CUDA_ENABLED
        if (t->on_cuda) {
            tg_cuda_zero_grad(t);
            continue;
        }
#endif
        for (int j = 0; j < sz; j++) t->grad[j] = 0.0f;
    }
}

void tg_backward_accum(Tensor *root) {
    Tensor *topo[TG_MAX_GRAPH];
    int n = 0;
    topo_sort(root, topo, &n);
    clear_visited(topo, n);

    /* No zero-grad phase: non-persistent intermediates are fresh calloc
       allocations each forward pass (grads start at 0); persistent param
       grads are left intact so successive calls accumulate. */
#ifdef OVG_CUDA_ENABLED
    if (root->on_cuda) {
        tg_cuda_set_grad_scalar(root, 1.0f);
    } else {
#endif
    for (int i = 0; i < root->rows * root->cols; i++) root->grad[i] = 1.0f;
#ifdef OVG_CUDA_ENABLED
    }
#endif

    for (int i = n - 1; i >= 0; i--)
        if (topo[i]->backward_fn) topo[i]->backward_fn(topo[i]);
}
