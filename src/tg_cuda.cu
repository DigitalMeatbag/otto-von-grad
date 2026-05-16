#ifdef OVG_CUDA_ENABLED

#include "tg_cuda.h"
#include "tg_tensor.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

void tg_to_cuda(Tensor *t) {
    size_t bytes = (size_t)t->rows * t->cols * sizeof(float);
    if (!t->cuda_data) {
        CUDA_CHECK(cudaMalloc(&t->cuda_data, bytes));
    }
    if (!t->cuda_grad) {
        CUDA_CHECK(cudaMalloc(&t->cuda_grad, bytes));
    }
    CUDA_CHECK(cudaMemcpy(t->cuda_data, t->data, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t->cuda_grad, t->grad, bytes, cudaMemcpyHostToDevice));
    t->on_cuda = 1;
}

void tg_from_cuda(Tensor *t) {
    if (!t->on_cuda) return;
    size_t bytes = (size_t)t->rows * t->cols * sizeof(float);
    CUDA_CHECK(cudaMemcpy(t->data, t->cuda_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(t->grad, t->cuda_grad, bytes, cudaMemcpyDeviceToHost));
}

void tg_cuda_alloc(Tensor *t) {
    size_t bytes = (size_t)t->rows * t->cols * sizeof(float);
    if (!t->cuda_data) {
        CUDA_CHECK(cudaMalloc(&t->cuda_data, bytes));
    }
    if (!t->cuda_grad) {
        CUDA_CHECK(cudaMalloc(&t->cuda_grad, bytes));
    }
    CUDA_CHECK(cudaMemset(t->cuda_data, 0, bytes));
    CUDA_CHECK(cudaMemset(t->cuda_grad, 0, bytes));
    t->on_cuda = 1;
}

void tg_cuda_alloc_cache(Tensor *t, int n) {
    if (t->cuda_cache) { cudaFree(t->cuda_cache); t->cuda_cache = NULL; }
    CUDA_CHECK(cudaMalloc(&t->cuda_cache, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemset(t->cuda_cache, 0, (size_t)n * sizeof(float)));
}

void tg_cuda_upload_cache(Tensor *t, const float *host_src, int n) {
    CUDA_CHECK(cudaMemcpy(t->cuda_cache, host_src, (size_t)n * sizeof(float),
                          cudaMemcpyHostToDevice));
}

void tg_cuda_zero_grad(Tensor *t) {
    if (!t->on_cuda || !t->cuda_grad) return;
    size_t bytes = (size_t)t->rows * t->cols * sizeof(float);
    CUDA_CHECK(cudaMemset(t->cuda_grad, 0, bytes));
}

void tg_cuda_free(Tensor *t) {
    if (t->cuda_data)  { cudaFree(t->cuda_data);  t->cuda_data  = NULL; }
    if (t->cuda_grad)  { cudaFree(t->cuda_grad);  t->cuda_grad  = NULL; }
    if (t->cuda_cache) { cudaFree(t->cuda_cache); t->cuda_cache = NULL; }
    t->on_cuda = 0;
}

float *tg_cuda_malloc_floats(int n) {
    float *p = NULL;
    CUDA_CHECK(cudaMalloc(&p, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMemset(p, 0, (size_t)n * sizeof(float)));
    return p;
}

void tg_cuda_free_floats(float *p) {
    if (p) cudaFree(p);
}

void tg_cuda_set_grad_scalar(Tensor *t, float val) {
    CUDA_CHECK(cudaMemcpy(t->cuda_grad, &val, sizeof(float), cudaMemcpyHostToDevice));
}

#endif // OVG_CUDA_ENABLED
