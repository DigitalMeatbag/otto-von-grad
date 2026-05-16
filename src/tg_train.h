#ifndef TG_TRAIN_H
#define TG_TRAIN_H

#include "tg_tensor.h"

/* Set to 0 before inference to disable dropout; 1 (default) during training. */
extern int tg_training;

void tg_sgd_step(Tensor **params, int n_params, float lr);
void tg_adam_step(Tensor **params, float **m, float **v, int n_params,
                  float lr, int step, float beta1, float beta2, float eps);

// Standard backward: zeros all grads in the graph, then runs reverse-mode AD.
void tg_backward(Tensor *root);

// Zeros grad (CPU and GPU) for each param in params[0..n-1].
// Call once before a gradient-accumulation batch, then use tg_backward_accum.
void tg_zero_grads(Tensor **params, int n_params);

// Backward pass that skips the zero-grad phase.
// Non-persistent intermediate tensors are fresh calloc allocations each forward
// pass, so their grads are already zero.  Persistent param grads accumulate
// across calls — caller must zero them first with tg_zero_grads.
void tg_backward_accum(Tensor *root);

// Traverses the computation graph rooted at `root` and frees every non-persistent
// tensor. Call after tg_backward + optimizer step instead of tg_free(root).
void tg_free_graph(Tensor *root);

#ifdef OVG_CUDA_ENABLED
// GPU Adam: m_gpu and v_gpu are device pointers (float*), one per param tensor.
// Allocate with cudaMalloc and zero with cudaMemset before the first call.
void tg_adam_step_gpu(Tensor **params, float **m_gpu, float **v_gpu, int n_params,
                      float lr, int step, float beta1, float beta2, float eps);
#endif

#endif
