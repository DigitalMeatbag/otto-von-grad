#ifndef TG_OPTIM_H
#define TG_OPTIM_H

#include "tg_tensor.h"

/* Adam optimizer state. The struct path (tg_adam_update) is numerically identical
   to the raw tg_adam_step / tg_adam_step_gpu it dispatches to; the raw functions
   remain public in tg_train.h. */
typedef struct {
    Tensor **params;     /* borrowed; the caller keeps the array and the tensors alive */
    int      n_params;
    float  **m, **v;     /* per-param first/second moments; host buffers, or device buffers when on_cuda */
    int      on_cuda;    /* 1 if params live on the device; fixed at create */
    int      step;       /* completed updates; 0 after create/reset */
    float    beta1, beta2, eps;
} TgAdam;

/* Allocates zeroed m/v matching each param's numel, on the device the params are on.
   Fatal if n_params <= 0, if params mix CPU and CUDA, or on allocation failure.
   Create AFTER moving params to their final device (tg_to_cuda). */
TgAdam tg_adam_create(Tensor **params, int n_params, float beta1, float beta2, float eps);
void   tg_adam_free(TgAdam *opt);       /* frees m/v (host or device); does not touch params */
void   tg_adam_reset(TgAdam *opt);      /* zeroes m/v, step = 0 */

/* One optimizer step over accumulated grads.
   step += 1, then Adam with bias correction at t = step.
   If max_grad_norm > 0, clips first via tg_clip_grad_norm(params, n, max_grad_norm, 1e-6f).
   Returns the pre-clip gradient norm on CPU; returns 0.0f on CUDA or when not clipping
   (no host sync is added). */
float  tg_adam_update(TgAdam *opt, float lr, float max_grad_norm);

/* Accumulation helpers. A step is:
     tg_adam_zero_grads(&opt);
     for each micro-step: loss = <caller forward>; tg_adam_accumulate(&opt, loss, n_micro);
     tg_adam_update(&opt, lr, max_grad_norm);
   tg_adam_accumulate scales loss by 1/n_micro when n_micro > 1, runs tg_backward_accum,
   then tg_free_graph(loss). loss must be a non-persistent root; read its value
   (tg_scalar_value) BEFORE calling, because the graph is freed on return.
   n_micro < 1 is fatal. */
void   tg_adam_zero_grads(TgAdam *opt);                          /* tg_zero_grads over params */
void   tg_adam_accumulate(TgAdam *opt, Tensor *loss, int n_micro);

#endif /* TG_OPTIM_H */
