/* tg_optim.c — TgAdam: optimizer state + the step body (zero / accumulate / update).
 *
 * Wraps the raw tg_adam_step (CPU) / tg_adam_step_gpu (device) in tg_train.c.
 * The moments follow the params: host buffers when the params are on the host,
 * device buffers (tg_cuda_malloc_floats) when they are on the device.
 */

#include "tg_optim.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "ovg_error.h"

#include <stdlib.h>
#include <string.h>

#ifdef OVG_CUDA_ENABLED
#  include "tg_cuda_internal.h"
#endif

TgAdam tg_adam_create(Tensor **params, int n_params, float beta1, float beta2, float eps) {
    if (!params || n_params <= 0)
        ovg_fatal("tg_adam_create: n_params must be > 0 (got %d)", n_params);

    TgAdam opt;
    memset(&opt, 0, sizeof opt);
    opt.params   = params;
    opt.n_params = n_params;
    opt.beta1    = beta1;
    opt.beta2    = beta2;
    opt.eps      = eps;
    opt.step     = 0;

    int any_gpu = 0, any_cpu = 0;
    for (int p = 0; p < n_params; p++) {
        if (params[p]->on_cuda) any_gpu = 1;
        else any_cpu = 1;
    }
    if (any_gpu && any_cpu)
        ovg_fatal("tg_adam_create: params mix CPU and CUDA tensors");
    opt.on_cuda = any_gpu;

    opt.m = calloc((size_t)n_params, sizeof(float *));
    opt.v = calloc((size_t)n_params, sizeof(float *));
    if (!opt.m || !opt.v)
        ovg_fatal("tg_adam_create: out of memory");

    for (int p = 0; p < n_params; p++) {
        int nel = tg_numel(params[p]);
#ifdef OVG_CUDA_ENABLED
        if (opt.on_cuda) {
            opt.m[p] = tg_cuda_malloc_floats(nel);   /* zeroed on device */
            opt.v[p] = tg_cuda_malloc_floats(nel);
            continue;
        }
#endif
        opt.m[p] = calloc((size_t)nel, sizeof(float));
        opt.v[p] = calloc((size_t)nel, sizeof(float));
        if (!opt.m[p] || !opt.v[p])
            ovg_fatal("tg_adam_create: out of memory");
    }
    return opt;
}

void tg_adam_free(TgAdam *opt) {
    if (!opt || !opt->m) return;
    for (int p = 0; p < opt->n_params; p++) {
#ifdef OVG_CUDA_ENABLED
        if (opt->on_cuda) {
            tg_cuda_free_floats(opt->m[p]);
            tg_cuda_free_floats(opt->v[p]);
            continue;
        }
#endif
        free(opt->m[p]);
        free(opt->v[p]);
    }
    free(opt->m);
    free(opt->v);
    opt->m = opt->v = NULL;
    opt->n_params = 0;
    opt->step = 0;
}

void tg_adam_reset(TgAdam *opt) {
    for (int p = 0; p < opt->n_params; p++) {
        int nel = tg_numel(opt->params[p]);
#ifdef OVG_CUDA_ENABLED
        if (opt->on_cuda) {
            tg_cuda_zero_floats(opt->m[p], nel);
            tg_cuda_zero_floats(opt->v[p], nel);
            continue;
        }
#endif
        memset(opt->m[p], 0, (size_t)nel * sizeof(float));
        memset(opt->v[p], 0, (size_t)nel * sizeof(float));
    }
    opt->step = 0;
}

float tg_adam_update(TgAdam *opt, float lr, float max_grad_norm) {
    float norm = 0.0f;
    /* tg_clip_grad_norm fatals on max_norm <= 0, so only call it when clipping. */
    if (max_grad_norm > 0.0f)
        norm = tg_clip_grad_norm(opt->params, opt->n_params, max_grad_norm, 1e-6f);

    opt->step += 1;
#ifdef OVG_CUDA_ENABLED
    if (opt->on_cuda) {
        tg_adam_step_gpu(opt->params, opt->m, opt->v, opt->n_params,
                         lr, opt->step, opt->beta1, opt->beta2, opt->eps);
        return norm;
    }
#endif
    tg_adam_step(opt->params, opt->m, opt->v, opt->n_params,
                 lr, opt->step, opt->beta1, opt->beta2, opt->eps);
    return norm;
}

void tg_adam_zero_grads(TgAdam *opt) {
    tg_zero_grads(opt->params, opt->n_params);
}

void tg_adam_accumulate(TgAdam *opt, Tensor *loss, int n_micro) {
    (void)opt;
    if (n_micro < 1)
        ovg_fatal("tg_adam_accumulate: n_micro must be >= 1 (got %d)", n_micro);
    if (!loss)
        ovg_fatal("tg_adam_accumulate: loss is NULL");
    if (n_micro > 1)
        loss = tg_scale(loss, 1.0f / (float)n_micro);
    tg_backward_accum(loss);
    tg_free_graph(loss);
}
