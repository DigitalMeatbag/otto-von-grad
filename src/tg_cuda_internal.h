#ifndef TG_CUDA_INTERNAL_H
#define TG_CUDA_INTERNAL_H

#ifdef OVG_CUDA_ENABLED

#include "tg_tensor.h"
#include "tg_cuda.h"

// Device-side plumbing used by tg_ops.c, tg_train.c, tg_optim.c and
// tg_checkpoint.c. Not part of the public surface; consumers use tg_cuda.h.
// Implemented in tg_cuda.cu.

#ifdef __cplusplus
extern "C" {
#endif

// Allocate device data+grad without uploading (output tensors created by GPU ops).
// Idempotent: safe to call if already allocated.
void tg_cuda_alloc(Tensor *t);

// Allocate n-float scratch buffer on device for op caches (inv_std, probs, mask).
// Idempotent: if cuda_cache is already non-NULL it is freed and reallocated.
void tg_cuda_alloc_cache(Tensor *t, int n);

// Copy n floats from host_src into t->cuda_cache (must be allocated first).
void tg_cuda_upload_cache(Tensor *t, const float *host_src, int n);

// Zero the device grad buffer (used by tg_backward for GPU tensors).
void tg_cuda_zero_grad(Tensor *t);

// Small scalar helpers for CUDA-side utility reductions.
void  tg_cuda_zero_float(float *p);
float tg_cuda_read_float(float *p);

// Raw device float buffers (optimizer moments from tg_cuda_malloc_floats):
// bulk host<->device copies and zeroing. Synchronous.
void tg_cuda_upload_floats(float *dst_dev, const float *src_host, int n);
void tg_cuda_download_floats(float *dst_host, const float *src_dev, int n);
void tg_cuda_zero_floats(float *p, int n);

// Set t->cuda_grad[0] = val (used to seed tg_backward for the root loss tensor).
void tg_cuda_set_grad_scalar(Tensor *t, float val);

#ifdef __cplusplus
}
#endif

#endif // OVG_CUDA_ENABLED
#endif // TG_CUDA_INTERNAL_H
