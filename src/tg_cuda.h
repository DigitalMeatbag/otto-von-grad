#ifndef TG_CUDA_H
#define TG_CUDA_H

#ifdef OVG_CUDA_ENABLED

#include "tg_tensor.h"

// All functions are implemented in tg_cuda.cu (CUDA translation unit).
// This header is safe to include from plain C files.

#ifdef __cplusplus
extern "C" {
#endif

// Allocate device data+grad and upload current host values.
// After call: t->on_cuda=1, cuda_data/cuda_grad valid, host mirrors stale.
void tg_to_cuda(Tensor *t);

// Download device data+grad back to host buffers.
// After call: host data/grad current; on_cuda remains 1.
void tg_from_cuda(Tensor *t);

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

// Free device memory and clear the on_cuda flag.
// Call before tg_free to avoid device memory leak.
void tg_cuda_free(Tensor *t);

// Allocate n floats on device (zeroed). Used for GPU optimizer moment arrays.
float *tg_cuda_malloc_floats(int n);

// Free a device float buffer previously allocated with tg_cuda_malloc_floats.
void tg_cuda_free_floats(float *p);

// Small scalar helpers for CUDA-side utility reductions.
void  tg_cuda_zero_float(float *p);
float tg_cuda_read_float(float *p);

// Set t->cuda_grad[0] = val (used to seed tg_backward for the root loss tensor).
void tg_cuda_set_grad_scalar(Tensor *t, float val);

#ifdef __cplusplus
}
#endif

#endif // OVG_CUDA_ENABLED
#endif // TG_CUDA_H
