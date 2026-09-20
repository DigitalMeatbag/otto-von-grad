#ifndef TG_CUDA_H
#define TG_CUDA_H

#ifdef OVG_CUDA_ENABLED

#include "tg_tensor.h"

// Public CUDA surface: moving a tensor between host and device, and raw
// device float buffers for optimizer state (see tg_adam_step_gpu).
// Implemented in tg_cuda.cu; safe to include from plain C files.
// Op-level plumbing (cache buffers, grad zeroing) lives in src/tg_cuda_internal.h.

#ifdef __cplusplus
extern "C" {
#endif

// Allocate device data+grad and upload current host values.
// After call: t->on_cuda=1, cuda_data/cuda_grad valid, host mirrors stale.
void tg_to_cuda(Tensor *t);

// Download device data+grad back to host buffers.
// After call: host data/grad current; on_cuda remains 1.
void tg_from_cuda(Tensor *t);

// Free device memory and clear the on_cuda flag.
// Call before tg_free to avoid device memory leak.
void tg_cuda_free(Tensor *t);

// Allocate n floats on device (zeroed). Used for GPU optimizer moment arrays.
float *tg_cuda_malloc_floats(int n);

// Free a device float buffer previously allocated with tg_cuda_malloc_floats.
void tg_cuda_free_floats(float *p);

#ifdef __cplusplus
}
#endif

#endif // OVG_CUDA_ENABLED
#endif // TG_CUDA_H
