#ifndef TG_POOL_H
#define TG_POOL_H

#include "tg_ops.h"

/* Token pooling: encoder output [B, T, C] → one vector per sequence [B, C].
   Both take 3D input only (an encoder output is always 3D) and are
   compositions of existing ops, so every node has a CPU path, a CUDA path,
   and a paired backward. */

/* enc: [B, T, C]. Token 0 of each sequence → [B, C]. Fatal if enc->ndim != 3.
   Assumes the model was built with a CLS token at position 0; the library
   cannot check this — on a CLS-less model it silently returns the first patch. */
Tensor *tg_pool_cls(Tensor *enc);

/* enc: [B, T, C]. Mean over T → [B, C] (CLS included if present; slice first
   for a patch-only mean). Fatal if enc->ndim != 3. */
Tensor *tg_pool_mean_tokens(Tensor *enc);

#endif
