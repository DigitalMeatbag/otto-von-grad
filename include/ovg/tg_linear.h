#ifndef TG_LINEAR_H
#define TG_LINEAR_H

#include "tg_ops.h"

typedef struct {
    Tensor *W;       /* [n_in, n_out] */
    Tensor *B;       /* [1, n_out]; expanded to the input's leading dims at forward */
    int n_in, n_out;
} TgLinear;

/* W ~ randn * w_scale, B = 0; both persistent. Fatal if n_in <= 0 or n_out <= 0. */
TgLinear  tg_linear_create(int n_in, int n_out, float w_scale);
void      tg_linear_free(TgLinear *l);

/* x: [..., n_in] with ndim 2..TG_MAX_DIMS. Returns x @ W + B with shape [..., n_out].
   Fatal if x->shape[ndim-1] != n_in. F32 only (BF16 callers cast around it).
   The intermediate x @ W and the expanded bias are ordinary graph nodes that
   tg_free_graph frees. */
Tensor   *tg_linear_forward(TgLinear *l, Tensor *x);

/* Returns a malloc'd {W, B} array; *n_out = 2. Caller frees the array, not the tensors. */
Tensor  **tg_linear_params(TgLinear *l, int *n_out);

#endif
