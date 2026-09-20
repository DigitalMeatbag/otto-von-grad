#ifndef TG_MLP_H
#define TG_MLP_H

#include "tg_ops.h"

typedef struct {
    Tensor *W;       // [n_in  x n_out]
    Tensor *B;       // [batch x n_out]  (no broadcasting yet)
    int n_in, n_out, batch;
} TgLinear;

// Allocate W (randn * w_scale) and B (zeros)
TgLinear  tg_linear_create(int n_in, int n_out, int batch, float w_scale);
void      tg_linear_free(TgLinear *l);

// Returns Z = X@W+B; writes the intermediate X@W into *xw_out (caller must free both)
Tensor   *tg_linear_forward(TgLinear *l, Tensor *x, Tensor **xw_out);

// Returns a malloc'd {W, B} array; caller frees the array (not the tensors)
Tensor  **tg_linear_params(TgLinear *l, int *n_out);

#endif
