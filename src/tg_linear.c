#include "tg_linear.h"
#include "ovg_error.h"
#include <stdlib.h>

TgLinear tg_linear_create(int n_in, int n_out, float w_scale) {
    if (n_in <= 0 || n_out <= 0)
        ovg_fatal("tg_linear_create: n_in (%d) and n_out (%d) must be positive", n_in, n_out);

    TgLinear l;
    l.n_in  = n_in;
    l.n_out = n_out;
    int wshape[2] = {n_in, n_out};
    int bshape[2] = {1,    n_out};
    l.W = tg_new(2, wshape);  tg_fill_randn(l.W, w_scale);  l.W->persistent = 1;
    l.B = tg_new(2, bshape);  tg_fill(l.B, 0.0f);           l.B->persistent = 1;
    return l;
}

void tg_linear_free(TgLinear *l) {
    tg_free(l->W);
    tg_free(l->B);
}

Tensor *tg_linear_forward(TgLinear *l, Tensor *x) {
    if (!l || !x)
        ovg_fatal("tg_linear_forward: NULL argument");
    if (x->ndim < 2 || x->ndim > TG_MAX_DIMS)
        ovg_fatal("tg_linear_forward: ndim %d not in [2, %d]", x->ndim, TG_MAX_DIMS);
    if (x->shape[x->ndim - 1] != l->n_in)
        ovg_fatal("tg_linear_forward: last dim (%d) != n_in (%d)",
                  x->shape[x->ndim - 1], l->n_in);
    if (x->dtype != TG_DTYPE_F32)
        ovg_fatal("tg_linear_forward: F32 only; cast BF16 inputs first");

    Tensor *xw = tg_matmul(x, l->W);   /* [..., n_out] */

    /* Make the bias explicit: [1, n_out] → [1, ..., 1, n_out] → tiled over each
       leading axis of xw. For 2D input B is already [1, n_out], so no reshape. */
    Tensor *bias = l->B;
    if (xw->ndim > 2) {
        int bshape[TG_MAX_DIMS];
        for (int i = 0; i < xw->ndim - 1; i++) bshape[i] = 1;
        bshape[xw->ndim - 1] = l->n_out;
        bias = tg_reshape(bias, xw->ndim, bshape);
    }
    for (int i = 0; i < xw->ndim - 1; i++)
        bias = tg_expand_dim(bias, i, xw->shape[i]);

    return tg_add(xw, bias);
}

Tensor **tg_linear_params(TgLinear *l, int *n_out) {
    Tensor **p = malloc(2 * sizeof(Tensor *));
    if (!p) ovg_fatal("tg_linear_params: out of memory");
    p[0] = l->W;
    p[1] = l->B;
    *n_out = 2;
    return p;
}
