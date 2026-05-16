#include "tg_mlp.h"
#include <stdlib.h>
#include <stdio.h>

TgLinear tg_linear_create(int n_in, int n_out, int batch, float w_scale) {
    TgLinear l;
    l.n_in  = n_in;
    l.n_out = n_out;
    l.batch = batch;
    l.W = tg_new(n_in,  n_out);  tg_fill_randn(l.W, w_scale);  l.W->persistent = 1;
    l.B = tg_new(batch, n_out);  tg_fill(l.B, 0.0f);           l.B->persistent = 1;
    return l;
}

void tg_linear_free(TgLinear *l) {
    tg_free(l->W);
    tg_free(l->B);
}

Tensor *tg_linear_forward(TgLinear *l, Tensor *x, Tensor **xw_out) {
    Tensor *xw = tg_matmul(x, l->W);
    if (xw_out) *xw_out = xw;
    return tg_add(xw, l->B);
}

Tensor **tg_linear_params(TgLinear *l, int *n_out) {
    Tensor **p = malloc(2 * sizeof(Tensor *));
    if (!p) { fprintf(stderr, "tg_linear_params: out of memory\n"); exit(1); }
    p[0] = l->W;
    p[1] = l->B;
    *n_out = 2;
    return p;
}
