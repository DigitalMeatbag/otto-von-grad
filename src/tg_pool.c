#include "tg_pool.h"
#include "tg_ops.h"
#include "ovg_error.h"

Tensor *tg_pool_cls(Tensor *enc) {
    if (!enc)
        ovg_fatal("tg_pool_cls: NULL argument");
    if (enc->ndim != 3)
        ovg_fatal("tg_pool_cls: expected [B, T, C], got ndim=%d", enc->ndim);
    int B = enc->shape[0], C = enc->shape[2];

    /* slice(enc, axis 1, 0, 1) [B, 1, C] → reshape [B, C] */
    Tensor *cls = tg_slice(enc, 1, 0, 1);
    int out_shape[2] = {B, C};
    return tg_reshape(cls, 2, out_shape);
}

Tensor *tg_pool_mean_tokens(Tensor *enc) {
    if (!enc)
        ovg_fatal("tg_pool_mean_tokens: NULL argument");
    if (enc->ndim != 3)
        ovg_fatal("tg_pool_mean_tokens: expected [B, T, C], got ndim=%d", enc->ndim);
    int B = enc->shape[0], T = enc->shape[1], C = enc->shape[2];

    /* transpose(enc, 0, 1) [T, B, C] → reshape [T, B*C] → mean_rows [1, B*C]
       (mean over T per (b, c)) → reshape [B, C]. No ones tensor, no new op;
       two extra full-size copies (transpose, reshape) accepted. */
    Tensor *tbc = tg_transpose(enc, 0, 1);
    int flat_shape[2] = {T, B * C};
    Tensor *rows = tg_reshape(tbc, 2, flat_shape);
    Tensor *mean = tg_mean_rows(rows);
    int out_shape[2] = {B, C};
    return tg_reshape(mean, 2, out_shape);
}
