#include "tg_patch_embed.h"
#include "tg_ops.h"
#include "tg_tensor.h"
#include "ovg_error.h"

#include <stddef.h>

TgPatchEmbed tg_patch_embed_create(int n_patches, int patch_size, int embed_dim, int use_cls) {
    if (n_patches <= 0 || patch_size <= 0 || embed_dim <= 0)
        ovg_fatal("tg_patch_embed_create: n_patches (%d), patch_size (%d), embed_dim (%d) must be positive",
                  n_patches, patch_size, embed_dim);
    if (use_cls != 0 && use_cls != 1)
        ovg_fatal("tg_patch_embed_create: use_cls must be 0 or 1, got %d", use_cls);

    TgPatchEmbed p;
    p.n_patches  = n_patches;
    p.patch_size = patch_size;
    p.embed_dim  = embed_dim;
    p.use_cls    = use_cls;

    int cshape[2] = {1,          embed_dim};
    int wshape[2] = {patch_size, embed_dim};
    int pshape[2] = {n_patches,  embed_dim};

    p.Cls = NULL;
    if (use_cls) {
        p.Cls = tg_new(2, cshape);
        tg_fill_xavier_uniform(p.Cls);
        p.Cls->persistent = 1;
    }
    p.Proj = tg_new(2, wshape);
    tg_fill_xavier_uniform(p.Proj);
    p.Proj->persistent = 1;

    p.PosEmb = tg_new(2, pshape);
    tg_fill(p.PosEmb, 0.0f);
    p.PosEmb->persistent = 1;

    return p;
}

void tg_patch_embed_free(TgPatchEmbed *p) {
    if (!p) return;
    if (p->Cls) tg_free(p->Cls);
    tg_free(p->Proj);
    tg_free(p->PosEmb);
    p->Cls    = NULL;
    p->Proj   = NULL;
    p->PosEmb = NULL;
}

Tensor *tg_patch_embed_forward(TgPatchEmbed *p, Tensor *patches) {
    if (!p || !patches)
        ovg_fatal("tg_patch_embed_forward: NULL argument");
    if (patches->ndim != 2 && patches->ndim != 3)
        ovg_fatal("tg_patch_embed_forward: expected 2D or 3D input, got ndim=%d", patches->ndim);
    if (patches->dtype != TG_DTYPE_F32)
        ovg_fatal("tg_patch_embed_forward: F32 only; cast BF16 inputs first");
    {
        int np = patches->shape[patches->ndim - 2];
        int ps = patches->shape[patches->ndim - 1];
        if (np != p->n_patches || ps != p->patch_size)
            ovg_fatal("tg_patch_embed_forward: expected patches [..., %d, %d], got [..., %d, %d]",
                      p->n_patches, p->patch_size, np, ps);
    }

    /* Promote 2D [n_patches, patch_size] to batch 1 so everything below is 3D. */
    if (patches->ndim == 2) {
        int s3[3] = {1, p->n_patches, p->patch_size};
        patches = tg_reshape(patches, 3, s3);
    }
    int B = patches->shape[0];
    int C = p->embed_dim;

    /*
        X    = patches @ Proj                                 [B, n_patches, C]
        Pos  = expand_dim(reshape(PosEmb, [1, n_patches, C]), 0, B)
        X    = X + Pos
        if use_cls:
          Cls3 = expand_dim(reshape(Cls, [1, 1, C]), 0, B)    [B, 1, C]
          X    = concat(Cls3, X, axis 1)                      [B, n_patches + 1, C]
    */
    Tensor *X = tg_matmul(patches, p->Proj);

    int pos_shape[3] = {1, p->n_patches, C};
    Tensor *pos1 = tg_reshape(p->PosEmb, 3, pos_shape);
    Tensor *pos  = tg_expand_dim(pos1, 0, B);
    X = tg_add(X, pos);

    if (p->use_cls) {
        int cls_shape[3] = {1, 1, C};
        Tensor *cls1 = tg_reshape(p->Cls, 3, cls_shape);
        Tensor *cls3 = tg_expand_dim(cls1, 0, B);
        X = tg_concat(cls3, X, 1);
    }
    return X;
}

int tg_patch_embed_collect_params(TgPatchEmbed *p, Tensor **params, int max_params) {
    if (!p || !params)
        ovg_fatal("tg_patch_embed_collect_params: NULL argument");
    int need = 2 + p->use_cls;
    if (max_params < need)
        ovg_fatal("tg_patch_embed_collect_params: max_params=%d < %d", max_params, need);
    int n = 0;
    if (p->use_cls) params[n++] = p->Cls;
    params[n++] = p->Proj;
    params[n++] = p->PosEmb;
    return n;
}
