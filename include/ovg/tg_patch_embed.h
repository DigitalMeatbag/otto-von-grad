#ifndef TG_PATCH_EMBED_H
#define TG_PATCH_EMBED_H

#include "tg_ops.h"

/* Patch embedding for a vision transformer: a linear projection of flattened
   patches, a learned positional embedding, and an optional CLS token.
   Takes tokens (rows of patch_size floats), never pixels: patch extraction
   from an image stays with the application. */
typedef struct {
    Tensor *Cls;      /* [1, C]; NULL when use_cls == 0 */
    Tensor *Proj;     /* [patch_size, C] */
    Tensor *PosEmb;   /* [n_patches, C]; applied to patch tokens only */
    int n_patches, patch_size, embed_dim, use_cls;
} TgPatchEmbed;

/* Cls and Proj: tg_fill_xavier_uniform; PosEmb: zeros. All persistent.
   Fatal if n_patches, patch_size, or embed_dim <= 0, or if use_cls is not 0 or 1. */
TgPatchEmbed tg_patch_embed_create(int n_patches, int patch_size, int embed_dim, int use_cls);
void         tg_patch_embed_free(TgPatchEmbed *p);

/* patches: [n_patches, patch_size] (treated as batch 1) or [B, n_patches, patch_size].
   Returns [B, n_patches + use_cls, C] — always 3D; the CLS token, if any, is token 0.
   Fatal on any other rank, or if the trailing two dims are not exactly
   [n_patches, patch_size]. F32 only. */
Tensor      *tg_patch_embed_forward(TgPatchEmbed *p, Tensor *patches);

/* Fills params in the order Cls (if use_cls), Proj, PosEmb; returns 3 or 2.
   Fatal if max_params is too small. This order is the checkpoint contract. */
int          tg_patch_embed_collect_params(TgPatchEmbed *p, Tensor **params, int max_params);

static inline int tg_patch_embed_n_tokens(const TgPatchEmbed *p) { return p->n_patches + p->use_cls; }

#endif
