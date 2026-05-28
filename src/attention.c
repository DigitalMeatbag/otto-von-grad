#include "attention.h"
#include "ovg_error.h"

#include <math.h>
#include <stdlib.h>

TgSelfAttention tg_attention_create(int embed_dim, int n_heads) {
    if (n_heads <= 0 || embed_dim % n_heads != 0)
        ovg_fatal("tg_attention_create: embed_dim %d not divisible by n_heads %d",
                  embed_dim, n_heads);

    TgSelfAttention a;
    a.embed_dim = embed_dim;
    a.n_heads   = n_heads;
    a.head_dim  = embed_dim / n_heads;
    a.causal    = 1;

    int wshape[2] = {embed_dim, embed_dim};
    a.Wq = tg_new(2, wshape);
    a.Wk = tg_new(2, wshape);
    a.Wv = tg_new(2, wshape);
    a.Wo = tg_new(2, wshape);

    float scale = 0.1f;
    tg_fill_randn(a.Wq, scale);
    tg_fill_randn(a.Wk, scale);
    tg_fill_randn(a.Wv, scale);
    tg_fill_randn(a.Wo, scale);

    a.Wq->persistent = 1;
    a.Wk->persistent = 1;
    a.Wv->persistent = 1;
    a.Wo->persistent = 1;

    return a;
}

void tg_attention_free(TgSelfAttention *a) {
    if (!a) return;
    tg_free(a->Wq);
    tg_free(a->Wk);
    tg_free(a->Wv);
    tg_free(a->Wo);
    a->Wq = NULL;
    a->Wk = NULL;
    a->Wv = NULL;
    a->Wo = NULL;
    a->embed_dim = 0;
    a->head_dim  = 0;
    a->n_heads   = 0;
    a->causal    = 0;
}

TgSelfAttention tg_attention_create_encoder(int embed_dim, int n_heads) {
    TgSelfAttention a = tg_attention_create(embed_dim, n_heads);
    a.causal = 0;
    return a;
}

TgAttentionForward tg_attention_forward_full(TgSelfAttention *a, Tensor *X) {
    if (!a || !X)
        ovg_fatal("tg_attention_forward_full: NULL argument");
    if (X->ndim < 2 || X->shape[X->ndim - 1] != a->embed_dim)
        ovg_fatal("tg_attention_forward_full: last dim of X (%d) != embed_dim (%d)",
                  X->shape[X->ndim - 1], a->embed_dim);

    int B = (X->ndim == 3) ? X->shape[0] : 1;
    int T = X->shape[X->ndim - 2];
    int C = a->embed_dim;
    int H = a->n_heads;
    int D = a->head_dim;  /* C = H * D */

    TgAttentionForward f;

    /*  Q/K/V = X @ Wq/Wk/Wv: [B,T,C] @ [C,C] → [B,T,C] */
    f.Q = tg_matmul(X, a->Wq);
    f.K = tg_matmul(X, a->Wk);
    f.V = tg_matmul(X, a->Wv);

    /* Reshape [B,T,C] → [B,T,H,D] then transpose(1,2) → [B,H,T,D]
       Direct reshape to [B,H,T,D] would scramble head/token ordering. */
    int q_4d_shape[4] = {B, T, H, D};
    Tensor *Q_bthd = tg_reshape(f.Q, 4, q_4d_shape);
    Tensor *K_bthd = tg_reshape(f.K, 4, q_4d_shape);
    Tensor *V_bthd = tg_reshape(f.V, 4, q_4d_shape);

    Tensor *Q_4d = tg_transpose(Q_bthd, 1, 2);  /* [B,H,T,D] */
    Tensor *K_4d = tg_transpose(K_bthd, 1, 2);  /* [B,H,T,D] */
    Tensor *V_4d = tg_transpose(V_bthd, 1, 2);  /* [B,H,T,D] */

    /* K^T: [B,H,T,D] → [B,H,D,T] */
    Tensor *K_t = tg_transpose(K_4d, 2, 3);

    /* Attention scores: [B,H,T,D] @ [B,H,D,T] → [B,H,T,T] */
    Tensor *scores = tg_matmul(Q_4d, K_t);
    Tensor *scaled = tg_scale(scores, 1.0f / sqrtf((float)D));

    /* Optional causal mask, then softmax over last axis */
    Tensor *weights = a->causal
        ? tg_softmax(tg_causal_mask(scaled), 3)
        : tg_softmax(scaled, 3);

    /* Context: [B,H,T,T] @ [B,H,T,D] → [B,H,T,D] */
    Tensor *context_4d = tg_matmul(weights, V_4d);

    /* Transpose back [B,H,T,D] → [B,T,H,D] then reshape → [B,T,C] */
    Tensor *context_bthd = tg_transpose(context_4d, 1, 2);
    int ctx_shape[3] = {B, T, C};
    f.O = tg_reshape(context_bthd, 3, ctx_shape);

    /* Output projection: [B,T,C] @ [C,C] → [B,T,C] */
    f.proj = tg_matmul(f.O, a->Wo);

    return f;
}

void tg_attention_forward_free(TgAttentionForward *f) {
    if (!f) return;
    /* Graph nodes are freed by tg_free_graph(f->proj).
       Just null the pointers so callers don't double-free. */
    f->proj = f->O = f->V = f->K = f->Q = NULL;
}

Tensor *tg_attention_forward(TgSelfAttention *a, Tensor *X) {
    TgAttentionForward f = tg_attention_forward_full(a, X);
    return f.proj;
}
