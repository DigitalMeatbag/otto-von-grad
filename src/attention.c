#include "attention.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

TgSelfAttention tg_attention_create(int embed_dim, int n_heads) {
    if (n_heads <= 0 || embed_dim % n_heads != 0) {
        fprintf(stderr,
                "tg_attention_create: embed_dim %d not divisible by n_heads %d\n",
                embed_dim, n_heads);
        exit(1);
    }
    if (n_heads > TG_MAX_PARENTS) {
        fprintf(stderr,
                "tg_attention_create: n_heads %d exceeds TG_MAX_PARENTS=%d\n",
                n_heads, TG_MAX_PARENTS);
        exit(1);
    }

    TgSelfAttention a;
    a.embed_dim = embed_dim;
    a.n_heads   = n_heads;
    a.head_dim  = embed_dim / n_heads;
    a.causal    = 1;

    a.Wq = tg_new(embed_dim, embed_dim);
    a.Wk = tg_new(embed_dim, embed_dim);
    a.Wv = tg_new(embed_dim, embed_dim);
    a.Wo = tg_new(embed_dim, embed_dim);

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
    if (!a || !X) {
        fprintf(stderr, "tg_attention_forward_full: NULL argument\n");
        exit(1);
    }
    if (X->cols != a->embed_dim) {
        fprintf(stderr,
                "tg_attention_forward_full: X has shape [%dx%d], expected cols=%d\n",
                X->rows, X->cols, a->embed_dim);
        exit(1);
    }

    /*
        Full Q/K/V projections:

        X:         [T x C]
        Wq/Wk/Wv:  [C x C]
        Q/K/V:     [T x C]
    */
    TgAttentionForward f;
    f.Q = tg_matmul(X, a->Wq);
    f.K = tg_matmul(X, a->Wk);
    f.V = tg_matmul(X, a->Wv);

    /*
        Per-head attention:

        For each head h:
          Qh = Q[:, h*D : (h+1)*D]   [T x D]
          Kh = K[:, h*D : (h+1)*D]   [T x D]
          Vh = V[:, h*D : (h+1)*D]   [T x D]

          Kht    = Kh^T               [D x T]
          scores = Qh @ Kht           [T x T]
          scaled = scores * (1/sqrt D)[T x T]
          masked = causal_mask(scaled)[T x T]
          wts    = softmax_rows(masked)[T x T]
          Oh     = wts @ Vh           [T x D]

        O = concat_cols(Oh...)        [T x C]
    */
    int D = a->head_dim;
    int n = a->n_heads;
    Tensor *head_out[TG_MAX_PARENTS];
    float scale = 1.0f / sqrtf((float)D);

    for (int h = 0; h < n; h++) {
        Tensor *Qh  = tg_slice_cols(f.Q, h * D, (h + 1) * D);
        Tensor *Kh  = tg_slice_cols(f.K, h * D, (h + 1) * D);
        Tensor *Vh  = tg_slice_cols(f.V, h * D, (h + 1) * D);

        Tensor *Kht     = tg_transpose(Kh);
        Tensor *scores  = tg_matmul(Qh, Kht);
        Tensor *scaled  = tg_scale(scores, scale);
        Tensor *weights = a->causal ? tg_softmax_rows(tg_causal_mask(scaled))
                                    : tg_softmax_rows(scaled);
        head_out[h]     = tg_matmul(weights, Vh);
    }

    f.O    = tg_concat_cols(head_out, n);
    f.proj = tg_matmul(f.O, a->Wo);

    return f;
}

void tg_attention_forward_free(TgAttentionForward *f) {
    if (!f) return;
    tg_free(f->proj);
    tg_free(f->O);
    tg_free(f->V);
    tg_free(f->K);
    tg_free(f->Q);
    f->proj = f->O = f->V = f->K = f->Q = NULL;
}

Tensor *tg_attention_forward(TgSelfAttention *a, Tensor *X) {
    TgAttentionForward f = tg_attention_forward_full(a, X);
    return f.proj;
}
