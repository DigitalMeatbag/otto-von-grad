#include "tg_block.h"
#include "tg_ops.h"
#include "tg_rng.h"
#include "tg_train.h"
#include "ovg_error.h"

#include <math.h>
#include <stdlib.h>

TgBlock tg_block_create(int embed_dim, int hidden_dim, int seq_len, int n_heads) {
    TgBlock b;

    b.attn = tg_attention_create(embed_dim, n_heads);

    int gshape[2] = {1, embed_dim};
    b.gamma1 = tg_new(2, gshape);  tg_fill(b.gamma1, 1.0f);  b.gamma1->persistent = 1;
    b.beta1  = tg_new(2, gshape);  tg_fill(b.beta1,  0.0f);  b.beta1->persistent  = 1;
    b.gamma2 = tg_new(2, gshape);  tg_fill(b.gamma2, 1.0f);  b.gamma2->persistent = 1;
    b.beta2  = tg_new(2, gshape);  tg_fill(b.beta2,  0.0f);  b.beta2->persistent  = 1;

    int w1shape[2] = {embed_dim, hidden_dim};
    int b1shape[2] = {1, hidden_dim};
    int w2shape[2] = {hidden_dim, embed_dim};
    int b2shape[2] = {1, embed_dim};

    b.W1 = tg_new(2, w1shape);
    b.B1 = tg_new(2, b1shape);
    b.W2 = tg_new(2, w2shape);
    b.B2 = tg_new(2, b2shape);

    float scale = 0.1f;
    tg_fill_randn(b.W1, scale);
    tg_fill(b.B1, 0.0f);
    tg_fill_randn(b.W2, scale);
    tg_fill(b.B2, 0.0f);

    b.W1->persistent = 1;
    b.B1->persistent = 1;
    b.W2->persistent = 1;
    b.B2->persistent = 1;

    b.embed_dim  = embed_dim;
    b.hidden_dim = hidden_dim;
    b.dropout    = 0.0f;
    b.drop_path_rate = 0.0f;

    (void)seq_len;
    return b;
}

TgBlock tg_block_create_encoder(int embed_dim, int hidden_dim, int seq_len, int n_heads) {
    TgBlock b = tg_block_create(embed_dim, hidden_dim, seq_len, n_heads);
    b.attn.causal = 0;
    return b;
}

void tg_block_free(TgBlock *b) {
    if (!b) return;

    tg_attention_free(&b->attn);
    tg_free(b->gamma1);
    tg_free(b->beta1);
    tg_free(b->gamma2);
    tg_free(b->beta2);
    tg_free(b->W1);
    tg_free(b->B1);
    tg_free(b->W2);
    tg_free(b->B2);

    b->gamma1 = NULL;
    b->beta1  = NULL;
    b->gamma2 = NULL;
    b->beta2  = NULL;
    b->W1 = NULL;
    b->B1 = NULL;
    b->W2 = NULL;
    b->B2 = NULL;
    b->embed_dim  = 0;
    b->hidden_dim = 0;
}

/* Expand a [1, C] parameter to [B, T, C] for use with tg_layer_norm or tg_add.
   [1, C] → reshape [1, 1, C] → expand_dim(0, B) → [B, 1, C] → expand_dim(1, T) → [B, T, C] */
static Tensor *expand_param_btc(Tensor *param, int B, int T) {
    int C = param->shape[1];
    int s3[3] = {1, 1, C};
    Tensor *p3 = tg_reshape(param, 3, s3);
    Tensor *pB = tg_expand_dim(p3, 0, B);   /* [B, 1, C] */
    return   tg_expand_dim(pB, 1, T);        /* [B, T, C] */
}

Tensor *tg_block_forward(TgBlock *b, Tensor *X) {
    if (!b || !X)
        ovg_fatal("tg_block_forward: NULL argument");
    if (X->ndim < 2 || X->shape[X->ndim - 1] != b->embed_dim)
        ovg_fatal("tg_block_forward: last dim (%d) != embed_dim (%d)",
                  X->shape[X->ndim - 1], b->embed_dim);

    /* Normalize to 3D [B, T, C] so all internal ops are uniform */
    if (X->ndim == 2) {
        int s3[3] = {1, X->shape[0], X->shape[1]};
        X = tg_reshape(X, 3, s3);
    }
    int B = X->shape[0];
    int T = X->shape[1];
    int C = b->embed_dim;
    int H = b->hidden_dim;

    /* Expand [1, C] affine params to [B, T, C] for layer norm */
    Tensor *g1 = expand_param_btc(b->gamma1, B, T);
    Tensor *b1_ln = expand_param_btc(b->beta1,  B, T);
    Tensor *g2 = expand_param_btc(b->gamma2, B, T);
    Tensor *b2_ln = expand_param_btc(b->beta2,  B, T);

    /* Expand [1, H] and [1, C] FFN biases to [B, T, H/C] */
    int s3h[3] = {1, 1, H};
    int s3c[3] = {1, 1, C};
    Tensor *B1_3 = tg_reshape(b->B1, 3, s3h);
    Tensor *B1_B = tg_expand_dim(B1_3, 0, B);
    Tensor *B1   = tg_expand_dim(B1_B, 1, T);  /* [B, T, H] */
    Tensor *B2_3 = tg_reshape(b->B2, 3, s3c);
    Tensor *B2_B = tg_expand_dim(B2_3, 0, B);
    Tensor *B2   = tg_expand_dim(B2_B, 1, T);  /* [B, T, C] */

    /*
        Pre-norm transformer block:

        LN_X  = layer_norm(X,  g1, b1_ln)     [B, T, C]
        A     = attention(LN_X)                [B, T, C]
        Y1    = X + A

        LN_Y1 = layer_norm(Y1, g2, b2_ln)
        F1    = tanh(LN_Y1 @ W1 + B1)         [B, T, H]
        F2    = F1 @ W2 + B2                   [B, T, C]
        Y2    = Y1 + F2
    */
    Tensor *LN_X  = tg_layer_norm(X, g1, b1_ln, 1.0e-5f);
    Tensor *A_raw = tg_attention_forward(&b->attn, LN_X);
    Tensor *A     = b->dropout > 0.0f ? tg_dropout(A_raw, b->dropout) : A_raw;
    if (tg_training && b->drop_path_rate > 0.0f) {
        float u = tg_rng_uniform();
        A = tg_scale(A, u < b->drop_path_rate ? 0.0f : 1.0f / (1.0f - b->drop_path_rate));
    }
    Tensor *Y1 = tg_add(X, A);

    Tensor *LN_Y1 = tg_layer_norm(Y1, g2, b2_ln, 1.0e-5f);
    Tensor *Y1W1  = tg_matmul(LN_Y1, b->W1);
    Tensor *F1pre = tg_add(Y1W1, B1);
    Tensor *F1act = tg_tanh(F1pre);
    Tensor *F1    = b->dropout > 0.0f ? tg_dropout(F1act, b->dropout) : F1act;
    Tensor *F1W2  = tg_matmul(F1, b->W2);
    Tensor *F2    = tg_add(F1W2, B2);
    if (tg_training && b->drop_path_rate > 0.0f) {
        float u = tg_rng_uniform();
        F2 = tg_scale(F2, u < b->drop_path_rate ? 0.0f : 1.0f / (1.0f - b->drop_path_rate));
    }
    return tg_add(Y1, F2);
}
