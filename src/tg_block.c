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

    b.gamma1 = tg_new(1, embed_dim);  tg_fill(b.gamma1, 1.0f);  b.gamma1->persistent = 1;
    b.beta1  = tg_new(1, embed_dim);  tg_fill(b.beta1,  0.0f);  b.beta1->persistent  = 1;
    b.gamma2 = tg_new(1, embed_dim);  tg_fill(b.gamma2, 1.0f);  b.gamma2->persistent = 1;
    b.beta2  = tg_new(1, embed_dim);  tg_fill(b.beta2,  0.0f);  b.beta2->persistent  = 1;

    b.W1 = tg_new(embed_dim, hidden_dim);
    b.B1 = tg_new(seq_len, hidden_dim);
    b.W2 = tg_new(hidden_dim, embed_dim);
    b.B2 = tg_new(seq_len, embed_dim);

    float scale = 0.1f;
    tg_fill_randn(b.W1, scale);
    tg_fill(b.B1, 0.0f);
    tg_fill_randn(b.W2, scale);
    tg_fill(b.B2, 0.0f);

    b.W1->persistent = 1;
    b.B1->persistent = 1;
    b.W2->persistent = 1;
    b.B2->persistent = 1;

    b.embed_dim = embed_dim;
    b.hidden_dim = hidden_dim;
    b.dropout = 0.0f;
    b.drop_path_rate = 0.0f;

    return b;
}

TgBlock tg_block_create_encoder(int embed_dim, int hidden_dim, int seq_len, int n_heads) {
    TgBlock b;

    b.attn = tg_attention_create_encoder(embed_dim, n_heads);

    b.gamma1 = tg_new(1, embed_dim);  tg_fill(b.gamma1, 1.0f);  b.gamma1->persistent = 1;
    b.beta1  = tg_new(1, embed_dim);  tg_fill(b.beta1,  0.0f);  b.beta1->persistent  = 1;
    b.gamma2 = tg_new(1, embed_dim);  tg_fill(b.gamma2, 1.0f);  b.gamma2->persistent = 1;
    b.beta2  = tg_new(1, embed_dim);  tg_fill(b.beta2,  0.0f);  b.beta2->persistent  = 1;

    b.W1 = tg_new(embed_dim, hidden_dim);
    b.B1 = tg_new(seq_len, hidden_dim);
    b.W2 = tg_new(hidden_dim, embed_dim);
    b.B2 = tg_new(seq_len, embed_dim);

    float scale = 0.1f;
    tg_fill_randn(b.W1, scale);
    tg_fill(b.B1, 0.0f);
    tg_fill_randn(b.W2, scale);
    tg_fill(b.B2, 0.0f);

    b.W1->persistent = 1;
    b.B1->persistent = 1;
    b.W2->persistent = 1;
    b.B2->persistent = 1;

    b.embed_dim = embed_dim;
    b.hidden_dim = hidden_dim;
    b.dropout = 0.0f;
    b.drop_path_rate = 0.0f;

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
    b->embed_dim = 0;
    b->hidden_dim = 0;
}

Tensor *tg_block_forward(TgBlock *b, Tensor *X) {
    if (!b || !X)
        ovg_fatal("tg_block_forward: NULL argument");

    if (X->cols != b->embed_dim)
        ovg_fatal("tg_block_forward: X has shape [%dx%d], expected cols=%d",
                  X->rows, X->cols, b->embed_dim);

    if (X->rows != b->B1->rows || X->rows != b->B2->rows)
        ovg_fatal("tg_block_forward: X rows=%d, expected bias rows B1=%d B2=%d",
                  X->rows, b->B1->rows, b->B2->rows);

    /*
        Pre-norm transformer block:

        LN_X  = layer_norm_rows(X)
        A     = attention(LN_X)      [T x C]
        Y1    = X + A                [T x C]

        LN_Y1 = layer_norm_rows(Y1)
        F1    = tanh(LN_Y1 @ W1 + B1) [T x H]
        F2    = F1 @ W2 + B2          [T x C]
        Y2    = Y1 + F2               [T x C]

        Intermediate graph tensors are intentionally kept alive for now because
        Y2 depends on them for backward.
    */
    Tensor *LN_X = tg_layer_norm_rows_affine(X, b->gamma1, b->beta1, 1.0e-5f);
    Tensor *A_raw = tg_attention_forward(&b->attn, LN_X);
    Tensor *A  = b->dropout > 0.0f ? tg_dropout(A_raw, b->dropout) : A_raw;
    if (tg_training && b->drop_path_rate > 0.0f) {
        float u = tg_rng_uniform();
        if (u < b->drop_path_rate)
            A = tg_scale(A, 0.0f);
        else
            A = tg_scale(A, 1.0f / (1.0f - b->drop_path_rate));
    }
    Tensor *Y1 = tg_add(X, A);

    Tensor *LN_Y1 = tg_layer_norm_rows_affine(Y1, b->gamma2, b->beta2, 1.0e-5f);
    Tensor *Y1W1  = tg_matmul(LN_Y1, b->W1);
    Tensor *F1pre = tg_add(Y1W1, b->B1);
    Tensor *F1act = tg_tanh(F1pre);
    Tensor *F1    = b->dropout > 0.0f ? tg_dropout(F1act, b->dropout) : F1act;
    Tensor *F1W2  = tg_matmul(F1, b->W2);
    Tensor *F2    = tg_add(F1W2, b->B2);
    if (tg_training && b->drop_path_rate > 0.0f) {
        float u = tg_rng_uniform();
        if (u < b->drop_path_rate)
            F2 = tg_scale(F2, 0.0f);
        else
            F2 = tg_scale(F2, 1.0f / (1.0f - b->drop_path_rate));
    }
    Tensor *Y2    = tg_add(Y1, F2);

    return Y2;
}
