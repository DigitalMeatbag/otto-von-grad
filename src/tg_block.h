#ifndef TG_BLOCK_H
#define TG_BLOCK_H

#include "attention.h"

typedef struct {
    TgSelfAttention attn;

    Tensor *gamma1;  /* LN affine scale before attention  [1 x embed_dim] */
    Tensor *beta1;   /* LN affine shift before attention  [1 x embed_dim] */
    Tensor *gamma2;  /* LN affine scale before FFN        [1 x embed_dim] */
    Tensor *beta2;   /* LN affine shift before FFN        [1 x embed_dim] */

    Tensor *W1;
    Tensor *B1;
    Tensor *W2;
    Tensor *B2;

    int   embed_dim;
    int   hidden_dim;
    float dropout;
} TgBlock;

TgBlock tg_block_create(int embed_dim, int hidden_dim, int seq_len, int n_heads);
TgBlock tg_block_create_encoder(int embed_dim, int hidden_dim, int seq_len, int n_heads);
void tg_block_free(TgBlock *b);

Tensor *tg_block_forward(TgBlock *b, Tensor *X);

#endif
