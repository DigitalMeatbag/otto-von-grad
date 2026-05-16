#ifndef TG_BLOCK_H
#define TG_BLOCK_H

#include "attention.h"

typedef struct {
    TgSelfAttention attn;

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
