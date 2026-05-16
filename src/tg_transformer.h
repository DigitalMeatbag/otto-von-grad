#ifndef TG_TRANSFORMER_H
#define TG_TRANSFORMER_H

#include "tg_block.h"

typedef struct {
    TgBlock *blocks;
    int n_blocks;
    int embed_dim;
    int hidden_dim;
} TgTransformer;

TgTransformer tg_transformer_create(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads);
TgTransformer tg_transformer_create_encoder(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads);
void tg_transformer_free(TgTransformer *t);

Tensor *tg_transformer_forward(TgTransformer *t, Tensor *X);

#endif
