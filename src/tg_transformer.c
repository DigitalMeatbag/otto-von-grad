#include "tg_transformer.h"
#include "ovg_error.h"

#include <stdlib.h>

TgTransformer tg_transformer_create(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads) {
    if (n_blocks <= 0)
        ovg_fatal("tg_transformer_create: n_blocks must be positive, got %d", n_blocks);

    TgTransformer t;
    t.blocks = calloc((size_t)n_blocks, sizeof(TgBlock));
    if (!t.blocks) ovg_fatal("tg_transformer_create: out of memory");

    t.n_blocks = n_blocks;
    t.embed_dim = embed_dim;
    t.hidden_dim = hidden_dim;

    for (int i = 0; i < n_blocks; i++)
        t.blocks[i] = tg_block_create(embed_dim, hidden_dim, seq_len, n_heads);

    return t;
}

TgTransformer tg_transformer_create_encoder(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads) {
    if (n_blocks <= 0)
        ovg_fatal("tg_transformer_create_encoder: n_blocks must be positive, got %d", n_blocks);

    TgTransformer t;
    t.blocks = calloc((size_t)n_blocks, sizeof(TgBlock));
    if (!t.blocks) ovg_fatal("tg_transformer_create_encoder: out of memory");

    t.n_blocks = n_blocks;
    t.embed_dim = embed_dim;
    t.hidden_dim = hidden_dim;

    for (int i = 0; i < n_blocks; i++)
        t.blocks[i] = tg_block_create_encoder(embed_dim, hidden_dim, seq_len, n_heads);

    return t;
}

void tg_transformer_free(TgTransformer *t) {
    if (!t) return;

    for (int i = 0; i < t->n_blocks; i++)
        tg_block_free(&t->blocks[i]);
    free(t->blocks);

    t->blocks = NULL;
    t->n_blocks = 0;
    t->embed_dim = 0;
    t->hidden_dim = 0;
}

Tensor *tg_transformer_forward(TgTransformer *t, Tensor *X) {
    if (!t || !X)
        ovg_fatal("tg_transformer_forward: NULL argument");

    if (X->cols != t->embed_dim)
        ovg_fatal("tg_transformer_forward: X has shape [%dx%d], expected cols=%d",
                  X->rows, X->cols, t->embed_dim);

    Tensor *Y = X;
    for (int i = 0; i < t->n_blocks; i++)
        Y = tg_block_forward(&t->blocks[i], Y);

    return Y;
}
