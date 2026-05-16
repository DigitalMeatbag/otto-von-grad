#include "tg_transformer.h"

#include <stdio.h>
#include <stdlib.h>

TgTransformer tg_transformer_create(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads) {
    if (n_blocks <= 0) {
        fprintf(stderr, "tg_transformer_create: n_blocks must be positive, got %d\n", n_blocks);
        exit(1);
    }

    TgTransformer t;
    t.blocks = calloc((size_t)n_blocks, sizeof(TgBlock));
    if (!t.blocks) { fprintf(stderr, "tg_transformer_create: out of memory\n"); exit(1); }

    t.n_blocks = n_blocks;
    t.embed_dim = embed_dim;
    t.hidden_dim = hidden_dim;

    for (int i = 0; i < n_blocks; i++)
        t.blocks[i] = tg_block_create(embed_dim, hidden_dim, seq_len, n_heads);

    return t;
}

TgTransformer tg_transformer_create_encoder(int n_blocks, int embed_dim, int hidden_dim, int seq_len, int n_heads) {
    if (n_blocks <= 0) {
        fprintf(stderr, "tg_transformer_create_encoder: n_blocks must be positive, got %d\n", n_blocks);
        exit(1);
    }

    TgTransformer t;
    t.blocks = calloc((size_t)n_blocks, sizeof(TgBlock));
    if (!t.blocks) { fprintf(stderr, "tg_transformer_create_encoder: out of memory\n"); exit(1); }

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
    if (!t || !X) {
        fprintf(stderr, "tg_transformer_forward: NULL argument\n");
        exit(1);
    }

    if (X->cols != t->embed_dim) {
        fprintf(stderr,
                "tg_transformer_forward: X has shape [%dx%d], expected cols=%d\n",
                X->rows, X->cols, t->embed_dim);
        exit(1);
    }

    Tensor *Y = X;
    for (int i = 0; i < t->n_blocks; i++)
        Y = tg_block_forward(&t->blocks[i], Y);

    return Y;
}
