#include "tg_gpt.h"

#include <stdio.h>
#include <stdlib.h>

TgGPT tg_gpt_create(int vocab_size, int embed_dim, int hidden_dim, int seq_len, int n_blocks, int n_heads) {
    TgGPT g;
    g.vocab_size = vocab_size;
    g.embed_dim  = embed_dim;
    g.seq_len    = seq_len;

    g.TokEmb = tg_new(vocab_size, embed_dim);
    g.PosEmb = tg_new(seq_len,    embed_dim);
    g.Wout   = tg_new(embed_dim,  vocab_size);
    tg_fill_randn(g.TokEmb, 0.1f);
    tg_fill_randn(g.PosEmb, 0.1f);
    tg_fill_randn(g.Wout,   0.1f);

    g.TokEmb->persistent = 1;
    g.PosEmb->persistent = 1;
    g.Wout->persistent   = 1;

    g.transformer = tg_transformer_create(n_blocks, embed_dim, hidden_dim, seq_len, n_heads);
    return g;
}

void tg_gpt_free(TgGPT *g) {
    if (!g) return;
    tg_free(g->TokEmb);
    tg_free(g->PosEmb);
    tg_free(g->Wout);
    tg_transformer_free(&g->transformer);
    g->TokEmb = NULL;
    g->PosEmb = NULL;
    g->Wout   = NULL;
}

Tensor *tg_gpt_forward(TgGPT *g, Tensor *token_one_hot) {
    if (!g || !token_one_hot) {
        fprintf(stderr, "tg_gpt_forward: NULL argument\n");
        exit(1);
    }
    if (token_one_hot->rows != g->seq_len || token_one_hot->cols != g->vocab_size) {
        fprintf(stderr, "tg_gpt_forward: expected one_hot [%dx%d], got [%dx%d]\n",
                g->seq_len, g->vocab_size, token_one_hot->rows, token_one_hot->cols);
        exit(1);
    }

    /*
        token_one_hot: [T x V]
        TokEmb:        [V x C]
        Xtok:          [T x C]

        X = Xtok + PosEmb  [T x C]
        Y = transformer(X) [T x C]
        logits = Y @ Wout  [T x V]
    */
    Tensor *Xtok   = tg_matmul(token_one_hot, g->TokEmb);
    Tensor *X      = tg_add(Xtok, g->PosEmb);
    Tensor *Y      = tg_transformer_forward(&g->transformer, X);
    return tg_matmul(Y, g->Wout);
}

int tg_gpt_collect_params(TgGPT *g, Tensor **params) {
    int n = 0;
    params[n++] = g->TokEmb;
    params[n++] = g->PosEmb;
    params[n++] = g->Wout;

    for (int i = 0; i < g->transformer.n_blocks; i++) {
        TgBlock *b  = &g->transformer.blocks[i];
        params[n++] = b->attn.Wq;
        params[n++] = b->attn.Wk;
        params[n++] = b->attn.Wv;
        params[n++] = b->attn.Wo;
        params[n++] = b->W1;
        params[n++] = b->B1;
        params[n++] = b->W2;
        params[n++] = b->B2;
    }

    return n;
}
