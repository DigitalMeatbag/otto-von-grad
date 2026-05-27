#include "tg_gpt.h"
#include "tg_ops.h"
#include "ovg_error.h"

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

Tensor *tg_gpt_forward(TgGPT *g, const int *token_ids) {
    if (!g || !token_ids)
        ovg_fatal("tg_gpt_forward: NULL argument");

    /*
        token_ids:  [seq_len]  (integer token indices)
        TokEmb:     [V x C]
        Xtok:       [T x C]   (gathered rows of TokEmb)

        X = Xtok + PosEmb  [T x C]
        Y = transformer(X) [T x C]
        logits = Y @ Wout  [T x V]
    */
    Tensor *Xtok   = tg_embed(g->TokEmb, token_ids, g->seq_len);
    Tensor *X      = tg_add(Xtok, g->PosEmb);
    Tensor *Y      = tg_transformer_forward(&g->transformer, X);
    return tg_matmul(Y, g->Wout);
}

TgGPT tg_gpt_create_from_config(const TgGPTConfig *cfg) {
    return tg_gpt_create(cfg->vocab_size, cfg->embed_dim, cfg->hidden_dim,
                         cfg->seq_len, cfg->n_blocks, cfg->n_heads);
}

int tg_gpt_collect_params(TgGPT *g, Tensor **params, int max_params) {
    int needed = 3 + g->transformer.n_blocks * 12;  /* 12 = gamma1+beta1+Wq+Wk+Wv+Wo + gamma2+beta2+W1+B1+W2+B2 */
    if (needed > max_params)
        ovg_fatal("tg_gpt_collect_params: need %d slots, capacity %d", needed, max_params);
    int n = 0;
    params[n++] = g->TokEmb;
    params[n++] = g->PosEmb;
    params[n++] = g->Wout;

    for (int i = 0; i < g->transformer.n_blocks; i++) {
        TgBlock *b  = &g->transformer.blocks[i];
        params[n++] = b->gamma1;
        params[n++] = b->beta1;
        params[n++] = b->attn.Wq;
        params[n++] = b->attn.Wk;
        params[n++] = b->attn.Wv;
        params[n++] = b->attn.Wo;
        params[n++] = b->gamma2;
        params[n++] = b->beta2;
        params[n++] = b->W1;
        params[n++] = b->B1;
        params[n++] = b->W2;
        params[n++] = b->B2;
    }

    return n;
}
