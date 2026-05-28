#include "tg_gpt.h"
#include "tg_ops.h"
#include "ovg_error.h"

#include <stdlib.h>

TgGPT tg_gpt_create(int vocab_size, int embed_dim, int hidden_dim, int seq_len, int n_blocks, int n_heads) {
    TgGPT g;
    g.vocab_size = vocab_size;
    g.embed_dim  = embed_dim;
    g.seq_len    = seq_len;

    int te_shape[2] = {vocab_size, embed_dim};
    int pe_shape[2] = {seq_len,    embed_dim};
    int wo_shape[2] = {embed_dim,  vocab_size};

    g.TokEmb = tg_new(2, te_shape);
    g.PosEmb = tg_new(2, pe_shape);
    g.Wout   = tg_new(2, wo_shape);
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

Tensor *tg_gpt_forward(TgGPT *g, const int *token_ids, int batch_size) {
    if (!g || !token_ids)
        ovg_fatal("tg_gpt_forward: NULL argument");
    if (batch_size < 1)
        ovg_fatal("tg_gpt_forward: batch_size must be >= 1, got %d", batch_size);

    int B = batch_size, T = g->seq_len, C = g->embed_dim, V = g->vocab_size;

    /*  Embed: flat [B*T] token ids → [B*T, C] then reshape to [B, T, C] */
    Tensor *Xtok_flat = tg_embed(g->TokEmb, token_ids, B * T);
    int btc[3] = {B, T, C};
    Tensor *Xtok = tg_reshape(Xtok_flat, 3, btc);

    /* PosEmb is [T, C]; expand to [B, T, C] */
    int pos3[3] = {1, T, C};
    Tensor *PosEmb_3d = tg_reshape(g->PosEmb, 3, pos3);
    Tensor *PosEmb_B  = tg_expand_dim(PosEmb_3d, 0, B);  /* [B, T, C] */

    Tensor *X = tg_add(Xtok, PosEmb_B);          /* [B, T, C] */
    Tensor *Y = tg_transformer_forward(&g->transformer, X);  /* [B, T, C] */

    /* Logits: [B, T, C] @ [C, V] → [B, T, V]; reshape to [B*T, V] for CE */
    Tensor *logits_3d = tg_matmul(Y, g->Wout);   /* [B, T, V] */
    int logits2d[2] = {B * T, V};
    return tg_reshape(logits_3d, 2, logits2d);    /* [B*T, V] */
}

TgGPT tg_gpt_create_from_config(const TgGPTConfig *cfg) {
    return tg_gpt_create(cfg->vocab_size, cfg->embed_dim, cfg->hidden_dim,
                         cfg->seq_len, cfg->n_blocks, cfg->n_heads);
}

int tg_gpt_collect_params(TgGPT *g, Tensor **params, int max_params) {
    int needed = 3 + g->transformer.n_blocks * 12;
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
