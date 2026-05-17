#ifndef TG_GPT_H
#define TG_GPT_H

#include "tg_transformer.h"

typedef struct {
    Tensor        *TokEmb;   // [vocab_size x embed_dim]
    Tensor        *PosEmb;   // [seq_len x embed_dim]
    Tensor        *Wout;     // [embed_dim x vocab_size]
    TgTransformer  transformer;
    int            vocab_size;
    int            embed_dim;
    int            seq_len;
} TgGPT;

TgGPT   tg_gpt_create(int vocab_size, int embed_dim, int hidden_dim, int seq_len, int n_blocks, int n_heads);
void    tg_gpt_free(TgGPT *g);

// token_one_hot: [seq_len x vocab_size] -> returns logits [seq_len x vocab_size]
Tensor *tg_gpt_forward(TgGPT *g, Tensor *token_one_hot);

// fills params[] with all trainable tensors, returns count; exits if max_params is too small
int     tg_gpt_collect_params(TgGPT *g, Tensor **params, int max_params);

#endif
