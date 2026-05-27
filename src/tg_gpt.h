#ifndef TG_GPT_H
#define TG_GPT_H

#include "tg_transformer.h"

typedef struct {
    int vocab_size;
    int embed_dim;
    int hidden_dim;
    int seq_len;
    int n_blocks;
    int n_heads;
} TgGPTConfig;

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
TgGPT   tg_gpt_create_from_config(const TgGPTConfig *cfg);
void    tg_gpt_free(TgGPT *g);

// token_ids: integer array of length seq_len -> returns logits [seq_len x vocab_size]
Tensor *tg_gpt_forward(TgGPT *g, const int *token_ids);

// fills params[] with all trainable tensors, returns count; exits if max_params is too small
int     tg_gpt_collect_params(TgGPT *g, Tensor **params, int max_params);

#endif
