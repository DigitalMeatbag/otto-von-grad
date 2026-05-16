#ifndef ATTENTION_H
#define ATTENTION_H

#include "tg_ops.h"

typedef struct {
    Tensor *Wq;    // [embed_dim x embed_dim]
    Tensor *Wk;
    Tensor *Wv;
    Tensor *Wo;    // [embed_dim x embed_dim]

    int embed_dim;
    int head_dim;  // = embed_dim / n_heads
    int n_heads;
    int causal;    // 1 = causal mask (GPT), 0 = no mask (encoder/ViT)
} TgSelfAttention;

// Debug context: full Q/K/V projections, concatenated head output, final projection
typedef struct {
    Tensor *Q;     // [T x C]
    Tensor *K;
    Tensor *V;
    Tensor *O;     // [T x C] concatenated head outputs (before Wo)
    Tensor *proj;  // [T x C] output projection (O @ Wo)
} TgAttentionForward;

TgSelfAttention    tg_attention_create(int embed_dim, int n_heads);         // causal = 1
TgSelfAttention    tg_attention_create_encoder(int embed_dim, int n_heads); // causal = 0
void               tg_attention_free(TgSelfAttention *a);

// tg_attention_forward_full: returns debug context with Q/K/V/O/proj kept alive.
// To free the entire graph (including internal head nodes), call tg_free_graph(f.proj).
// Do NOT call both tg_attention_forward_free and tg_free_graph — that will double-free.
// tg_attention_forward_free is only for discarding the struct fields after tg_free_graph.
TgAttentionForward tg_attention_forward_full(TgSelfAttention *a, Tensor *X);
void               tg_attention_forward_free(TgAttentionForward *f);

Tensor            *tg_attention_forward(TgSelfAttention *a, Tensor *X);

#endif
