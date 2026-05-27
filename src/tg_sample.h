#ifndef TG_SAMPLE_H
#define TG_SAMPLE_H

#include "tg_tensor.h"
#include "tg_gpt.h"
#include "tg_tokenizer.h"

/* Greedy argmax over logits row `row`. */
int  tg_sample_argmax(const Tensor *logits, int row);

/* Temperature-scaled top-k sampling over logits row `row`.
   temperature: 1.0 = unchanged; < 1.0 = sharper; > 1.0 = flatter.
   top_k: 0 = use all tokens; 1 = deterministic argmax; > 1 = restrict to top k.
   Uses tg_rng_uniform() internally. */
int  tg_sample_topk(const Tensor *logits, int row, float temperature, int top_k);

/* Autoregressive text generation.
   context[0..ctx_len) seeds the window; ctx_len must equal g->seq_len.
   on_token is called for each produced character (may be NULL).
   tg_training is set to 0 internally for the duration of generation. */
void tg_generate(TgGPT *g, const TgVocab *v,
                 const int *context, int ctx_len,
                 int steps, float temperature, int top_k,
                 void (*on_token)(char c, void *ud), void *userdata);

#endif /* TG_SAMPLE_H */
