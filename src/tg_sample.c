#include "tg_sample.h"
#include "tg_train.h"
#include "tg_rng.h"
#include "ovg_error.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef OVG_CUDA_ENABLED
#  include "tg_cuda.h"
#endif

int tg_sample_argmax(const Tensor *logits, int row) {
    if (!logits || logits->ndim != 2 || row < 0 || row >= logits->shape[0])
        ovg_fatal("tg_sample_argmax: invalid arguments");
#ifdef OVG_CUDA_ENABLED
    if (logits->on_cuda) tg_from_cuda((Tensor *)logits);
#endif
    int V = logits->shape[1];
    const float *d = TG_DATAF(logits);
    int   best     = 0;
    float best_val = d[row * V];
    for (int j = 1; j < V; j++) {
        float val = d[row * V + j];
        if (val > best_val) { best = j; best_val = val; }
    }
    return best;
}

int tg_sample_topk(const Tensor *logits, int row, float temperature, int top_k) {
    if (!logits || logits->ndim != 2 || row < 0 || row >= logits->shape[0])
        ovg_fatal("tg_sample_topk: invalid arguments");
    if (temperature <= 0.0f)
        ovg_fatal("tg_sample_topk: temperature must be > 0");
#ifdef OVG_CUDA_ENABLED
    if (logits->on_cuda) tg_from_cuda((Tensor *)logits);
#endif

    int V = logits->shape[1];
    const float *d = TG_DATAF(logits);
    if (top_k <= 0 || top_k > V) top_k = V;
    if (top_k == 1) return tg_sample_argmax(logits, row);

    const float *src = d + row * V;

    float *scores = malloc((size_t)V * sizeof(float));
    int   *idx    = malloc((size_t)V * sizeof(int));
    if (!scores || !idx) { free(scores); free(idx); ovg_fatal("tg_sample_topk: out of memory"); }

    for (int j = 0; j < V; j++) { scores[j] = src[j] / temperature; idx[j] = j; }

    for (int k = 0; k < top_k; k++) {
        int best_j = k;
        for (int j = k + 1; j < V; j++)
            if (scores[idx[j]] > scores[idx[best_j]]) best_j = j;
        int tmp = idx[k]; idx[k] = idx[best_j]; idx[best_j] = tmp;
    }

    float max_s = scores[idx[0]];
    for (int k = 1; k < top_k; k++)
        if (scores[idx[k]] > max_s) max_s = scores[idx[k]];

    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        scores[k] = expf(scores[idx[k]] - max_s);
        sum += scores[k];
    }

    float r   = tg_rng_uniform() * sum;
    float cum = 0.0f;
    int result = idx[top_k - 1];
    for (int k = 0; k < top_k; k++) {
        cum += scores[k];
        if (r < cum) { result = idx[k]; break; }
    }

    free(scores);
    free(idx);
    return result;
}

void tg_generate(TgGPT *g, const TgVocab *v,
                 const int *context, int ctx_len,
                 int steps, float temperature, int top_k,
                 void (*on_token)(char c, void *ud), void *userdata) {
    if (!g || !v || !context)
        ovg_fatal("tg_generate: NULL argument");
    if (ctx_len != g->seq_len)
        ovg_fatal("tg_generate: ctx_len (%d) must equal g->seq_len (%d)",
                  ctx_len, g->seq_len);

    int *ctx = malloc((size_t)g->seq_len * sizeof(int));
    if (!ctx) ovg_fatal("tg_generate: out of memory");
    memcpy(ctx, context, (size_t)g->seq_len * sizeof(int));

    int saved_training = tg_training;
    tg_training = 0;

    for (int s = 0; s < steps; s++) {
        Tensor *logits = tg_gpt_forward(g, ctx, 1);
        int next = (top_k == 1 || temperature <= 0.0f)
                 ? tg_sample_argmax(logits, g->seq_len - 1)
                 : tg_sample_topk(logits, g->seq_len - 1, temperature, top_k);

        if (on_token) on_token(tg_vocab_decode(v, next), userdata);

        for (int i = 0; i < g->seq_len - 1; i++) ctx[i] = ctx[i + 1];
        ctx[g->seq_len - 1] = next;
        tg_free_graph(logits);
    }

    tg_training = saved_training;
    free(ctx);
}
