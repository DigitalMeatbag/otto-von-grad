#include "ovg_test.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "attention.h"

#include <math.h>

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static int any_nonzero(float *arr, int n) {
    for (int i = 0; i < n; i++) if (arr[i] != 0.0f) return 1;
    return 0;
}

/* ── Causal attention: shape + grad coverage ─────────────────────────────── */

static void test_causal_attention_grads(void) {
    int T = 4, C = 4, n_heads = 1;
    Tensor *X = tg_new(T, C);
    tg_fill_randn(X, 0.5f);
    X->persistent = 1;

    TgSelfAttention attn = tg_attention_create(C, n_heads);
    TgAttentionForward f = tg_attention_forward_full(&attn, X);

    OVG_CHECK_SHAPE(f.proj, T, C);

    Tensor *Y    = tg_add(X, f.proj);
    Tensor *loss = tg_sum(Y);
    tg_backward(loss);

    OVG_CHECK(any_nonzero(attn.Wq->grad, C * C));
    OVG_CHECK(any_nonzero(attn.Wk->grad, C * C));
    OVG_CHECK(any_nonzero(attn.Wv->grad, C * C));
    OVG_CHECK(any_nonzero(attn.Wo->grad, C * C));
    OVG_CHECK(any_nonzero(X->grad, T * C));

    tg_free_graph(loss);
    tg_attention_free(&attn);
    X->persistent = 0;
    tg_free(X);
}

static void test_causal_attention_multihead(void) {
    int T = 4, C = 4, n_heads = 2;
    Tensor *X = tg_new(T, C);
    tg_fill_randn(X, 0.5f);
    X->persistent = 1;

    TgSelfAttention attn = tg_attention_create(C, n_heads);
    TgAttentionForward f = tg_attention_forward_full(&attn, X);

    OVG_CHECK_SHAPE(f.proj, T, C);

    Tensor *Y    = tg_add(X, f.proj);
    Tensor *loss = tg_sum(Y);
    tg_backward(loss);

    OVG_CHECK(any_nonzero(attn.Wq->grad, C * C));
    OVG_CHECK(any_nonzero(X->grad, T * C));

    tg_free_graph(loss);
    tg_attention_free(&attn);
    X->persistent = 0;
    tg_free(X);
}

/* ── Encoder attention: no causal mask ───────────────────────────────────── */

static void test_encoder_weights(void) {
    /* Build the encoder graph manually to inspect softmax weights.
       All positions should attend to all others (no upper-triangle masking). */
    int T = 4, C = 4;
    Tensor *X = tg_new(T, C);
    tg_fill_randn(X, 0.5f);

    TgSelfAttention attn = tg_attention_create_encoder(C, 1);

    Tensor *Q       = tg_matmul(X, attn.Wq);
    Tensor *K       = tg_matmul(X, attn.Wk);
    Tensor *V       = tg_matmul(X, attn.Wv);
    Tensor *Kt      = tg_transpose(K);
    Tensor *scores  = tg_matmul(Q, Kt);
    Tensor *scaled  = tg_scale(scores, 1.0f / sqrtf((float)C));
    Tensor *weights = tg_softmax_rows(scaled);
    Tensor *O       = tg_matmul(weights, V);
    Tensor *proj    = tg_matmul(O, attn.Wo);

    /* Row sums must be 1.0 */
    for (int i = 0; i < T; i++) {
        float sum = 0.0f;
        for (int j = 0; j < T; j++) sum += weights->data[i * T + j];
        OVG_CHECK_NEAR(sum, 1.0f, 1e-5f);
    }

    /* Upper triangle should NOT be near zero (encoder is bidirectional) */
    int upper_zeros = 0;
    for (int i = 0; i < T; i++)
        for (int j = i + 1; j < T; j++)
            if (weights->data[i * T + j] < 1e-6f) upper_zeros++;
    OVG_CHECK_EQ(upper_zeros, 0);

    tg_free(proj);
    tg_free(O);
    tg_free(weights);
    tg_free(scaled);
    tg_free(scores);
    tg_free(Kt);
    tg_free(V);
    tg_free(K);
    tg_free(Q);
    tg_attention_free(&attn);
    tg_free(X);
}

/* ── Suite entry point ───────────────────────────────────────────────────── */

void run_attention_tests(int *passed, int *failed) {
    RUN_TEST(test_causal_attention_grads,      passed, failed);
    RUN_TEST(test_causal_attention_multihead,  passed, failed);
    RUN_TEST(test_encoder_weights,             passed, failed);
}
