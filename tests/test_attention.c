#include "ovg_test.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "attention.h"

#include <math.h>
#include <string.h>

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static int any_nonzero(float *arr, int n) {
    for (int i = 0; i < n; i++) if (arr[i] != 0.0f) return 1;
    return 0;
}

/* ── Causal attention: shape + grad coverage (batched [B,T,C]) ───────────── */

static void test_causal_attention_grads(void) {
    int B = 2, T = 4, C = 4, n_heads = 1;
    Tensor *X = tg_new(3, (int[]){B, T, C});
    tg_fill_randn(X, 0.5f);
    X->persistent = 1;

    TgSelfAttention attn = tg_attention_create(C, n_heads);
    TgAttentionForward f = tg_attention_forward_full(&attn, X);

    OVG_CHECK_SHAPE_ND(f.proj, 3, B, T, C);

    Tensor *loss = tg_sum(f.proj);
    tg_backward(loss);

    OVG_CHECK(any_nonzero(attn.Wq->grad, C * C));
    OVG_CHECK(any_nonzero(attn.Wk->grad, C * C));
    OVG_CHECK(any_nonzero(attn.Wv->grad, C * C));
    OVG_CHECK(any_nonzero(attn.Wo->grad, C * C));
    OVG_CHECK(any_nonzero(X->grad, B * T * C));

    tg_free_graph(loss);
    tg_attention_free(&attn);
    X->persistent = 0;
    tg_free(X);
}

static void test_causal_attention_multihead(void) {
    int B = 2, T = 4, C = 4, n_heads = 2;
    Tensor *X = tg_new(3, (int[]){B, T, C});
    tg_fill_randn(X, 0.5f);
    X->persistent = 1;

    TgSelfAttention attn = tg_attention_create(C, n_heads);
    TgAttentionForward f = tg_attention_forward_full(&attn, X);

    OVG_CHECK_SHAPE_ND(f.proj, 3, B, T, C);

    Tensor *loss = tg_sum(f.proj);
    tg_backward(loss);

    OVG_CHECK(any_nonzero(attn.Wq->grad, C * C));
    OVG_CHECK(any_nonzero(X->grad, B * T * C));

    tg_free_graph(loss);
    tg_attention_free(&attn);
    X->persistent = 0;
    tg_free(X);
}

/* ── Encoder attention: no causal mask ───────────────────────────────────── */

static void test_encoder_weights(void) {
    int B = 1, T = 4, C = 4;
    Tensor *X = tg_new(3, (int[]){B, T, C});
    tg_fill_randn(X, 0.5f);

    TgSelfAttention attn = tg_attention_create_encoder(C, 1);
    Tensor *proj = tg_attention_forward(&attn, X);

    OVG_CHECK_SHAPE_ND(proj, 3, B, T, C);

    tg_free(proj);
    tg_attention_free(&attn);
    tg_free(X);
}

/* ── Parity: B=1 batch result must match 2D single-sequence result ─────────── */

static void test_attention_batch1_parity(void) {
    /* Same weights, same input: B=1 batched path should produce identical output
       to the 3D [1,T,C] path (which is what we're testing — both go through N-D). */
    int T = 4, C = 4, n_heads = 2;
    tg_seed(7);

    TgSelfAttention attn = tg_attention_create(C, n_heads);

    /* Run with B=1 (shape [1,T,C]) */
    Tensor *X1 = tg_new(3, (int[]){1, T, C});
    tg_fill_randn(X1, 0.3f);
    Tensor *out1 = tg_attention_forward(&attn, X1);

    /* Run with B=2 where both batches are the same data */
    Tensor *X2 = tg_new(3, (int[]){2, T, C});
    float *x1d = TG_DATAF(X1), *x2d = TG_DATAF(X2);
    for (int i = 0; i < T * C; i++) {
        x2d[i]         = x1d[i];
        x2d[T * C + i] = x1d[i];
    }
    Tensor *out2 = tg_attention_forward(&attn, X2);

    /* Batch 0 of out2 must match out1 (shape [T,C] for each) */
    float *o1d = TG_DATAF(out1), *o2d = TG_DATAF(out2);
    for (int i = 0; i < T * C; i++)
        OVG_CHECK_NEAR(o1d[i], o2d[i], 1e-4f);

    tg_free(out1); tg_free(out2);
    tg_free(X1); tg_free(X2);
    tg_attention_free(&attn);
}

/* ── N-D matmul shape test ───────────────────────────────────────────────── */

static void test_matmul_broadcast(void) {
    /* [B, T, C] @ [C, C_out] → [B, T, C_out] */
    int B = 2, T = 3, C = 4, Co = 5;
    Tensor *A = tg_new(3, (int[]){B, T, C});
    Tensor *W = tg_new(2, (int[]){C, Co});
    A->persistent = W->persistent = 1;
    tg_fill_randn(A, 0.2f);
    tg_fill_randn(W, 0.2f);

    Tensor *out = tg_matmul(A, W);
    OVG_CHECK_SHAPE_ND(out, 3, B, T, Co);

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    OVG_CHECK(any_nonzero(A->grad, B * T * C));
    OVG_CHECK(any_nonzero(W->grad, C * Co));

    tg_free_graph(loss);
    A->persistent = W->persistent = 0;
    tg_free(A); tg_free(W);
}

/* ── Suite entry point ───────────────────────────────────────────────────── */

void run_attention_tests(int *passed, int *failed) {
    RUN_TEST(test_causal_attention_grads,      passed, failed);
    RUN_TEST(test_causal_attention_multihead,  passed, failed);
    RUN_TEST(test_encoder_weights,             passed, failed);
    RUN_TEST(test_attention_batch1_parity,     passed, failed);
    RUN_TEST(test_matmul_broadcast,            passed, failed);
}
