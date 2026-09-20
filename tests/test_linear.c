#include "ovg_test.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "tg_linear.h"
#include "ovg_error.h"

#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#ifdef OVG_CUDA_ENABLED
#include "tg_cuda.h"
#endif

/* ── Error-path capture ──────────────────────────────────────────────────── */

static jmp_buf g_test_escape;

static void capture_handler(const char *msg) {
    (void)msg;
    longjmp(g_test_escape, 1);
}

/* ── Fixture: tg_linear_create(4, 3, 0.1f) with B = {1, 2, 3} ─────────────── */

#define N_IN  4
#define N_OUT 3

static TgLinear make_layer(void) {
    TgLinear l = tg_linear_create(N_IN, N_OUT, 0.1f);
    for (int j = 0; j < N_OUT; j++) TG_DATAF(l.B)[j] = (float)(j + 1);
    return l;
}

/* Hand reference: out[i][j] = sum_k x[i][k] W[k][j] + B[j] over `rows` rows of x. */
static float ref_linear(const TgLinear *l, const float *x, int i, int j) {
    float acc = TG_DATAF(l->B)[j];
    for (int k = 0; k < N_IN; k++) acc += x[i * N_IN + k] * TG_DATAF(l->W)[k * N_OUT + j];
    return acc;
}

/* ── Forward, 2D ─────────────────────────────────────────────────────────── */

static void test_linear_forward_2d(void) {
    TgLinear l = make_layer();
    Tensor *x = tg_new(2, (int[]){5, N_IN});
    tg_fill_randn(x, 1.0f);
    x->persistent = 1;

    Tensor *out = tg_linear_forward(&l, x);
    OVG_CHECK_SHAPE_ND(out, 2, 5, N_OUT);

    for (int i = 0; i < 5; i++)
        for (int j = 0; j < N_OUT; j++)
            OVG_CHECK_NEAR(TG_DATAF(out)[i * N_OUT + j], ref_linear(&l, TG_DATAF(x), i, j), 1e-6f);

    tg_free_graph(out);
    tg_free(x);
    tg_linear_free(&l);
}

/* ── Forward, 3D: each batch equals the 2D forward of that slice ─────────── */

static void test_linear_forward_3d(void) {
    TgLinear l = make_layer();
    int B = 2, T = 5;
    Tensor *x = tg_new(3, (int[]){B, T, N_IN});
    tg_fill_randn(x, 1.0f);
    x->persistent = 1;

    Tensor *out = tg_linear_forward(&l, x);
    OVG_CHECK_SHAPE_ND(out, 3, B, T, N_OUT);

    for (int b = 0; b < B; b++) {
        Tensor *xb = tg_new(2, (int[]){T, N_IN});
        memcpy(TG_DATAF(xb), TG_DATAF(x) + b * T * N_IN, (size_t)T * N_IN * sizeof(float));
        xb->persistent = 1;
        Tensor *ob = tg_linear_forward(&l, xb);
        for (int i = 0; i < T * N_OUT; i++)
            OVG_CHECK_NEAR(TG_DATAF(out)[b * T * N_OUT + i], TG_DATAF(ob)[i], 1e-6f);
        tg_free_graph(ob);
        tg_free(xb);
    }

    tg_free_graph(out);
    tg_free(x);
    tg_linear_free(&l);
}

/* ── Backward of sum(out) ────────────────────────────────────────────────── */

static void test_linear_backward(void) {
    TgLinear l = make_layer();
    int B = 2, T = 5;
    Tensor *x = tg_new(3, (int[]){B, T, N_IN});
    tg_fill_randn(x, 1.0f);
    x->persistent = 1;

    Tensor *out  = tg_linear_forward(&l, x);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    /* dB[j] = B*T exactly */
    for (int j = 0; j < N_OUT; j++) OVG_CHECK(l.B->grad[j] == (float)(B * T));

    /* dW[k][j] = sum_{b,t} x[b][t][k] */
    for (int k = 0; k < N_IN; k++) {
        float s = 0.0f;
        for (int bt = 0; bt < B * T; bt++) s += TG_DATAF(x)[bt * N_IN + k];
        for (int j = 0; j < N_OUT; j++) OVG_CHECK_NEAR(l.W->grad[k * N_OUT + j], s, 1e-5f);
    }

    /* dx[b][t][k] = sum_j W[k][j] */
    for (int k = 0; k < N_IN; k++) {
        float s = 0.0f;
        for (int j = 0; j < N_OUT; j++) s += TG_DATAF(l.W)[k * N_OUT + j];
        for (int bt = 0; bt < B * T; bt++) OVG_CHECK_NEAR(x->grad[bt * N_IN + k], s, 1e-5f);
    }

    tg_free_graph(loss);
    tg_free(x);
    tg_linear_free(&l);
}

/* ── Params ──────────────────────────────────────────────────────────────── */

static void test_linear_params(void) {
    TgLinear l = tg_linear_create(N_IN, N_OUT, 0.1f);
    int n = 0;
    Tensor **p = tg_linear_params(&l, &n);
    OVG_CHECK_EQ(n, 2);
    OVG_CHECK(p[0] == l.W);
    OVG_CHECK(p[1] == l.B);
    OVG_CHECK_SHAPE_ND(l.B, 2, 1, N_OUT);
    OVG_CHECK_SHAPE_ND(l.W, 2, N_IN, N_OUT);
    OVG_CHECK(l.W->persistent == 1);
    OVG_CHECK(l.B->persistent == 1);
    free(p);
    tg_linear_free(&l);
}

/* ── Fatal paths ─────────────────────────────────────────────────────────── */

static void test_linear_bad_input_fatal(void) {
    /* x last dim 3 into n_in = 4 */
    int triggered = 0;
    TgLinear l = tg_linear_create(N_IN, N_OUT, 0.1f);
    ovg_set_fatal_handler(capture_handler);
    if (setjmp(g_test_escape) == 0) {
        Tensor *x = tg_new(2, (int[]){5, 3});
        tg_linear_forward(&l, x);
    } else {
        triggered = 1;
    }
    tg_linear_free(&l);
    OVG_CHECK(triggered);

    /* n_in = 0 */
    triggered = 0;
    ovg_set_fatal_handler(capture_handler);   /* re-arm: longjmp skipped the reset */
    if (setjmp(g_test_escape) == 0) {
        tg_linear_create(0, N_OUT, 0.1f);
    } else {
        triggered = 1;
    }
    OVG_CHECK(triggered);

    ovg_set_fatal_handler(NULL);
}

/* ── CUDA parity ─────────────────────────────────────────────────────────── */

#ifdef OVG_CUDA_ENABLED
static void test_linear_cuda_parity(void) {
    TgLinear l = make_layer();
    int B = 2, T = 5;
    Tensor *x = tg_new(3, (int[]){B, T, N_IN});
    tg_fill_randn(x, 1.0f);
    x->persistent = 1;

    /* Host run */
    Tensor *out_h  = tg_linear_forward(&l, x);
    Tensor *loss_h = tg_sum(out_h);
    tg_backward(loss_h);
    int n_out = tg_numel(out_h);
    float *ref_out = malloc((size_t)n_out * sizeof(float));
    float *ref_dW  = malloc((size_t)N_IN * N_OUT * sizeof(float));
    float *ref_dB  = malloc((size_t)N_OUT * sizeof(float));
    OVG_CHECK(ref_out && ref_dW && ref_dB);
    memcpy(ref_out, TG_DATAF(out_h), (size_t)n_out * sizeof(float));
    memcpy(ref_dW, l.W->grad, (size_t)N_IN * N_OUT * sizeof(float));
    memcpy(ref_dB, l.B->grad, (size_t)N_OUT * sizeof(float));
    tg_free_graph(loss_h);

    /* Device run */
    tg_to_cuda(l.W); tg_to_cuda(l.B); tg_to_cuda(x);
    Tensor *out_d  = tg_linear_forward(&l, x);
    Tensor *loss_d = tg_sum(out_d);
    tg_backward(loss_d);
    tg_from_cuda(out_d); tg_from_cuda(l.W); tg_from_cuda(l.B);

    for (int i = 0; i < n_out; i++)       OVG_CHECK_NEAR(TG_DATAF(out_d)[i], ref_out[i], 1e-5f);
    for (int i = 0; i < N_IN * N_OUT; i++) OVG_CHECK_NEAR(l.W->grad[i], ref_dW[i], 1e-5f);
    for (int i = 0; i < N_OUT; i++)        OVG_CHECK_NEAR(l.B->grad[i], ref_dB[i], 1e-5f);

    free(ref_out); free(ref_dW); free(ref_dB);
    tg_free_graph(loss_d);
    tg_cuda_free(x); tg_cuda_free(l.W); tg_cuda_free(l.B);
    tg_free(x);
    tg_linear_free(&l);
}
#endif

/* ── Suite entry point ───────────────────────────────────────────────────── */

void run_linear_tests(int *passed, int *failed) {
    RUN_TEST(test_linear_forward_2d,      passed, failed);
    RUN_TEST(test_linear_forward_3d,      passed, failed);
    RUN_TEST(test_linear_backward,        passed, failed);
    RUN_TEST(test_linear_params,          passed, failed);
    RUN_TEST(test_linear_bad_input_fatal, passed, failed);
#ifdef OVG_CUDA_ENABLED
    RUN_TEST(test_linear_cuda_parity,     passed, failed);
#endif
}
