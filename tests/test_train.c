#include "ovg_test.h"
#include "tg_ops.h"
#include "tg_train.h"

#ifdef OVG_CUDA_ENABLED
#include "tg_cuda.h"
#endif

/* ── SGD ─────────────────────────────────────────────────────────────────── */

static void test_sgd_direction(void) {
    /* W = [[3.0]], loss = sum(W), grad = 1.0
       after sgd step with lr=0.5: W = 3.0 - 0.5*1.0 = 2.5 */
    Tensor *W = tg_new(1, 1);
    W->data[0] = 3.0f;
    W->persistent = 1;

    Tensor *loss = tg_sum(W);
    tg_backward(loss);
    tg_free_graph(loss);

    Tensor *params[1] = {W};
    tg_sgd_step(params, 1, 0.5f);

    OVG_CHECK_NEAR(W->data[0], 2.5f, 1e-5f);

    W->persistent = 0;
    tg_free(W);
}

/* ── backward zeros grads ────────────────────────────────────────────────── */

static void test_backward_zeros_grads(void) {
    /* Verify tg_backward resets grads before accumulating.
       Pass 1: loss1 = sum(scale(a, 1)) → a.grad = 1.0
       Pass 2: loss2 = sum(scale(a, 3)) → a.grad = 3.0 (not 4.0) */
    Tensor *a = tg_new(1, 2);
    a->data[0] = 1.0f; a->data[1] = 1.0f;
    a->persistent = 1;

    Tensor *loss1 = tg_sum(tg_scale(a, 1.0f));
    tg_backward(loss1);
    tg_free_graph(loss1);

    OVG_CHECK_NEAR(a->grad[0], 1.0f, 1e-5f);

    Tensor *loss2 = tg_sum(tg_scale(a, 3.0f));
    tg_backward(loss2);
    tg_free_graph(loss2);

    /* tg_backward zeros grads first, so grad should be 3.0, not 4.0 */
    OVG_CHECK_NEAR(a->grad[0], 3.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 3.0f, 1e-5f);

    a->persistent = 0;
    tg_free(a);
}

/* ── backward_accum accumulates without zeroing ──────────────────────────── */

static void test_backward_accum(void) {
    /* Two identical backward passes via tg_backward_accum should
       leave grads at 2× a single pass. */
    Tensor *a = tg_new(1, 2);
    a->data[0] = 1.0f; a->data[1] = 2.0f;
    a->persistent = 1;

    Tensor *loss1 = tg_sum(a);
    Tensor *loss2 = tg_sum(a);

    Tensor *params[1] = {a};
    tg_zero_grads(params, 1);

    tg_backward_accum(loss1);  /* a.grad = 1.0 */
    tg_backward_accum(loss2);  /* a.grad = 2.0 (accumulated, not reset) */

    OVG_CHECK_NEAR(a->grad[0], 2.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 2.0f, 1e-5f);

    tg_free_graph(loss1);
    tg_free_graph(loss2);
    a->persistent = 0;
    tg_free(a);
}

/* ── Transpose grad accumulation (migrated from smoke_tests.c) ───────────── */

static void test_transpose_grad_accum(void) {
    /* A used as parent of two independent transposes.
       Both grads must accumulate into A.grad, not overwrite. */
    Tensor *A = tg_new(3, 4);
    tg_fill(A, 1.0f);
    A->persistent = 1;

    Tensor *B    = tg_transpose(A);
    Tensor *C    = tg_transpose(A);
    Tensor *loss = tg_add(tg_sum(B), tg_sum(C));
    tg_backward(loss);

    int n = A->rows * A->cols;
    for (int i = 0; i < n; i++)
        OVG_CHECK_NEAR(A->grad[i], 2.0f, 1e-5f);

    tg_free_graph(loss);

#ifdef OVG_CUDA_ENABLED
    tg_to_cuda(A);

    Tensor *Bg    = tg_transpose(A);
    Tensor *Cg    = tg_transpose(A);
    Tensor *lossg = tg_add(tg_sum(Bg), tg_sum(Cg));
    tg_backward(lossg);
    tg_from_cuda(A);

    for (int i = 0; i < n; i++)
        OVG_CHECK_NEAR(A->grad[i], 2.0f, 1e-5f);

    tg_free_graph(lossg);
    tg_cuda_free(A);
#endif

    A->persistent = 0;
    tg_free(A);
}

static void test_clip_grad_norm(void) {
    Tensor *a = tg_new(1, 2);
    Tensor *b = tg_new(1, 1);
    a->grad[0] = 3.0f;
    a->grad[1] = 4.0f;
    b->grad[0] = 0.0f;

    Tensor *params[2] = {a, b};
    float norm = tg_clip_grad_norm(params, 2, 10.0f, 1e-6f);
    OVG_CHECK_NEAR(norm, 5.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[0], 3.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 4.0f, 1e-5f);

    norm = tg_clip_grad_norm(params, 2, 2.5f, 1e-6f);
    OVG_CHECK_NEAR(norm, 5.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[0], 1.5f, 1e-4f);
    OVG_CHECK_NEAR(a->grad[1], 2.0f, 1e-4f);

    tg_free(a);
    tg_free(b);
}

#ifdef OVG_CUDA_ENABLED
static void test_cuda_clip_grad_norm(void) {
    Tensor *a = tg_new(1, 2);
    a->grad[0] = 6.0f;
    a->grad[1] = 8.0f;
    tg_to_cuda(a);

    Tensor *params[1] = {a};
    float norm = tg_clip_grad_norm(params, 1, 5.0f, 1e-6f);
    tg_from_cuda(a);

    OVG_CHECK_NEAR(norm, 10.0f, 1e-4f);
    OVG_CHECK_NEAR(a->grad[0], 3.0f, 1e-4f);
    OVG_CHECK_NEAR(a->grad[1], 4.0f, 1e-4f);

    tg_cuda_free(a);
    tg_free(a);
}
#endif

/* ── Suite entry point ───────────────────────────────────────────────────── */

void run_train_tests(int *passed, int *failed) {
    RUN_TEST(test_sgd_direction,        passed, failed);
    RUN_TEST(test_backward_zeros_grads, passed, failed);
    RUN_TEST(test_backward_accum,       passed, failed);
    RUN_TEST(test_transpose_grad_accum, passed, failed);
    RUN_TEST(test_clip_grad_norm,       passed, failed);
#ifdef OVG_CUDA_ENABLED
    RUN_TEST(test_cuda_clip_grad_norm,  passed, failed);
#endif
}
