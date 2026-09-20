#include "ovg_test.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "tg_optim.h"
#include "tg_sched.h"
#include "tg_rng.h"
#include "ovg_error.h"

#include <setjmp.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef OVG_CUDA_ENABLED
#include "tg_cuda.h"
#endif

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* ── Error-path capture ──────────────────────────────────────────────────── */

static jmp_buf g_test_escape;
static char    g_last_error[512];

static void capture_handler(const char *msg) {
    strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
    longjmp(g_test_escape, 1);
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

/* A tiny "model": two persistent params W [2,3] and b [1,3], filled from a
   deterministic pattern so two copies are bitwise identical. */
static void make_params(Tensor **W, Tensor **b) {
    *W = tg_new(2, (int[]){2, 3});
    *b = tg_new(2, (int[]){1, 3});
    float *wd = TG_DATAF(*W), *bd = TG_DATAF(*b);
    for (int i = 0; i < 6; i++) wd[i] = 0.5f * (float)(i + 1) - 1.0f;
    for (int i = 0; i < 3; i++) bd[i] = 0.25f * (float)i;
    (*W)->persistent = (*b)->persistent = 1;
}

static void free_params(Tensor *W, Tensor *b) {
    W->persistent = b->persistent = 0;
    tg_free(W);
    tg_free(b);
}

/* loss = sum((W * W) * scale) + sum(b * b) — quadratic so grads depend on values. */
static Tensor *make_loss(Tensor *W, Tensor *b, float scale) {
    Tensor *lw = tg_sum(tg_scale(tg_mul(W, W), scale));
    Tensor *lb = tg_sum(tg_mul(b, b));
    return tg_add(lw, lb);
}

static int same_bits(const Tensor *a, const Tensor *b) {
    return memcmp(TG_DATAF(a), TG_DATAF(b), (size_t)tg_numel(a) * sizeof(float)) == 0;
}

/* ── TgAdam ──────────────────────────────────────────────────────────────── */

static void test_adam_parity(void) {
    Tensor *W1, *b1, *W2, *b2;
    make_params(&W1, &b1);
    make_params(&W2, &b2);
    Tensor *p1[2] = {W1, b1};
    Tensor *p2[2] = {W2, b2};

    TgAdam opt = tg_adam_create(p1, 2, 0.9f, 0.999f, 1e-8f);

    float *m[2], *v[2];
    for (int i = 0; i < 2; i++) {
        m[i] = calloc((size_t)tg_numel(p2[i]), sizeof(float));
        v[i] = calloc((size_t)tg_numel(p2[i]), sizeof(float));
    }

    for (int step = 1; step <= 3; step++) {
        Tensor *l1 = make_loss(W1, b1, 1.5f);
        tg_adam_zero_grads(&opt);
        tg_adam_accumulate(&opt, l1, 1);
        tg_adam_update(&opt, 1e-2f, 0.0f);

        Tensor *l2 = make_loss(W2, b2, 1.5f);
        tg_backward(l2);
        tg_adam_step(p2, m, v, 2, 1e-2f, step, 0.9f, 0.999f, 1e-8f);
        tg_free_graph(l2);
    }

    OVG_CHECK(same_bits(W1, W2));
    OVG_CHECK(same_bits(b1, b2));
    OVG_CHECK_EQ(opt.step, 3);

    for (int i = 0; i < 2; i++) { free(m[i]); free(v[i]); }
    tg_adam_free(&opt);
    free_params(W1, b1);
    free_params(W2, b2);
}

static void test_adam_reset(void) {
    Tensor *W, *b;
    make_params(&W, &b);
    Tensor *p[2] = {W, b};
    TgAdam opt = tg_adam_create(p, 2, 0.9f, 0.999f, 1e-8f);

    for (int s = 0; s < 2; s++) {
        Tensor *l = make_loss(W, b, 1.0f);
        tg_adam_zero_grads(&opt);
        tg_adam_accumulate(&opt, l, 1);
        tg_adam_update(&opt, 1e-2f, 0.0f);
    }
    int nonzero = 0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < tg_numel(p[i]); j++)
            if (opt.m[i][j] != 0.0f || opt.v[i][j] != 0.0f) nonzero++;
    OVG_CHECK(nonzero > 0);

    tg_adam_reset(&opt);
    OVG_CHECK_EQ(opt.step, 0);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < tg_numel(p[i]); j++) {
            OVG_CHECK(opt.m[i][j] == 0.0f);
            OVG_CHECK(opt.v[i][j] == 0.0f);
        }

    tg_adam_free(&opt);
    free_params(W, b);
}

static void test_adam_accumulate_equals_batch(void) {
    Tensor *W1, *b1, *W2, *b2;
    make_params(&W1, &b1);
    make_params(&W2, &b2);
    Tensor *p1[2] = {W1, b1};
    TgAdam opt = tg_adam_create(p1, 2, 0.9f, 0.999f, 1e-8f);

    const float scales[4] = {0.5f, 1.0f, 2.0f, 3.5f};

    /* Four micro-steps through the harness. */
    tg_adam_zero_grads(&opt);
    for (int k = 0; k < 4; k++)
        tg_adam_accumulate(&opt, make_loss(W1, b1, scales[k]), 4);

    /* One loss that is the mean of the four, through tg_backward. */
    Tensor *sum = tg_add(tg_add(make_loss(W2, b2, scales[0]), make_loss(W2, b2, scales[1])),
                         tg_add(make_loss(W2, b2, scales[2]), make_loss(W2, b2, scales[3])));
    Tensor *mean = tg_scale(sum, 0.25f);
    tg_backward(mean);

    for (int i = 0; i < tg_numel(W1); i++) OVG_CHECK_NEAR(W1->grad[i], W2->grad[i], 1e-6f);
    for (int i = 0; i < tg_numel(b1); i++) OVG_CHECK_NEAR(b1->grad[i], b2->grad[i], 1e-6f);

    tg_free_graph(mean);
    tg_adam_free(&opt);
    free_params(W1, b1);
    free_params(W2, b2);
}

static void test_adam_update_returns_norm(void) {
    Tensor *W, *b;
    make_params(&W, &b);
    Tensor *p[2] = {W, b};
    TgAdam opt = tg_adam_create(p, 2, 0.9f, 0.999f, 1e-8f);

    /* Clipping: the returned value is the pre-clip norm. */
    Tensor *l = make_loss(W, b, 1.0f);
    tg_adam_zero_grads(&opt);
    tg_adam_accumulate(&opt, l, 1);
    float sumsq = 0.0f;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < tg_numel(p[i]); j++) sumsq += p[i]->grad[j] * p[i]->grad[j];
    float expected = sqrtf(sumsq);
    OVG_CHECK(expected > 1.0f);  /* so the clip is actually exercised */
    float got = tg_adam_update(&opt, 1e-2f, 1.0f);
    OVG_CHECK_NEAR(got, expected, 1e-6f);

    /* No clipping: returns 0 and the step equals an unclipped raw step. */
    Tensor *W2, *b2;
    make_params(&W2, &b2);
    Tensor *p2[2] = {W2, b2};
    float *m[2], *v[2];
    for (int i = 0; i < 2; i++) {
        m[i] = calloc((size_t)tg_numel(p2[i]), sizeof(float));
        v[i] = calloc((size_t)tg_numel(p2[i]), sizeof(float));
    }
    tg_adam_reset(&opt);
    tg_fill(W, 1.0f); tg_fill(b, 1.0f);
    tg_fill(W2, 1.0f); tg_fill(b2, 1.0f);

    Tensor *la = make_loss(W, b, 4.0f);
    tg_adam_zero_grads(&opt);
    tg_adam_accumulate(&opt, la, 1);
    float got0 = tg_adam_update(&opt, 1e-2f, 0.0f);
    OVG_CHECK(got0 == 0.0f);

    Tensor *lb = make_loss(W2, b2, 4.0f);
    tg_backward(lb);
    tg_adam_step(p2, m, v, 2, 1e-2f, 1, 0.9f, 0.999f, 1e-8f);
    tg_free_graph(lb);
    OVG_CHECK(same_bits(W, W2));
    OVG_CHECK(same_bits(b, b2));

    for (int i = 0; i < 2; i++) { free(m[i]); free(v[i]); }
    tg_adam_free(&opt);
    free_params(W, b);
    free_params(W2, b2);
}

static void test_adam_step_counter(void) {
    Tensor *W, *b;
    make_params(&W, &b);
    Tensor *p[2] = {W, b};
    TgAdam opt = tg_adam_create(p, 2, 0.9f, 0.999f, 1e-8f);
    OVG_CHECK_EQ(opt.step, 0);
    for (int s = 1; s <= 7; s++) {
        tg_adam_zero_grads(&opt);
        tg_adam_accumulate(&opt, make_loss(W, b, 1.0f), 1);
        tg_adam_update(&opt, 1e-3f, 0.0f);
        OVG_CHECK_EQ(opt.step, s);
    }
    tg_adam_free(&opt);
    free_params(W, b);
}

static void test_adam_bad_args_fatal(void) {
    Tensor *W, *b;
    make_params(&W, &b);
    Tensor *p[2] = {W, b};

    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);
    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        tg_adam_create(p, 0, 0.9f, 0.999f, 1e-8f);
    } else {
        triggered = 1;
    }
    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "n_params") != NULL);

    TgAdam opt = tg_adam_create(p, 2, 0.9f, 0.999f, 1e-8f);
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);
    triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        tg_adam_accumulate(&opt, make_loss(W, b, 1.0f), 0);
    } else {
        triggered = 1;
    }
    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "n_micro") != NULL);

    tg_adam_free(&opt);
    free_params(W, b);
}

/* ── Schedules ───────────────────────────────────────────────────────────── */

static void test_lr_warmup_cosine(void) {
    /* warmup = 10, total = 100, base = 1, floor = 0.1 */
    OVG_CHECK_NEAR(tg_lr_warmup_cosine(1,   100, 10, 1.0f, 0.1f), 0.1f,   1e-6f);
    OVG_CHECK_NEAR(tg_lr_warmup_cosine(10,  100, 10, 1.0f, 0.1f), 0.982f, 1e-3f);
    OVG_CHECK_NEAR(tg_lr_warmup_cosine(101, 100, 10, 1.0f, 0.1f), 0.1f,   1e-6f);
    OVG_CHECK_NEAR(tg_lr_warmup_cosine(500, 100, 10, 1.0f, 0.1f), 0.1f,   1e-6f);

    float prev = tg_lr_warmup_cosine(10, 100, 10, 1.0f, 0.1f);
    for (int step = 11; step <= 101; step++) {
        float lr = tg_lr_warmup_cosine(step, 100, 10, 1.0f, 0.1f);
        OVG_CHECK(lr <= prev);
        prev = lr;
    }

    /* warmup = 0 never divides by zero. */
    OVG_CHECK_NEAR(tg_lr_warmup_cosine(1, 100, 0, 1.0f, 0.0f), 1.0f, 1e-6f);
    OVG_CHECK(isfinite(tg_lr_warmup_cosine(50, 100, 0, 1.0f, 0.0f)));

    /* vexilloscope grid: total = 60000, warmup = 2400, base = 3e-4, floor = 0.
       Absolute tolerance scaled to base: 1 + cos(pi * progress) cancels
       catastrophically near progress = 1, so a relative bound is unsatisfiable. */
    const int   total  = 60000, warmup = 2400;
    const float base   = 3e-4f;
    const float tol    = 1e-6f * base;
    for (int step = 1; step <= total; step++) {
        float cosine_lr = base * 0.5f * (1.0f + cosf((float)M_PI * (step - 1) / total));
        float warm      = step < warmup ? (float)step / warmup : 1.0f;
        float lr_vex    = cosine_lr * warm;
        float lr_lib    = tg_lr_warmup_cosine(step, total, warmup, base, 0.0f);
        if (fabsf(lr_lib - lr_vex) > tol) {
            fprintf(stderr, "  step %d: lib %.9g vex %.9g\n", step, lr_lib, lr_vex);
            OVG_CHECK(fabsf(lr_lib - lr_vex) <= tol);
        }
    }
}

static void test_lr_warmup_linear(void) {
    OVG_CHECK_NEAR(tg_lr_warmup_linear(1,   100, 10, 1.0f, 0.1f), 0.1f,   1e-6f);
    OVG_CHECK_NEAR(tg_lr_warmup_linear(10,  100, 10, 1.0f, 0.1f), 0.919f, 1e-6f);
    OVG_CHECK_NEAR(tg_lr_warmup_linear(51,  100, 10, 1.0f, 0.1f), 0.55f,  1e-6f);
    OVG_CHECK_NEAR(tg_lr_warmup_linear(101, 100, 10, 1.0f, 0.1f), 0.1f,   1e-6f);
    OVG_CHECK_NEAR(tg_lr_warmup_linear(1,   100, 0,  1.0f, 0.0f), 1.0f,   1e-6f);
}

/* ── Eval guard ──────────────────────────────────────────────────────────── */

static void test_eval_guard(void) {
    tg_training = 1;
    int prev = tg_eval_begin();
    OVG_CHECK_EQ(tg_training, 0);
    OVG_CHECK_EQ(prev, 1);
    tg_eval_end(prev);
    OVG_CHECK_EQ(tg_training, 1);

    TgMeter m = {0};
    OVG_CHECK(tg_meter_mean(&m) == 0.0f);
    tg_meter_add(&m, 1.0f);
    tg_meter_add(&m, 2.0f);
    tg_meter_add(&m, 3.0f);
    OVG_CHECK_EQ(m.n, 3);
    OVG_CHECK_NEAR(tg_meter_mean(&m), 2.0f, 1e-7f);
}

/* ── RNG state ───────────────────────────────────────────────────────────── */

static void test_rng_state_roundtrip(void) {
    uint32_t s = tg_rng_get_state();
    OVG_CHECK(s != 0);
    uint32_t a[3], b[3];
    for (int i = 0; i < 3; i++) a[i] = tg_rng_xorshift32();
    tg_rng_set_state(s);
    for (int i = 0; i < 3; i++) b[i] = tg_rng_xorshift32();
    for (int i = 0; i < 3; i++) OVG_CHECK(a[i] == b[i]);

    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);
    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        tg_rng_set_state(0);
    } else {
        triggered = 1;
    }
    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(tg_rng_get_state() != 0);
}

/* ── CUDA ────────────────────────────────────────────────────────────────── */

#ifdef OVG_CUDA_ENABLED
static void test_adam_cuda_parity(void) {
    Tensor *W1, *b1, *W2, *b2;
    make_params(&W1, &b1);
    make_params(&W2, &b2);
    Tensor *p1[2] = {W1, b1};
    Tensor *p2[2] = {W2, b2};
    tg_to_cuda(W2);
    tg_to_cuda(b2);

    TgAdam cpu = tg_adam_create(p1, 2, 0.9f, 0.999f, 1e-8f);
    TgAdam gpu = tg_adam_create(p2, 2, 0.9f, 0.999f, 1e-8f);
    OVG_CHECK_EQ(cpu.on_cuda, 0);
    OVG_CHECK_EQ(gpu.on_cuda, 1);

    for (int s = 0; s < 5; s++) {
        tg_adam_zero_grads(&cpu);
        tg_adam_accumulate(&cpu, make_loss(W1, b1, 1.5f), 1);
        tg_adam_update(&cpu, 1e-2f, 1.0f);

        tg_adam_zero_grads(&gpu);
        tg_adam_accumulate(&gpu, make_loss(W2, b2, 1.5f), 1);
        tg_adam_update(&gpu, 1e-2f, 1.0f);
    }
    OVG_CHECK_EQ(cpu.step, 5);
    OVG_CHECK_EQ(gpu.step, 5);

    tg_from_cuda(W2);
    tg_from_cuda(b2);
    for (int i = 0; i < tg_numel(W1); i++) OVG_CHECK_NEAR(TG_DATAF(W1)[i], TG_DATAF(W2)[i], 1e-5f);
    for (int i = 0; i < tg_numel(b1); i++) OVG_CHECK_NEAR(TG_DATAF(b1)[i], TG_DATAF(b2)[i], 1e-5f);

    tg_adam_free(&cpu);
    tg_adam_free(&gpu);
    tg_cuda_free(W2);
    tg_cuda_free(b2);
    free_params(W1, b1);
    free_params(W2, b2);
}

static void test_adam_mixed_device_fatal(void) {
    Tensor *W, *b;
    make_params(&W, &b);
    tg_to_cuda(b);
    Tensor *p[2] = {W, b};

    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);
    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        tg_adam_create(p, 2, 0.9f, 0.999f, 1e-8f);
    } else {
        triggered = 1;
    }
    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "mix") != NULL);

    tg_cuda_free(b);
    free_params(W, b);
}
#endif

void run_optim_tests(int *passed, int *failed) {
    RUN_TEST(test_adam_parity,                  passed, failed);
    RUN_TEST(test_adam_reset,                   passed, failed);
    RUN_TEST(test_adam_accumulate_equals_batch, passed, failed);
    RUN_TEST(test_adam_update_returns_norm,     passed, failed);
    RUN_TEST(test_adam_step_counter,            passed, failed);
    RUN_TEST(test_adam_bad_args_fatal,          passed, failed);
    RUN_TEST(test_lr_warmup_cosine,             passed, failed);
    RUN_TEST(test_lr_warmup_linear,             passed, failed);
    RUN_TEST(test_eval_guard,                   passed, failed);
    RUN_TEST(test_rng_state_roundtrip,          passed, failed);
#ifdef OVG_CUDA_ENABLED
    RUN_TEST(test_adam_cuda_parity,             passed, failed);
    RUN_TEST(test_adam_mixed_device_fatal,      passed, failed);
#endif
}
