#include "ovg_test.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "tg_gpt.h"

/* ── collect_params capacity ─────────────────────────────────────────────── */

static void test_collect_params_capacity(void) {
    /* 1 block → need exactly 3 + 1*12 = 15 slots */
    TgGPT gpt = tg_gpt_create(16, 4, 8, 4, 1, 1);
    Tensor *params[15];
    int n = tg_gpt_collect_params(&gpt, params, 15);
    OVG_CHECK_EQ(n, 15);
    tg_gpt_free(&gpt);
}

/* ── GPT forward shape ───────────────────────────────────────────────────── */

static void test_gpt_forward_shape(void) {
    int T = 4, vocab = 8;
    tg_training = 0;

    TgGPT gpt = tg_gpt_create(vocab, 4, 8, T, 1, 1);
    int ids[4] = {0, 1, 2, 3};
    Tensor *logits = tg_gpt_forward(&gpt, ids, 1);

    /* batch_size=1: output is [1*T, vocab] = [T, vocab] */
    OVG_CHECK_SHAPE(logits, T, vocab);

    tg_free_graph(logits);
    tg_gpt_free(&gpt);
}

static void test_gpt_forward_batch2(void) {
    int B = 2, T = 4, vocab = 8;
    tg_training = 0;

    TgGPT gpt = tg_gpt_create(vocab, 4, 8, T, 1, 1);
    /* flat [B*T] token ids */
    int ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    Tensor *logits = tg_gpt_forward(&gpt, ids, B);

    OVG_CHECK_SHAPE(logits, B * T, vocab);

    tg_free_graph(logits);
    tg_gpt_free(&gpt);
}

/* ── Suite entry point ───────────────────────────────────────────────────── */

void run_gpt_tests(int *passed, int *failed) {
    RUN_TEST(test_collect_params_capacity, passed, failed);
    RUN_TEST(test_gpt_forward_shape,       passed, failed);
    RUN_TEST(test_gpt_forward_batch2,      passed, failed);
}
