#include "ovg_test.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "tg_gpt.h"

/* ── collect_params capacity (migrated from smoke_tests.c) ──────────────── */

static void test_collect_params_capacity(void) {
    /* 1 block → need exactly 3 + 1*8 = 11 slots */
    TgGPT gpt = tg_gpt_create(16, 4, 8, 4, 1, 1);
    Tensor *params[11];
    int n = tg_gpt_collect_params(&gpt, params, 11);
    OVG_CHECK_EQ(n, 11);
    tg_gpt_free(&gpt);
}

/* ── GPT forward shape ───────────────────────────────────────────────────── */

static void test_gpt_forward_shape(void) {
    /* Tiny GPT: vocab=8, embed=4, hidden=8, seq=4, blocks=1, heads=1
       forward should return [T x vocab_size] = [4 x 8] */
    int T = 4, vocab = 8;
    tg_training = 0;  /* disable dropout so shapes aren't affected */

    TgGPT gpt = tg_gpt_create(vocab, 4, 8, T, 1, 1);
    int ids[4] = {0, 1, 2, 3};
    Tensor *logits = tg_gpt_forward(&gpt, ids);

    OVG_CHECK_SHAPE(logits, T, vocab);

    tg_free_graph(logits);
    tg_gpt_free(&gpt);
}

/* ── Suite entry point ───────────────────────────────────────────────────── */

void run_gpt_tests(int *passed, int *failed) {
    RUN_TEST(test_collect_params_capacity, passed, failed);
    RUN_TEST(test_gpt_forward_shape,       passed, failed);
}
