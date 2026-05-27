#include "ovg_test.h"
#include "tg_sample.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "tg_rng.h"
#include <string.h>

static void test_argmax_basic(void) {
    /* Row 0: [0.1, 0.9, 0.3] — argmax should be index 1 */
    Tensor *t = tg_new(1, 3);
    t->data[0] = 0.1f; t->data[1] = 0.9f; t->data[2] = 0.3f;
    OVG_CHECK_EQ(tg_sample_argmax(t, 0), 1);
    tg_free(t);
}

static void test_topk_1_deterministic(void) {
    /* top_k=1 always picks argmax regardless of temperature */
    Tensor *t = tg_new(1, 4);
    t->data[0] = 0.1f; t->data[1] = 0.5f; t->data[2] = 0.9f; t->data[3] = 0.2f;
    for (int trial = 0; trial < 10; trial++) {
        OVG_CHECK_EQ(tg_sample_topk(t, 0, 0.5f, 1), 2);
        OVG_CHECK_EQ(tg_sample_topk(t, 0, 2.0f, 1), 2);
    }
    tg_free(t);
}

static void test_topk_range(void) {
    /* Result must always be a valid token index */
    Tensor *t = tg_new(1, 8);
    for (int j = 0; j < 8; j++) t->data[j] = (float)j;
    for (int trial = 0; trial < 30; trial++) {
        int r = tg_sample_topk(t, 0, 1.0f, 4);
        OVG_CHECK(r >= 0 && r < 8);
    }
    tg_free(t);
}

static void count_token(char c, void *ud) { (void)c; (*(int *)ud)++; }

static TgVocab make_alpha_vocab(int n) {
    TgVocab v;
    memset(v.ids, -1, sizeof(v.ids));
    v.size = n;
    for (int i = 0; i < n; i++) {
        v.chars[i] = (char)('a' + i);
        v.ids[(unsigned char)('a' + i)] = i;
    }
    return v;
}

static void test_generate_callback_count(void) {
    /* Callback must fire exactly `steps` times */
    tg_seed(42);
    TgGPT gpt   = tg_gpt_create(8, 4, 8, 4, 1, 1);
    TgVocab v   = make_alpha_vocab(8);
    int ids[4]  = {0, 1, 2, 3};
    int count   = 0;
    tg_generate(&gpt, &v, ids, 4, 3, 1.0f, 1, count_token, &count);
    OVG_CHECK_EQ(count, 3);
    tg_gpt_free(&gpt);
}

static void test_generate_restores_training_flag(void) {
    /* tg_training must be restored to its pre-call value */
    tg_seed(42);
    TgGPT gpt   = tg_gpt_create(8, 4, 8, 4, 1, 1);
    TgVocab v   = make_alpha_vocab(8);
    int ids[4]  = {0, 1, 2, 3};
    tg_training = 1;
    tg_generate(&gpt, &v, ids, 4, 1, 1.0f, 1, NULL, NULL);
    OVG_CHECK_EQ(tg_training, 1);
    tg_gpt_free(&gpt);
}

void run_sample_tests(int *passed, int *failed) {
    RUN_TEST(test_argmax_basic,                  passed, failed);
    RUN_TEST(test_topk_1_deterministic,          passed, failed);
    RUN_TEST(test_topk_range,                    passed, failed);
    RUN_TEST(test_generate_callback_count,       passed, failed);
    RUN_TEST(test_generate_restores_training_flag, passed, failed);
}
