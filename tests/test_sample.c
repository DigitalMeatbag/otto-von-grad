#include "ovg_test.h"
#include "tg_sample.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "tg_rng.h"
#include <string.h>

#ifdef OVG_CUDA_ENABLED
#  include "tg_cuda.h"
#endif

static void test_argmax_basic(void) {
    Tensor *t = tg_new(2, (int[]){1, 3});
    float *td = TG_DATAF(t);
    td[0] = 0.1f; td[1] = 0.9f; td[2] = 0.3f;
    OVG_CHECK_EQ(tg_sample_argmax(t, 0), 1);
    tg_free(t);
}

static void test_topk_1_deterministic(void) {
    Tensor *t = tg_new(2, (int[]){1, 4});
    float *td = TG_DATAF(t);
    td[0] = 0.1f; td[1] = 0.5f; td[2] = 0.9f; td[3] = 0.2f;
    for (int trial = 0; trial < 10; trial++) {
        OVG_CHECK_EQ(tg_sample_topk(t, 0, 0.5f, 1), 2);
        OVG_CHECK_EQ(tg_sample_topk(t, 0, 2.0f, 1), 2);
    }
    tg_free(t);
}

static void test_topk_range(void) {
    Tensor *t = tg_new(2, (int[]){1, 8});
    float *td = TG_DATAF(t);
    for (int j = 0; j < 8; j++) td[j] = (float)j;
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
    tg_seed(42);
    TgGPT gpt  = tg_gpt_create(8, 4, 8, 4, 1, 1);
    TgVocab v  = make_alpha_vocab(8);
    int ids[4] = {0, 1, 2, 3};
    int count  = 0;
    tg_generate(&gpt, &v, ids, 4, 3, 1.0f, 1, count_token, &count);
    OVG_CHECK_EQ(count, 3);
    tg_gpt_free(&gpt);
}

static void test_generate_restores_training_flag(void) {
    tg_seed(42);
    TgGPT gpt  = tg_gpt_create(8, 4, 8, 4, 1, 1);
    TgVocab v  = make_alpha_vocab(8);
    int ids[4] = {0, 1, 2, 3};
    tg_training = 1;
    tg_generate(&gpt, &v, ids, 4, 1, 1.0f, 1, NULL, NULL);
    OVG_CHECK_EQ(tg_training, 1);
    tg_gpt_free(&gpt);
}

#ifdef OVG_CUDA_ENABLED
static void test_argmax_cuda(void) {
    Tensor *t = tg_new(2, (int[]){1, 3});
    float *td = TG_DATAF(t);
    td[0] = 0.1f; td[1] = 0.9f; td[2] = 0.3f;
    tg_to_cuda(t);
    /* Zero host buffer to prove sampling reads from device, not stale host */
    td[0] = 0.0f; td[1] = 0.0f; td[2] = 0.0f;
    OVG_CHECK_EQ(tg_sample_argmax(t, 0), 1);
    tg_cuda_free(t);
    tg_free(t);
}

static void test_topk_cuda(void) {
    Tensor *t = tg_new(2, (int[]){1, 4});
    float *td = TG_DATAF(t);
    td[0] = 0.1f; td[1] = 0.5f; td[2] = 0.9f; td[3] = 0.2f;
    tg_to_cuda(t);
    td[0] = 0.0f; td[1] = 0.0f; td[2] = 0.0f; td[3] = 0.0f;
    for (int trial = 0; trial < 10; trial++)
        OVG_CHECK_EQ(tg_sample_topk(t, 0, 0.5f, 1), 2);
    tg_cuda_free(t);
    tg_free(t);
}
#endif

void run_sample_tests(int *passed, int *failed) {
    RUN_TEST(test_argmax_basic,                    passed, failed);
    RUN_TEST(test_topk_1_deterministic,            passed, failed);
    RUN_TEST(test_topk_range,                      passed, failed);
    RUN_TEST(test_generate_callback_count,         passed, failed);
    RUN_TEST(test_generate_restores_training_flag, passed, failed);
#ifdef OVG_CUDA_ENABLED
    RUN_TEST(test_argmax_cuda,                     passed, failed);
    RUN_TEST(test_topk_cuda,                       passed, failed);
#endif
}
