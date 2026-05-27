#include "ovg_test.h"
#include "tg_checkpoint.h"
#include "tg_tensor.h"
#include "tg_gpt.h"
#include <stdlib.h>
#include <stdio.h>

#define CKPT_TMP "test_ckpt_tmp.bin"

static void test_checkpoint_roundtrip(void) {
    TgGPT gpt = tg_gpt_create(8, 4, 8, 4, 1, 1);
    Tensor *params[16];
    int n = tg_gpt_collect_params(&gpt, params, 16);

    int nel   = params[0]->rows * params[0]->cols;
    float *orig = malloc((size_t)nel * sizeof(float));
    OVG_CHECK(orig != NULL);
    for (int i = 0; i < nel; i++) orig[i] = params[0]->data[i];

    OVG_CHECK_EQ(tg_checkpoint_save(CKPT_TMP, params, n), 0);

    for (int i = 0; i < nel; i++) params[0]->data[i] = 0.0f;

    OVG_CHECK_EQ(tg_checkpoint_load(CKPT_TMP, params, n), 0);
    for (int i = 0; i < nel; i++) OVG_CHECK_NEAR(params[0]->data[i], orig[i], 1e-6f);

    free(orig);
    tg_gpt_free(&gpt);
    remove(CKPT_TMP);
}

static void test_checkpoint_bad_magic(void) {
    FILE *f = fopen(CKPT_TMP, "wb");
    OVG_CHECK(f != NULL);
    unsigned char bad[4] = {0xDE, 0xAD, 0xBE, 0xEF};
    fwrite(bad, 1, 4, f);
    fclose(f);

    TgGPT gpt = tg_gpt_create(8, 4, 8, 4, 1, 1);
    Tensor *params[16];
    int n = tg_gpt_collect_params(&gpt, params, 16);
    OVG_CHECK_EQ(tg_checkpoint_load(CKPT_TMP, params, n), -1);

    tg_gpt_free(&gpt);
    remove(CKPT_TMP);
}

static void test_checkpoint_count_mismatch(void) {
    TgGPT gpt = tg_gpt_create(8, 4, 8, 4, 1, 1);
    Tensor *params[16];
    int n = tg_gpt_collect_params(&gpt, params, 16);

    tg_checkpoint_save(CKPT_TMP, params, n);
    OVG_CHECK_EQ(tg_checkpoint_load(CKPT_TMP, params, n + 1), -1);

    tg_gpt_free(&gpt);
    remove(CKPT_TMP);
}

void run_checkpoint_tests(int *passed, int *failed) {
    RUN_TEST(test_checkpoint_roundtrip,      passed, failed);
    RUN_TEST(test_checkpoint_bad_magic,      passed, failed);
    RUN_TEST(test_checkpoint_count_mismatch, passed, failed);
}
