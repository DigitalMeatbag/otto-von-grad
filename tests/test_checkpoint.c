#include "ovg_test.h"
#include "tg_checkpoint.h"
#include "tg_tensor.h"
#include "tg_gpt.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef OVG_CUDA_ENABLED
#  include "tg_cuda.h"
#endif

#define CKPT_TMP "test_ckpt_tmp.bin"

static void test_checkpoint_roundtrip(void) {
    TgGPT gpt = tg_gpt_create(8, 4, 8, 4, 1, 1);
    Tensor *params[16];
    int n = tg_gpt_collect_params(&gpt, params, 16);

    int nel = tg_numel(params[0]);
    float *orig = malloc((size_t)nel * sizeof(float));
    OVG_CHECK(orig != NULL);
    float *pd = TG_DATAF(params[0]);
    for (int i = 0; i < nel; i++) orig[i] = pd[i];

    OVG_CHECK_EQ(tg_checkpoint_save(CKPT_TMP, params, n), 0);

    for (int i = 0; i < nel; i++) pd[i] = 0.0f;

    OVG_CHECK_EQ(tg_checkpoint_load(CKPT_TMP, params, n), 0);
    pd = TG_DATAF(params[0]);
    for (int i = 0; i < nel; i++) OVG_CHECK_NEAR(pd[i], orig[i], 1e-6f);

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

#ifdef OVG_CUDA_ENABLED
static void test_checkpoint_cuda_roundtrip(void) {
    TgGPT gpt = tg_gpt_create(8, 4, 8, 4, 1, 1);
    Tensor *params[16];
    int n = tg_gpt_collect_params(&gpt, params, 16);

    int nel = tg_numel(params[0]);
    float *orig = malloc((size_t)nel * sizeof(float));
    OVG_CHECK(orig != NULL);
    for (int i = 0; i < nel; i++) orig[i] = TG_DATAF(params[0])[i];

    /* Upload to GPU, then zero the host buffer to prove save reads from device */
    for (int i = 0; i < n; i++) tg_to_cuda(params[i]);
    for (int i = 0; i < nel; i++) TG_DATAF(params[0])[i] = 0.0f;

    OVG_CHECK_EQ(tg_checkpoint_save(CKPT_TMP, params, n), 0);

    /* Corrupt host buffer; load should re-upload correct values to device */
    for (int i = 0; i < nel; i++) TG_DATAF(params[0])[i] = -1.0f;
    OVG_CHECK_EQ(tg_checkpoint_load(CKPT_TMP, params, n), 0);

    /* Sync device → host and verify values were restored */
    tg_from_cuda(params[0]);
    for (int i = 0; i < nel; i++) OVG_CHECK_NEAR(TG_DATAF(params[0])[i], orig[i], 1e-6f);

    free(orig);
    for (int i = 0; i < n; i++) tg_cuda_free(params[i]);
    tg_gpt_free(&gpt);
    remove(CKPT_TMP);
}
#endif

void run_checkpoint_tests(int *passed, int *failed) {
    RUN_TEST(test_checkpoint_roundtrip,      passed, failed);
    RUN_TEST(test_checkpoint_bad_magic,      passed, failed);
    RUN_TEST(test_checkpoint_count_mismatch, passed, failed);
#ifdef OVG_CUDA_ENABLED
    RUN_TEST(test_checkpoint_cuda_roundtrip, passed, failed);
#endif
}
