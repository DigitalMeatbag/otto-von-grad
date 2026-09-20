#include "ovg_test.h"
#include "tg_checkpoint.h"
#include "tg_tensor.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "tg_optim.h"
#include "tg_rng.h"
#include "tg_gpt.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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

/* ── v3 run-state tests ──────────────────────────────────────────────────── */

#define CKPT_TMP_TMP CKPT_TMP ".tmp"
#define RUN_TOKENS   4   /* seq_len of the test model */

/* Small GPT + TgAdam fixture. Params stay on the host unless the caller asks
   for the device; either way the optimizer is created after placement. */
typedef struct {
    TgGPT    gpt;
    Tensor  *params[16];
    int      n;
    TgAdam   opt;
} RunFixture;

static void fixture_create(RunFixture *fx, float dropout, int on_cuda) {
    fx->gpt = tg_gpt_create(8, 4, 8, RUN_TOKENS, 1, 1);
    for (int i = 0; i < fx->gpt.transformer.n_blocks; i++)
        fx->gpt.transformer.blocks[i].dropout = dropout;
    fx->n = tg_gpt_collect_params(&fx->gpt, fx->params, 16);
#ifdef OVG_CUDA_ENABLED
    if (on_cuda)
        for (int i = 0; i < fx->n; i++) tg_to_cuda(fx->params[i]);
#else
    (void)on_cuda;
#endif
    fx->opt = tg_adam_create(fx->params, fx->n, 0.9f, 0.999f, 1e-8f);
}

static void fixture_free(RunFixture *fx) {
    tg_adam_free(&fx->opt);
#ifdef OVG_CUDA_ENABLED
    for (int i = 0; i < fx->n; i++) tg_cuda_free(fx->params[i]);
#endif
    tg_gpt_free(&fx->gpt);
}

/* One training step on fixed data. */
static void fixture_step(RunFixture *fx) {
    static const int ids[RUN_TOKENS]  = {1, 2, 3, 4};
    static const int tgts[RUN_TOKENS] = {2, 3, 4, 5};
    Tensor *hot = tg_new(2, (int[]){RUN_TOKENS, 8});
    for (int i = 0; i < RUN_TOKENS; i++) TG_DATAF(hot)[i * 8 + tgts[i]] = 1.0f;
#ifdef OVG_CUDA_ENABLED
    if (fx->opt.on_cuda) tg_to_cuda(hot);
#endif
    Tensor *logits = tg_gpt_forward(&fx->gpt, ids, 1);
    Tensor *loss   = tg_cross_entropy(logits, hot);
    tg_adam_zero_grads(&fx->opt);
    tg_adam_accumulate(&fx->opt, loss, 1);
    tg_adam_update(&fx->opt, 1e-2f, 0.0f);
}

/* Host copies of every param and (host-resident) moment, for before/after
   comparisons. Device moments are compared through a save/load instead. */
typedef struct { float **data, **m, **v; int n; } Snapshot;

static Snapshot snapshot_take(RunFixture *fx) {
    Snapshot s;
    s.n    = fx->n;
    s.data = calloc((size_t)fx->n, sizeof(float *));
    s.m    = calloc((size_t)fx->n, sizeof(float *));
    s.v    = calloc((size_t)fx->n, sizeof(float *));
    for (int i = 0; i < fx->n; i++) {
        size_t nel = (size_t)tg_numel(fx->params[i]);
#ifdef OVG_CUDA_ENABLED
        if (fx->opt.on_cuda) tg_from_cuda(fx->params[i]);
#endif
        s.data[i] = malloc(nel * sizeof(float));
        s.m[i]    = calloc(nel, sizeof(float));
        s.v[i]    = calloc(nel, sizeof(float));
        memcpy(s.data[i], TG_DATAF(fx->params[i]), nel * sizeof(float));
        if (!fx->opt.on_cuda) {
            memcpy(s.m[i], fx->opt.m[i], nel * sizeof(float));
            memcpy(s.v[i], fx->opt.v[i], nel * sizeof(float));
        }
    }
    return s;
}

static void snapshot_free(Snapshot *s) {
    for (int i = 0; i < s->n; i++) { free(s->data[i]); free(s->m[i]); free(s->v[i]); }
    free(s->data); free(s->m); free(s->v);
}

/* tol == 0 means bitwise. */
static int snapshot_params_equal(const Snapshot *s, RunFixture *fx, float tol) {
    for (int i = 0; i < fx->n; i++) {
#ifdef OVG_CUDA_ENABLED
        if (fx->opt.on_cuda) tg_from_cuda(fx->params[i]);
#endif
        int nel = tg_numel(fx->params[i]);
        for (int j = 0; j < nel; j++) {
            float a = s->data[i][j], b = TG_DATAF(fx->params[i])[j];
            if (tol == 0.0f ? (a != b) : (fabsf(a - b) > tol)) return 0;
        }
    }
    return 1;
}

static int snapshot_moments_equal(const Snapshot *s, RunFixture *fx) {
    for (int i = 0; i < fx->n; i++) {
        size_t nel = (size_t)tg_numel(fx->params[i]);
        if (memcmp(s->m[i], fx->opt.m[i], nel * sizeof(float)) != 0) return 0;
        if (memcmp(s->v[i], fx->opt.v[i], nel * sizeof(float)) != 0) return 0;
    }
    return 1;
}

static int moments_all_zero(RunFixture *fx) {
    for (int i = 0; i < fx->n; i++) {
        int nel = tg_numel(fx->params[i]);
        for (int j = 0; j < nel; j++)
            if (fx->opt.m[i][j] != 0.0f || fx->opt.v[i][j] != 0.0f) return 0;
    }
    return 1;
}

/* Scribble over params, host moments, and the step counter. */
static void corrupt_fixture(RunFixture *fx) {
    for (int i = 0; i < fx->n; i++) {
        int nel = tg_numel(fx->params[i]);
        for (int j = 0; j < nel; j++) TG_DATAF(fx->params[i])[j] = -7.0f;
        if (!fx->opt.on_cuda)
            for (int j = 0; j < nel; j++) { fx->opt.m[i][j] = 9.0f; fx->opt.v[i][j] = 9.0f; }
#ifdef OVG_CUDA_ENABLED
        if (fx->opt.on_cuda) tg_to_cuda(fx->params[i]);
#endif
    }
    fx->opt.step = 99;
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void test_checkpoint_v3_run_roundtrip(void) {
    RunFixture fx;
    fixture_create(&fx, 0.0f, 0);
    for (int s = 0; s < 3; s++) fixture_step(&fx);
    OVG_CHECK_EQ(fx.opt.step, 3);

    tg_rng_set_state(0xC0FFEE);
    Snapshot before = snapshot_take(&fx);
    OVG_CHECK_EQ(tg_checkpoint_save_run(CKPT_TMP, &fx.opt, "cfg!", 4), 0);

    corrupt_fixture(&fx);
    tg_rng_set_state(0x1234);

    TgCheckpointInfo info;
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &fx.opt, TG_LOAD_RESUME, &info), 0);
    OVG_CHECK(snapshot_params_equal(&before, &fx, 0.0f));
    OVG_CHECK(snapshot_moments_equal(&before, &fx));
    OVG_CHECK_EQ(fx.opt.step, 3);
    OVG_CHECK(tg_rng_get_state() == 0xC0FFEE);
    OVG_CHECK_EQ(info.version, 3);
    OVG_CHECK_EQ(info.has_optimizer, 1);
    OVG_CHECK_EQ(info.config_len, 4);
    OVG_CHECK_EQ(info.n_params, fx.n);

    snapshot_free(&before);
    fixture_free(&fx);
    remove(CKPT_TMP);
}

static void test_checkpoint_info_before_model(void) {
    RunFixture fx;
    fixture_create(&fx, 0.0f, 0);
    for (int s = 0; s < 3; s++) fixture_step(&fx);
    tg_rng_set_state(0xC0FFEE);
    OVG_CHECK_EQ(tg_checkpoint_save_run(CKPT_TMP, &fx.opt, "cfg!", 4), 0);
    int n = fx.n;
    fixture_free(&fx);

    /* No tensors exist now. */
    char buf[16];
    memset(buf, 0, sizeof buf);
    TgCheckpointInfo info;
    OVG_CHECK_EQ(tg_checkpoint_info(CKPT_TMP, &info, buf, (int)sizeof buf), 0);
    OVG_CHECK(memcmp(buf, "cfg!", 4) == 0);
    OVG_CHECK_EQ(buf[4], 0);
    OVG_CHECK_EQ(info.config_len, 4);
    OVG_CHECK_EQ(info.n_params, n);
    OVG_CHECK_EQ(info.has_optimizer, 1);
    OVG_CHECK_EQ(info.step, 3);
    OVG_CHECK(info.rng_state == 0xC0FFEE);
    OVG_CHECK_EQ(info.version, 3);

    /* Zero-capacity call with a NULL buffer is fine. */
    OVG_CHECK_EQ(tg_checkpoint_info(CKPT_TMP, &info, NULL, 0), 0);
    OVG_CHECK_EQ(info.config_len, 4);

    remove(CKPT_TMP);
    OVG_CHECK_EQ(tg_checkpoint_info(CKPT_TMP, &info, NULL, 0), -1);
}

static void test_checkpoint_v2_loads_with_reset(void) {
    RunFixture fx;
    fixture_create(&fx, 0.0f, 0);
    for (int s = 0; s < 2; s++) fixture_step(&fx);
    Snapshot before = snapshot_take(&fx);

    /* Write a v2 file by hand: magic, count, then each param. */
    FILE *f = fopen(CKPT_TMP, "wb");
    OVG_CHECK(f != NULL);
    uint32_t magic = TG_CHECKPOINT_MAGIC_V2;
    int32_t  cnt   = (int32_t)fx.n;
    fwrite(&magic, sizeof magic, 1, f);
    fwrite(&cnt,   sizeof cnt,   1, f);
    for (int i = 0; i < fx.n; i++) {
        int32_t nd = (int32_t)fx.params[i]->ndim;
        int32_t sh[TG_MAX_DIMS] = {0};
        for (int d = 0; d < nd; d++) sh[d] = (int32_t)fx.params[i]->shape[d];
        fwrite(&nd, sizeof nd, 1, f);
        fwrite(sh,  sizeof sh, 1, f);
        fwrite(TG_DATAF(fx.params[i]), sizeof(float), (size_t)tg_numel(fx.params[i]), f);
    }
    fclose(f);

    corrupt_fixture(&fx);
    tg_rng_set_state(0xABCD);

    TgCheckpointInfo info;
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &fx.opt, TG_LOAD_RESUME, &info), 0);
    OVG_CHECK(snapshot_params_equal(&before, &fx, 0.0f));
    OVG_CHECK(moments_all_zero(&fx));
    OVG_CHECK_EQ(fx.opt.step, 0);
    OVG_CHECK(tg_rng_get_state() == 0xABCD);
    OVG_CHECK_EQ(info.version, 2);
    OVG_CHECK_EQ(info.has_optimizer, 0);
    OVG_CHECK_EQ(info.n_params, fx.n);

    /* The weights-only reader accepts it too. */
    corrupt_fixture(&fx);
    OVG_CHECK_EQ(tg_checkpoint_load(CKPT_TMP, fx.params, fx.n), 0);
    OVG_CHECK(snapshot_params_equal(&before, &fx, 0.0f));

    snapshot_free(&before);
    fixture_free(&fx);
    remove(CKPT_TMP);
}

static void test_checkpoint_init_from_weights(void) {
    RunFixture fx;
    fixture_create(&fx, 0.0f, 0);
    for (int s = 0; s < 3; s++) fixture_step(&fx);
    Snapshot before = snapshot_take(&fx);
    OVG_CHECK_EQ(tg_checkpoint_save_run(CKPT_TMP, &fx.opt, NULL, 0), 0);

    corrupt_fixture(&fx);
    tg_rng_set_state(0x5555);

    TgCheckpointInfo info;
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &fx.opt, TG_LOAD_INIT_FROM_WEIGHTS, &info), 0);
    OVG_CHECK(snapshot_params_equal(&before, &fx, 0.0f));
    OVG_CHECK(moments_all_zero(&fx));
    OVG_CHECK_EQ(fx.opt.step, 0);
    OVG_CHECK(tg_rng_get_state() == 0x5555);
    OVG_CHECK_EQ(info.has_optimizer, 1);   /* the file has it; we chose not to use it */
    OVG_CHECK_EQ(info.step, 3);

    snapshot_free(&before);
    fixture_free(&fx);
    remove(CKPT_TMP);
}

static void test_checkpoint_weights_only_then_run(void) {
    RunFixture fx;
    fixture_create(&fx, 0.0f, 0);
    for (int s = 0; s < 2; s++) fixture_step(&fx);
    Snapshot before = snapshot_take(&fx);

    /* Weights-only save, run-state load. */
    OVG_CHECK_EQ(tg_checkpoint_save(CKPT_TMP, fx.params, fx.n), 0);
    corrupt_fixture(&fx);
    tg_rng_set_state(0x7777);
    TgCheckpointInfo info;
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &fx.opt, TG_LOAD_RESUME, &info), 0);
    OVG_CHECK_EQ(info.version, 3);
    OVG_CHECK_EQ(info.has_optimizer, 0);
    OVG_CHECK(info.rng_state == 0);
    OVG_CHECK_EQ(info.step, 0);
    OVG_CHECK(snapshot_params_equal(&before, &fx, 0.0f));
    OVG_CHECK(moments_all_zero(&fx));
    OVG_CHECK_EQ(fx.opt.step, 0);
    OVG_CHECK(tg_rng_get_state() == 0x7777);

    /* Run-state save, weights-only load. */
    for (int s = 0; s < 2; s++) fixture_step(&fx);
    snapshot_free(&before);
    before = snapshot_take(&fx);
    OVG_CHECK_EQ(tg_checkpoint_save_run(CKPT_TMP, &fx.opt, "abc", 3), 0);
    corrupt_fixture(&fx);
    OVG_CHECK_EQ(tg_checkpoint_load(CKPT_TMP, fx.params, fx.n), 0);
    OVG_CHECK(snapshot_params_equal(&before, &fx, 0.0f));
    OVG_CHECK_EQ(fx.opt.step, 99);   /* untouched by the weights-only reader */

    snapshot_free(&before);
    fixture_free(&fx);
    remove(CKPT_TMP);
}

/* Overwrite 4 bytes at `offset` in the file. */
static void patch_file(const char *path, long offset, const void *bytes) {
    FILE *f = fopen(path, "r+b");
    if (!f) return;
    fseek(f, offset, SEEK_SET);
    fwrite(bytes, 1, 4, f);
    fclose(f);
}

static void test_checkpoint_hparam_mismatch(void) {
    RunFixture fx;
    fixture_create(&fx, 0.0f, 0);
    for (int s = 0; s < 2; s++) fixture_step(&fx);
    OVG_CHECK_EQ(tg_checkpoint_save_run(CKPT_TMP, &fx.opt, NULL, 0), 0);

    /* beta1 differs -> -1. */
    TgAdam other = tg_adam_create(fx.params, fx.n, 0.8f, 0.999f, 1e-8f);
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &other, TG_LOAD_RESUME, NULL), -1);
    /* Same file, INIT_FROM_WEIGHTS ignores the section -> 0. */
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &other, TG_LOAD_INIT_FROM_WEIGHTS, NULL), 0);
    tg_adam_free(&other);

    /* Optimizer section with rng_state == 0 (offset 12): -1 on RESUME, 0 on INIT. */
    uint32_t zero = 0;
    patch_file(CKPT_TMP, 12, &zero);
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &fx.opt, TG_LOAD_RESUME, NULL), -1);
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &fx.opt, TG_LOAD_INIT_FROM_WEIGHTS, NULL), 0);

    /* Unknown flag bit (offset 4) -> -1 from all three readers. */
    uint32_t bad_flags = 0x1u | 0x80u;
    patch_file(CKPT_TMP, 4, &bad_flags);
    TgCheckpointInfo info;
    OVG_CHECK_EQ(tg_checkpoint_info(CKPT_TMP, &info, NULL, 0), -1);
    OVG_CHECK_EQ(tg_checkpoint_load(CKPT_TMP, fx.params, fx.n), -1);
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &fx.opt, TG_LOAD_RESUME, NULL), -1);
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &fx.opt, TG_LOAD_INIT_FROM_WEIGHTS, NULL), -1);

    fixture_free(&fx);
    remove(CKPT_TMP);
}

static void test_checkpoint_exact_resume(void) {
    const uint32_t seed = 0xBEEF1234u;
    int prev_training = tg_training;
    tg_training = 1;

    /* Run A: 6 uninterrupted updates with dropout active. */
    RunFixture a;
    fixture_create(&a, 0.5f, 0);
    Snapshot init = snapshot_take(&a);   /* initial params, copied into run B */
    tg_rng_set_state(seed);
    for (int s = 0; s < 6; s++) fixture_step(&a);
    Snapshot final_a = snapshot_take(&a);
    fixture_free(&a);

    /* Run B: 3 updates, save, fresh model + TgAdam, load, 3 more. */
    RunFixture b;
    fixture_create(&b, 0.5f, 0);
    for (int i = 0; i < b.n; i++)
        memcpy(TG_DATAF(b.params[i]), init.data[i], (size_t)tg_numel(b.params[i]) * sizeof(float));
    tg_rng_set_state(seed);
    for (int s = 0; s < 3; s++) fixture_step(&b);
    OVG_CHECK_EQ(tg_checkpoint_save_run(CKPT_TMP, &b.opt, NULL, 0), 0);
    fixture_free(&b);

    RunFixture c;
    fixture_create(&c, 0.5f, 0);
    tg_rng_set_state(0x1);   /* anything; the load must overwrite it */
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &c.opt, TG_LOAD_RESUME, NULL), 0);
    OVG_CHECK_EQ(c.opt.step, 3);
    for (int s = 0; s < 3; s++) fixture_step(&c);
    OVG_CHECK_EQ(c.opt.step, 6);

    OVG_CHECK(snapshot_params_equal(&final_a, &c, 0.0f));

    snapshot_free(&init);
    snapshot_free(&final_a);
    fixture_free(&c);
    remove(CKPT_TMP);
    tg_training = prev_training;
}

static void test_checkpoint_tmp_replaced(void) {
    RunFixture fx;
    fixture_create(&fx, 0.0f, 0);
    remove(CKPT_TMP);
    remove(CKPT_TMP_TMP);

    OVG_CHECK_EQ(tg_checkpoint_save_run(CKPT_TMP, &fx.opt, NULL, 0), 0);
    OVG_CHECK(file_exists(CKPT_TMP));
    OVG_CHECK(!file_exists(CKPT_TMP_TMP));

    /* Overwriting an existing checkpoint also leaves no .tmp. */
    OVG_CHECK_EQ(tg_checkpoint_save(CKPT_TMP, fx.params, fx.n), 0);
    OVG_CHECK(file_exists(CKPT_TMP));
    OVG_CHECK(!file_exists(CKPT_TMP_TMP));

    /* Unopenable path: -1, no partial file. */
    const char *bad = "no_such_dir_ovg/model.bin";
    OVG_CHECK_EQ(tg_checkpoint_save_run(bad, &fx.opt, NULL, 0), -1);
    OVG_CHECK(!file_exists(bad));
    OVG_CHECK(!file_exists("no_such_dir_ovg/model.bin.tmp"));
    OVG_CHECK_EQ(tg_checkpoint_save(bad, fx.params, fx.n), -1);
    OVG_CHECK(!file_exists(bad));

    fixture_free(&fx);
    remove(CKPT_TMP);
}

#ifdef OVG_CUDA_ENABLED
static void test_checkpoint_v3_cuda_roundtrip(void) {
    RunFixture fx;
    fixture_create(&fx, 0.0f, 1);
    OVG_CHECK_EQ(fx.opt.on_cuda, 1);
    for (int s = 0; s < 3; s++) fixture_step(&fx);
    tg_rng_set_state(0xC0FFEE);
    Snapshot before = snapshot_take(&fx);
    OVG_CHECK_EQ(tg_checkpoint_save_run(CKPT_TMP, &fx.opt, "cfg!", 4), 0);

    /* A host fixture loading the same file sees what the device held. */
    RunFixture host;
    fixture_create(&host, 0.0f, 0);
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &host.opt, TG_LOAD_RESUME, NULL), 0);
    OVG_CHECK(snapshot_params_equal(&before, &host, 1e-6f));
    OVG_CHECK(!moments_all_zero(&host));

    corrupt_fixture(&fx);
    tg_rng_set_state(0x1234);
    TgCheckpointInfo info;
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP, &fx.opt, TG_LOAD_RESUME, &info), 0);
    OVG_CHECK(snapshot_params_equal(&before, &fx, 1e-6f));
    OVG_CHECK_EQ(fx.opt.step, 3);
    OVG_CHECK(tg_rng_get_state() == 0xC0FFEE);
    OVG_CHECK_EQ(info.version, 3);
    OVG_CHECK_EQ(info.has_optimizer, 1);

    /* Device moments survived the round trip: save again, compare on the host. */
    OVG_CHECK_EQ(tg_checkpoint_save_run(CKPT_TMP "2", &fx.opt, "cfg!", 4), 0);
    RunFixture host2;
    fixture_create(&host2, 0.0f, 0);
    OVG_CHECK_EQ(tg_checkpoint_load_run(CKPT_TMP "2", &host2.opt, TG_LOAD_RESUME, NULL), 0);
    for (int i = 0; i < host.n; i++) {
        int nel = tg_numel(host.params[i]);
        for (int j = 0; j < nel; j++) {
            OVG_CHECK_NEAR(host.opt.m[i][j], host2.opt.m[i][j], 1e-6f);
            OVG_CHECK_NEAR(host.opt.v[i][j], host2.opt.v[i][j], 1e-6f);
        }
    }

    snapshot_free(&before);
    fixture_free(&host);
    fixture_free(&host2);
    fixture_free(&fx);
    remove(CKPT_TMP);
    remove(CKPT_TMP "2");
}
#endif

void run_checkpoint_tests(int *passed, int *failed) {
    RUN_TEST(test_checkpoint_roundtrip,             passed, failed);
    RUN_TEST(test_checkpoint_bad_magic,             passed, failed);
    RUN_TEST(test_checkpoint_count_mismatch,        passed, failed);
#ifdef OVG_CUDA_ENABLED
    RUN_TEST(test_checkpoint_cuda_roundtrip,        passed, failed);
#endif
    RUN_TEST(test_checkpoint_v3_run_roundtrip,      passed, failed);
    RUN_TEST(test_checkpoint_info_before_model,     passed, failed);
    RUN_TEST(test_checkpoint_v2_loads_with_reset,   passed, failed);
    RUN_TEST(test_checkpoint_init_from_weights,     passed, failed);
    RUN_TEST(test_checkpoint_weights_only_then_run, passed, failed);
    RUN_TEST(test_checkpoint_hparam_mismatch,       passed, failed);
    RUN_TEST(test_checkpoint_exact_resume,          passed, failed);
    RUN_TEST(test_checkpoint_tmp_replaced,          passed, failed);
#ifdef OVG_CUDA_ENABLED
    RUN_TEST(test_checkpoint_v3_cuda_roundtrip,     passed, failed);
#endif
}
