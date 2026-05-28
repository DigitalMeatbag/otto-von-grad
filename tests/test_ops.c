#include "ovg_test.h"
#include "tg_ops.h"
#include "tg_rng.h"
#include "tg_train.h"
#include "tg_transformer.h"
#include "ovg_error.h"

#include <setjmp.h>
#include <string.h>
#include <math.h>

#ifdef OVG_CUDA_ENABLED
#include "tg_cuda.h"
#endif

/* ── Error-path capture ──────────────────────────────────────────────────── */

static jmp_buf g_test_escape;
static char    g_last_error[512];

static void capture_handler(const char *msg) {
    strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
    longjmp(g_test_escape, 1);
}

/* ── Arithmetic forward + backward ──────────────────────────────────────── */

static void test_add(void) {
    Tensor *a = tg_new(2, (int[]){2, 2});
    Tensor *b = tg_new(2, (int[]){2, 2});
    a->persistent = b->persistent = 1;
    float *ad = TG_DATAF(a), *bd = TG_DATAF(b);
    ad[0] = 1; ad[1] = 2; ad[2] = 3; ad[3] = 4;
    bd[0] = 5; bd[1] = 6; bd[2] = 7; bd[3] = 8;

    Tensor *out  = tg_add(a, b);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    OVG_CHECK_NEAR(TG_DATAF(out)[0], 6.0f,  1e-5f);
    OVG_CHECK_NEAR(TG_DATAF(out)[3], 12.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[0], 1.0f, 1e-5f);
    OVG_CHECK_NEAR(b->grad[2], 1.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = b->persistent = 0;
    tg_free(a); tg_free(b);
}

static void test_sub(void) {
    Tensor *a = tg_new(2, (int[]){1, 3});
    Tensor *b = tg_new(2, (int[]){1, 3});
    a->persistent = b->persistent = 1;
    float *ad = TG_DATAF(a), *bd = TG_DATAF(b);
    ad[0] = 5; ad[1] = 3; ad[2] = 1;
    bd[0] = 1; bd[1] = 1; bd[2] = 1;

    Tensor *out  = tg_sub(a, b);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    OVG_CHECK_NEAR(TG_DATAF(out)[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(TG_DATAF(out)[2], 0.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[0],  1.0f, 1e-5f);
    OVG_CHECK_NEAR(b->grad[0], -1.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = b->persistent = 0;
    tg_free(a); tg_free(b);
}

static void test_mul(void) {
    Tensor *a = tg_new(2, (int[]){1, 2});
    Tensor *b = tg_new(2, (int[]){1, 2});
    a->persistent = b->persistent = 1;
    float *ad = TG_DATAF(a), *bd = TG_DATAF(b);
    ad[0] = 2.0f; ad[1] = 3.0f;
    bd[0] = 4.0f; bd[1] = 5.0f;

    Tensor *out  = tg_mul(a, b);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    OVG_CHECK_NEAR(TG_DATAF(out)[0], 8.0f,  1e-5f);
    OVG_CHECK_NEAR(TG_DATAF(out)[1], 15.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 5.0f, 1e-5f);
    OVG_CHECK_NEAR(b->grad[0], 2.0f, 1e-5f);
    OVG_CHECK_NEAR(b->grad[1], 3.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = b->persistent = 0;
    tg_free(a); tg_free(b);
}

static void test_pow(void) {
    Tensor *a = tg_new(2, (int[]){1, 2});
    a->persistent = 1;
    float *ad = TG_DATAF(a);
    ad[0] = 2.0f; ad[1] = 3.0f;

    Tensor *out  = tg_pow(a, 2.0f);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    OVG_CHECK_NEAR(TG_DATAF(out)[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(TG_DATAF(out)[1], 9.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 6.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_matmul(void) {
    Tensor *a = tg_new(2, (int[]){2, 3});
    Tensor *b = tg_new(2, (int[]){3, 2});
    a->persistent = b->persistent = 1;

    float a_vals[] = {1, 2, 3, 4, 5, 6};
    float b_vals[] = {7, 8, 9, 10, 11, 12};
    float *ad = TG_DATAF(a), *bd = TG_DATAF(b);
    for (int i = 0; i < 6; i++) { ad[i] = a_vals[i]; bd[i] = b_vals[i]; }

    Tensor *out = tg_matmul(a, b);
    OVG_CHECK_SHAPE(out, 2, 2);
    float *od = TG_DATAF(out);
    OVG_CHECK_NEAR(od[0], 58.0f,  1e-4f);
    OVG_CHECK_NEAR(od[1], 64.0f,  1e-4f);
    OVG_CHECK_NEAR(od[2], 139.0f, 1e-4f);
    OVG_CHECK_NEAR(od[3], 154.0f, 1e-4f);

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    OVG_CHECK_NEAR(a->grad[0], 15.0f, 1e-4f);
    OVG_CHECK_NEAR(a->grad[1], 19.0f, 1e-4f);
    OVG_CHECK_NEAR(a->grad[2], 23.0f, 1e-4f);
    OVG_CHECK_NEAR(b->grad[0], 5.0f, 1e-4f);
    OVG_CHECK_NEAR(b->grad[2], 7.0f, 1e-4f);
    OVG_CHECK_NEAR(b->grad[4], 9.0f, 1e-4f);

    tg_free_graph(loss);
    a->persistent = b->persistent = 0;
    tg_free(a); tg_free(b);
}

/* ── Reductions ──────────────────────────────────────────────────────────── */

static void test_mean_rows(void) {
    Tensor *A = tg_new(2, (int[]){4, 3});
    A->persistent = 1;
    float *ad = TG_DATAF(A);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            ad[i * 3 + j] = (float)(i + 1);

    Tensor *M    = tg_mean_rows(A);
    Tensor *loss = tg_sum(M);
    tg_backward(loss);

    OVG_CHECK_SHAPE(M, 1, 3);
    float *md = TG_DATAF(M);
    OVG_CHECK_NEAR(md[0], 2.5f, 1e-5f);
    OVG_CHECK_NEAR(md[1], 2.5f, 1e-5f);
    OVG_CHECK_NEAR(md[2], 2.5f, 1e-5f);
    for (int i = 0; i < 12; i++)
        OVG_CHECK_NEAR(A->grad[i], 0.25f, 1e-5f);

    tg_free_graph(loss);
    A->persistent = 0;
    tg_free(A);
}

/* ── Activation + normalization ──────────────────────────────────────────── */

static void test_layer_norm_forward(void) {
    /* Non-affine case: gamma = ones, beta = zeros */
    Tensor *a     = tg_new(2, (int[]){1, 4});
    Tensor *gamma = tg_new(2, (int[]){1, 4});
    Tensor *beta  = tg_new(2, (int[]){1, 4});
    float *ad = TG_DATAF(a), *gd = TG_DATAF(gamma), *bd = TG_DATAF(beta);
    ad[0] = 1; ad[1] = 2; ad[2] = 3; ad[3] = 4;
    for (int j = 0; j < 4; j++) { gd[j] = 1.0f; bd[j] = 0.0f; }

    Tensor *out = tg_layer_norm(a, gamma, beta, 1e-5f);
    OVG_CHECK_SHAPE(out, 1, 4);

    float *od = TG_DATAF(out);
    float sum = 0.0f, sum2 = 0.0f;
    for (int j = 0; j < 4; j++) { sum += od[j]; sum2 += od[j] * od[j]; }
    OVG_CHECK_NEAR(sum,        0.0f, 1e-4f);
    OVG_CHECK_NEAR(sum2 / 4.0f, 1.0f, 1e-4f);

    tg_free(out);
    tg_free(a); tg_free(gamma); tg_free(beta);
}

static void test_softmax_rows_sum(void) {
    Tensor *a = tg_new(2, (int[]){3, 5});
    tg_fill_randn(a, 1.0f);

    Tensor *out = tg_softmax(a, 1);
    OVG_CHECK_SHAPE(out, 3, 5);

    float *od = TG_DATAF(out);
    for (int r = 0; r < 3; r++) {
        float row_sum = 0.0f;
        for (int j = 0; j < 5; j++) row_sum += od[r * 5 + j];
        OVG_CHECK_NEAR(row_sum, 1.0f, 1e-5f);
    }

    tg_free(out);
    tg_free(a);
}

static void test_gelu_forward_and_grad(void) {
    Tensor *a = tg_new(2, (int[]){1, 3});
    a->persistent = 1;
    float *ad = TG_DATAF(a);
    ad[0] = -1.0f; ad[1] = 0.0f; ad[2] = 2.0f;

    Tensor *out = tg_gelu(a);
    float *od = TG_DATAF(out);
    OVG_CHECK_NEAR(od[0], -0.158808f, 1e-4f);
    OVG_CHECK_NEAR(od[1],  0.0f,      1e-5f);
    OVG_CHECK_NEAR(od[2],  1.954598f, 1e-4f);

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    float saved = ad[2];
    float h = 1e-3f;
    ad[2] = saved + h;
    Tensor *plus = tg_sum(tg_gelu(a));
    float y_plus = TG_DATAF(plus)[0];
    tg_free_graph(plus);
    ad[2] = saved - h;
    Tensor *minus = tg_sum(tg_gelu(a));
    float y_minus = TG_DATAF(minus)[0];
    tg_free_graph(minus);
    ad[2] = saved;

    float numeric = (y_plus - y_minus) / (2.0f * h);
    OVG_CHECK_NEAR(a->grad[2], numeric, 2e-3f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_layer_norm_rows_affine(void) {
    Tensor *a     = tg_new(2, (int[]){2, 3});
    Tensor *gamma = tg_new(2, (int[]){1, 3});
    Tensor *beta  = tg_new(2, (int[]){1, 3});
    Tensor *w     = tg_new(2, (int[]){2, 3});
    a->persistent = gamma->persistent = beta->persistent = w->persistent = 1;

    float av[] = {1, 2, 4, 2, 3, 8};
    float gv[] = {1.0f, 0.5f, -1.0f};
    float bv[] = {0.1f, -0.2f, 0.3f};
    float wv[] = {0.2f, -0.7f, 1.1f, -0.3f, 0.4f, 0.9f};
    float *ad = TG_DATAF(a), *gd = TG_DATAF(gamma), *bd = TG_DATAF(beta), *wd = TG_DATAF(w);
    for (int i = 0; i < 6; i++) { ad[i] = av[i]; wd[i] = wv[i]; }
    for (int i = 0; i < 3; i++) { gd[i] = gv[i]; bd[i] = bv[i]; }

    Tensor *out = tg_layer_norm(a, gamma, beta, 1e-5f);
    OVG_CHECK_SHAPE(out, 2, 3);
    float *od = TG_DATAF(out);
    OVG_CHECK_NEAR(od[0], -0.969041f, 1e-4f);
    OVG_CHECK_NEAR(od[1], -0.333630f, 1e-4f);
    OVG_CHECK_NEAR(od[2], -1.036306f, 1e-4f);

    Tensor *loss = tg_sum(tg_mul(out, w));
    tg_backward(loss);

    OVG_CHECK_NEAR(beta->grad[0], -0.1f, 1e-5f);
    OVG_CHECK_NEAR(beta->grad[1], -0.3f, 1e-5f);
    OVG_CHECK_NEAR(beta->grad[2],  2.0f, 1e-5f);
    OVG_CHECK(fabsf(gamma->grad[0]) > 0.01f);
    OVG_CHECK(fabsf(ad[0]) > 0.01f);

    tg_free_graph(loss);
    a->persistent = gamma->persistent = beta->persistent = w->persistent = 0;
    tg_free(a); tg_free(gamma); tg_free(beta); tg_free(w);
}

/* ── Loss ────────────────────────────────────────────────────────────────── */

static void test_cross_entropy_value(void) {
    Tensor *logits  = tg_new(2, (int[]){1, 3});
    Tensor *targets = tg_new(2, (int[]){1, 3});
    float *ld = TG_DATAF(logits), *td = TG_DATAF(targets);
    ld[0] = 1.0f; ld[1] = 0.0f; ld[2] = 0.0f;
    td[0] = 1.0f; td[1] = 0.0f; td[2] = 0.0f;

    Tensor *loss = tg_cross_entropy(logits, targets);
    OVG_CHECK_SHAPE(loss, 1, 1);

    float expected = logf(expf(1.0f) + 2.0f) - 1.0f;
    OVG_CHECK_NEAR(TG_DATAF(loss)[0], expected, 1e-4f);

    tg_free(loss);
    tg_free(logits);
    tg_free(targets);
}

static Tensor *make_dense_targets(const int *ids, int rows, int cols, float smoothing) {
    Tensor *targets = tg_new(2, (int[]){rows, cols});
    float *td = TG_DATAF(targets);
    float off = cols > 1 ? smoothing / (float)(cols - 1) : 0.0f;
    for (int r = 0; r < rows; r++)
        for (int j = 0; j < cols; j++)
            td[r * cols + j] = (j == ids[r]) ? (1.0f - smoothing) : off;
    return targets;
}

static void test_cross_entropy_no_sync_cpu(void) {
    Tensor *logits  = tg_new(2, (int[]){1, 3});
    Tensor *targets = tg_new(2, (int[]){1, 3});
    float *ld = TG_DATAF(logits), *td = TG_DATAF(targets);
    ld[0] = 0.2f; ld[1] = -0.4f; ld[2] = 1.3f;
    td[2] = 1.0f;

    Tensor *a = tg_cross_entropy(logits, targets);
    Tensor *b = tg_cross_entropy_no_sync(logits, targets);
    OVG_CHECK_NEAR(TG_DATAF(a)[0], TG_DATAF(b)[0], 1e-6f);

    tg_free(a); tg_free(b); tg_free(logits); tg_free(targets);
}

static void test_cross_entropy_sparse_matches_dense(void) {
    int ids[2] = {0, 2};
    Tensor *logits = tg_new(2, (int[]){2, 3});
    float vals[] = {1.0f, 0.0f, -0.5f, -0.2f, 0.3f, 1.7f};
    float *ld = TG_DATAF(logits);
    for (int i = 0; i < 6; i++) ld[i] = vals[i];

    Tensor *hard_targets   = make_dense_targets(ids, 2, 3, 0.0f);
    Tensor *smooth_targets = make_dense_targets(ids, 2, 3, 0.2f);
    Tensor *hard_dense   = tg_cross_entropy(logits, hard_targets);
    Tensor *hard_sparse  = tg_cross_entropy_sparse(logits, ids, 2, 0.0f);
    Tensor *smooth_dense = tg_cross_entropy(logits, smooth_targets);
    Tensor *smooth_sparse= tg_cross_entropy_sparse(logits, ids, 2, 0.2f);

    OVG_CHECK_NEAR(TG_DATAF(hard_dense)[0],   TG_DATAF(hard_sparse)[0],   1e-6f);
    OVG_CHECK_NEAR(TG_DATAF(smooth_dense)[0], TG_DATAF(smooth_sparse)[0], 1e-6f);

    tg_free(hard_dense); tg_free(hard_sparse);
    tg_free(smooth_dense); tg_free(smooth_sparse);
    tg_free(hard_targets); tg_free(smooth_targets); tg_free(logits);
}

/* ── Slicing / embedding ─────────────────────────────────────────────────── */

static void test_embed_grad_accum(void) {
    int V = 4, C = 3, T = 3;
    int ids[3] = {0, 2, 0};

    Tensor *W = tg_new(2, (int[]){V, C});
    W->persistent = 1;
    tg_fill_randn(W, 0.5f);

    Tensor *out  = tg_embed(W, ids, T);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    for (int c = 0; c < C; c++) {
        OVG_CHECK_NEAR(W->grad[0 * C + c], 2.0f, 1e-5f);
        OVG_CHECK_NEAR(W->grad[1 * C + c], 0.0f, 1e-5f);
        OVG_CHECK_NEAR(W->grad[2 * C + c], 1.0f, 1e-5f);
        OVG_CHECK_NEAR(W->grad[3 * C + c], 0.0f, 1e-5f);
    }

    tg_free_graph(loss);
    W->persistent = 0;
    tg_free(W);
}

/* ── Error-path tests ────────────────────────────────────────────────────── */

static void test_add_shape_mismatch(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(2, (int[]){2, 3});
        Tensor *b = tg_new(2, (int[]){3, 4});
        tg_add(a, b);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "mismatch") != NULL);
}

static void test_matmul_shape_mismatch(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(2, (int[]){2, 3});
        Tensor *b = tg_new(2, (int[]){4, 2});
        tg_matmul(a, b);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "mismatch") != NULL);
}

static void test_embed_oob(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *W = tg_new(2, (int[]){4, 3});
        int ids[2] = {0, 99};
        tg_embed(W, ids, 2);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "range") != NULL);
}

static void test_sparse_ce_oob(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *logits = tg_new(2, (int[]){1, 3});
        int ids[1] = {4};
        tg_cross_entropy_sparse(logits, ids, 1, 0.0f);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "range") != NULL);
}

static void test_layer_norm_affine_shape_mismatch(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a     = tg_new(2, (int[]){2, 3});
        Tensor *gamma = tg_new(2, (int[]){1, 2}); /* numel=2 != C=3 */
        Tensor *beta  = tg_new(2, (int[]){1, 3});
        tg_layer_norm(a, gamma, beta, 1e-5f);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "last dim") != NULL);
}

static void test_layer_norm_3d_input(void) {
    /* [B,T,C] input with [1,C] gamma/beta — must succeed */
    Tensor *a     = tg_new(3, (int[]){2, 3, 4});
    Tensor *gamma = tg_new(2, (int[]){1, 4});
    Tensor *beta  = tg_new(2, (int[]){1, 4});
    tg_fill(gamma, 1.0f);
    tg_fill(beta,  0.0f);
    float *ad = TG_DATAF(a);
    for (int i = 0; i < 24; i++) ad[i] = (float)(i + 1);

    Tensor *out = tg_layer_norm(a, gamma, beta, 1e-5f);
    OVG_CHECK_SHAPE_ND(out, 3, 2, 3, 4);

    /* each group of 4 elements normalizes to mean~0, var~1 */
    float *od = TG_DATAF(out);
    float sum = 0.0f;
    for (int j = 0; j < 4; j++) sum += od[j];
    OVG_CHECK_NEAR(sum, 0.0f, 1e-4f);

    tg_free(out);
    tg_free(a);
    tg_free(gamma);
    tg_free(beta);
}

static void test_layer_norm_rejects_expanded_params(void) {
    /* [B,T,C] gamma/beta has last dim == C but numel > C — must fatal */
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a     = tg_new(3, (int[]){2, 3, 4});
        Tensor *gamma = tg_new(3, (int[]){2, 3, 4}); /* numel=24 != C=4 */
        Tensor *beta  = tg_new(2, (int[]){1, 4});
        tg_layer_norm(a, gamma, beta, 1e-5f);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "exactly") != NULL);
}

/* ── tg_rng_uniform ──────────────────────────────────────────────────────── */

static void test_rng_uniform(void) {
    tg_seed(42);
    int n = 1000;
    for (int i = 0; i < n; i++) {
        float v = tg_rng_uniform();
        OVG_CHECK(v >= 0.0f);
        OVG_CHECK(v <= 1.0f);
    }
}

static void test_drop_path_rate_schedule(void) {
    int n = 4;
    float max_rate = 0.2f;
    TgTransformer t = tg_transformer_create_encoder(n, 8, 16, 4, 2, max_rate);

    OVG_CHECK_NEAR(t.blocks[0].drop_path_rate, 0.0f,          1e-5f);
    OVG_CHECK_NEAR(t.blocks[1].drop_path_rate, 0.2f / 3.0f,   1e-5f);
    OVG_CHECK_NEAR(t.blocks[2].drop_path_rate, 0.4f / 3.0f,   1e-5f);
    OVG_CHECK_NEAR(t.blocks[3].drop_path_rate, 0.2f,           1e-5f);

    tg_transformer_free(&t);
}

static void test_drop_path_inference_noop(void) {
    tg_training = 0;
    int seq = 4, dim = 8, hidden = 16, heads = 2;
    TgTransformer t = tg_transformer_create_encoder(2, dim, hidden, seq, heads, 0.5f);

    Tensor *X = tg_new(2, (int[]){seq, dim});
    float *xd = TG_DATAF(X);
    for (int i = 0; i < seq * dim; i++) xd[i] = (float)(i % 7) * 0.1f;
    X->persistent = 1;

    tg_seed(42);
    Tensor *Y0 = tg_transformer_forward(&t, X);
    float saved[32];
    memcpy(saved, TG_DATAF(Y0), (size_t)(seq * dim) * sizeof(float));
    tg_free_graph(Y0);

    tg_seed(99);
    Tensor *Y1 = tg_transformer_forward(&t, X);
    float *y1d = TG_DATAF(Y1);
    for (int i = 0; i < seq * dim; i++)
        OVG_CHECK_NEAR(saved[i], y1d[i], 1e-4f);

    tg_free_graph(Y1);
    tg_transformer_free(&t);
    X->persistent = 0;
    tg_free(X);
}

#ifdef OVG_CUDA_ENABLED
static void test_cuda_cast_bf16_roundtrip(void) {
    /* F32 → BF16 → F32 should roundtrip within BF16 precision (7 mantissa bits) */
    Tensor *a = tg_new(2, (int[]){2, 4});
    float vals[] = {1.0f, -0.5f, 3.25f, 0.125f, -2.0f, 0.75f, 10.0f, 0.0f};
    for (int i = 0; i < 8; i++) TG_DATAF(a)[i] = vals[i];
    tg_to_cuda(a);

    Tensor *bf16 = tg_cast(a, TG_DTYPE_BF16);
    OVG_CHECK(bf16->dtype == TG_DTYPE_BF16);
    OVG_CHECK(bf16->on_cuda == 1);

    Tensor *back = tg_cast(bf16, TG_DTYPE_F32);
    OVG_CHECK(back->dtype == TG_DTYPE_F32);
    tg_from_cuda(back);

    /* BF16 has ~0.78% relative error; all test values are BF16-exact */
    float *bd = TG_DATAF(back);
    for (int i = 0; i < 8; i++)
        OVG_CHECK_NEAR(bd[i], vals[i], 0.02f);

    tg_cuda_free(a); tg_free(a);
    tg_cuda_free(bf16); tg_free(bf16);
    tg_cuda_free(back); tg_free(back);
}

static void test_cuda_bf16_matmul(void) {
    /* [2x4] BF16 @ [4x3] BF16 → [2x3] F32; compare with F32 matmul */
    Tensor *A = tg_new(2, (int[]){2, 4});
    Tensor *B = tg_new(2, (int[]){4, 3});
    float av[] = {1,2,3,4, -1,0,0.5f,2};
    float bv[] = {1,0,0, 0,1,0, 0,0,1, 1,1,1};
    for (int i = 0; i < 8; i++) TG_DATAF(A)[i] = av[i];
    for (int i = 0; i < 12; i++) TG_DATAF(B)[i] = bv[i];
    tg_to_cuda(A); tg_to_cuda(B);

    /* F32 reference */
    Tensor *ref = tg_matmul(A, B);
    tg_from_cuda(ref);

    /* BF16 path */
    Tensor *A_bf16 = tg_cast(A, TG_DTYPE_BF16);
    Tensor *B_bf16 = tg_cast(B, TG_DTYPE_BF16);
    Tensor *out = tg_matmul(A_bf16, B_bf16);
    OVG_CHECK(out->dtype == TG_DTYPE_F32);
    tg_from_cuda(out);

    float *od = TG_DATAF(out), *rd = TG_DATAF(ref);
    for (int i = 0; i < 6; i++)
        OVG_CHECK_NEAR(od[i], rd[i], 0.05f);  /* BF16 relative error tolerance */

    tg_cuda_free(A); tg_cuda_free(B);
    tg_free(A); tg_free(B);
    tg_cuda_free(ref); tg_free(ref);
    tg_cuda_free(A_bf16); tg_free(A_bf16);
    tg_cuda_free(B_bf16); tg_free(B_bf16);
    tg_cuda_free(out); tg_free(out);
}

static void test_cuda_new_ops(void) {
    Tensor *a     = tg_new(2, (int[]){2, 3});
    Tensor *gamma = tg_new(2, (int[]){1, 3});
    Tensor *beta  = tg_new(2, (int[]){1, 3});
    int ids[2] = {1, 2};
    float av[] = {-1.0f, 0.0f, 2.0f, 0.4f, -0.7f, 1.1f};
    float *ad = TG_DATAF(a), *gd = TG_DATAF(gamma), *bd = TG_DATAF(beta);
    for (int i = 0; i < 6; i++) ad[i] = av[i];
    gd[0] = 1.0f; gd[1] = 0.5f; gd[2] = -1.0f;
    bd[0] = 0.1f; bd[1] = -0.2f; bd[2] = 0.3f;
    a->persistent = gamma->persistent = beta->persistent = 1;
    tg_to_cuda(a); tg_to_cuda(gamma); tg_to_cuda(beta);

    Tensor *g  = tg_gelu(a);
    Tensor *ln = tg_layer_norm(g, gamma, beta, 1e-5f);
    Tensor *loss = tg_cross_entropy_sparse_no_sync(ln, ids, 2, 0.1f);
    float v = tg_scalar_value(loss);
    OVG_CHECK(v > 0.0f);
    tg_backward(loss);
    tg_from_cuda(a);
    OVG_CHECK(fabsf(a->grad[0]) > 1e-6f);

    tg_free_graph(loss);
    tg_cuda_free(a); tg_cuda_free(gamma); tg_cuda_free(beta);
    a->persistent = gamma->persistent = beta->persistent = 0;
    tg_free(a); tg_free(gamma); tg_free(beta);
}

static void test_cuda_causal_mask_large(void) {
    int T = 20;
    Tensor *scores = tg_new(2, (int[]){T, T});
    tg_fill(scores, 0.5f);
    scores->persistent = 1;
    tg_to_cuda(scores);

    Tensor *masked = tg_causal_mask(scores);
    tg_from_cuda(masked);

    float *md = TG_DATAF(masked);
    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++) {
            float v = md[i * T + j];
            if (j <= i)
                OVG_CHECK_NEAR(v, 0.5f, 1e-4f);
            else
                OVG_CHECK_NEAR(v, -1.0e9f, 1e4f);
        }

    Tensor *loss = tg_sum(masked);
    tg_backward(loss);
    tg_from_cuda(scores);

    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++) {
            float g = scores->grad[i * T + j];
            if (j <= i)
                OVG_CHECK_NEAR(g, 1.0f, 1e-4f);
            else
                OVG_CHECK_NEAR(g, 0.0f, 1e-4f);
        }

    tg_free_graph(loss);
    tg_cuda_free(scores);
    scores->persistent = 0;
    tg_free(scores);
}
#endif

/* ── tg_reshape ──────────────────────────────────────────────────────────── */

static void test_reshape_forward_and_grad(void) {
    /* [2x3] → [3x2]: same data, different shape */
    Tensor *a = tg_new(2, (int[]){2, 3});
    a->persistent = 1;
    float *ad = TG_DATAF(a);
    for (int i = 0; i < 6; i++) ad[i] = (float)(i + 1);

    Tensor *b = tg_reshape(a, 2, (int[]){3, 2});
    OVG_CHECK_SHAPE(b, 3, 2);

    /* Data is preserved in row-major order */
    float *bd = TG_DATAF(b);
    for (int i = 0; i < 6; i++) OVG_CHECK_NEAR(bd[i], (float)(i + 1), 1e-5f);

    Tensor *loss = tg_sum(b);
    tg_backward(loss);

    /* Gradient flows back: a->grad[i] += 1 for all i */
    for (int i = 0; i < 6; i++) OVG_CHECK_NEAR(a->grad[i], 1.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_reshape_element_count_mismatch(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(2, (int[]){2, 3});
        tg_reshape(a, 2, (int[]){2, 4});  /* 6 != 8 */
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "mismatch") != NULL);
}

static void test_reshape_to_3d(void) {
    /* [6x4] → [2x3x4]: 24 elements preserved */
    Tensor *a = tg_new(2, (int[]){6, 4});
    a->persistent = 1;
    for (int i = 0; i < 24; i++) TG_DATAF(a)[i] = (float)i;

    Tensor *b = tg_reshape(a, 3, (int[]){2, 3, 4});
    OVG_CHECK_SHAPE_ND(b, 3, 2, 3, 4);

    float *bd = TG_DATAF(b);
    for (int i = 0; i < 24; i++) OVG_CHECK_NEAR(bd[i], (float)i, 1e-5f);

    Tensor *loss = tg_sum(b);
    tg_backward(loss);
    for (int i = 0; i < 24; i++) OVG_CHECK_NEAR(a->grad[i], 1.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

/* ── tg_expand_dim ───────────────────────────────────────────────────────── */

static void test_expand_dim_forward_and_grad(void) {
    /* [1x3] → expand axis=0 by 4 → [4x3] */
    Tensor *a = tg_new(2, (int[]){1, 3});
    a->persistent = 1;
    float *ad = TG_DATAF(a);
    ad[0] = 1.0f; ad[1] = 2.0f; ad[2] = 3.0f;

    Tensor *out = tg_expand_dim(a, 0, 4);
    OVG_CHECK_SHAPE(out, 4, 3);

    float *od = TG_DATAF(out);
    for (int i = 0; i < 4; i++) {
        OVG_CHECK_NEAR(od[i*3+0], 1.0f, 1e-5f);
        OVG_CHECK_NEAR(od[i*3+1], 2.0f, 1e-5f);
        OVG_CHECK_NEAR(od[i*3+2], 3.0f, 1e-5f);
    }

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    /* Grad sums 4 copies back: a->grad[j] = 4 */
    OVG_CHECK_NEAR(a->grad[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[2], 4.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_expand_dim_bad_axis(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(2, (int[]){2, 3}); /* shape[0]=2, not 1 */
        tg_expand_dim(a, 0, 3);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "must be 1") != NULL);
}

/* ── tg_slice ────────────────────────────────────────────────────────────── */

static void test_slice_axis0(void) {
    /* [4x3] → slice axis=0, start=1, len=2 → [2x3] = rows 1 and 2 */
    Tensor *a = tg_new(2, (int[]){4, 3});
    a->persistent = 1;
    float *ad = TG_DATAF(a);
    for (int i = 0; i < 12; i++) ad[i] = (float)(i + 1);

    Tensor *out = tg_slice(a, 0, 1, 2);
    OVG_CHECK_SHAPE(out, 2, 3);

    float *od = TG_DATAF(out);
    OVG_CHECK_NEAR(od[0], 4.0f, 1e-5f); /* a[1][0] */
    OVG_CHECK_NEAR(od[1], 5.0f, 1e-5f);
    OVG_CHECK_NEAR(od[5], 9.0f, 1e-5f); /* a[2][2] */

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    /* Only rows 1 and 2 get gradient; rows 0 and 3 = 0 */
    for (int j = 0; j < 3; j++) OVG_CHECK_NEAR(a->grad[j],    0.0f, 1e-5f);
    for (int j = 0; j < 3; j++) OVG_CHECK_NEAR(a->grad[3+j],  1.0f, 1e-5f);
    for (int j = 0; j < 3; j++) OVG_CHECK_NEAR(a->grad[6+j],  1.0f, 1e-5f);
    for (int j = 0; j < 3; j++) OVG_CHECK_NEAR(a->grad[9+j],  0.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_slice_axis1(void) {
    /* [2x6] → slice axis=1, start=2, len=3 → [2x3] */
    Tensor *a = tg_new(2, (int[]){2, 6});
    a->persistent = 1;
    float *ad = TG_DATAF(a);
    for (int i = 0; i < 12; i++) ad[i] = (float)(i);

    Tensor *out = tg_slice(a, 1, 2, 3);
    OVG_CHECK_SHAPE(out, 2, 3);

    float *od = TG_DATAF(out);
    OVG_CHECK_NEAR(od[0], 2.0f, 1e-5f); /* a[0][2] */
    OVG_CHECK_NEAR(od[2], 4.0f, 1e-5f); /* a[0][4] */
    OVG_CHECK_NEAR(od[3], 8.0f, 1e-5f); /* a[1][2] */

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    /* Cols 2,3,4 get grad=1; cols 0,1,5 get grad=0 */
    for (int r = 0; r < 2; r++) {
        OVG_CHECK_NEAR(a->grad[r*6+0], 0.0f, 1e-5f);
        OVG_CHECK_NEAR(a->grad[r*6+1], 0.0f, 1e-5f);
        OVG_CHECK_NEAR(a->grad[r*6+2], 1.0f, 1e-5f);
        OVG_CHECK_NEAR(a->grad[r*6+3], 1.0f, 1e-5f);
        OVG_CHECK_NEAR(a->grad[r*6+4], 1.0f, 1e-5f);
        OVG_CHECK_NEAR(a->grad[r*6+5], 0.0f, 1e-5f);
    }

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_slice_oob(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(2, (int[]){3, 4});
        tg_slice(a, 0, 2, 2); /* 2+2=4 > 3 */
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "invalid") != NULL);
}

/* ── Suite entry point ───────────────────────────────────────────────────── */

void run_ops_tests(int *passed, int *failed) {
    RUN_TEST(test_add,                 passed, failed);
    RUN_TEST(test_sub,                 passed, failed);
    RUN_TEST(test_mul,                 passed, failed);
    RUN_TEST(test_pow,                 passed, failed);
    RUN_TEST(test_matmul,              passed, failed);
    RUN_TEST(test_mean_rows,           passed, failed);
    RUN_TEST(test_layer_norm_forward,  passed, failed);
    RUN_TEST(test_softmax_rows_sum,    passed, failed);
    RUN_TEST(test_gelu_forward_and_grad, passed, failed);
    RUN_TEST(test_layer_norm_rows_affine, passed, failed);
    RUN_TEST(test_cross_entropy_value, passed, failed);
    RUN_TEST(test_cross_entropy_no_sync_cpu, passed, failed);
    RUN_TEST(test_cross_entropy_sparse_matches_dense, passed, failed);
    RUN_TEST(test_embed_grad_accum,    passed, failed);
    RUN_TEST(test_add_shape_mismatch,  passed, failed);
    RUN_TEST(test_matmul_shape_mismatch, passed, failed);
    RUN_TEST(test_embed_oob,           passed, failed);
    RUN_TEST(test_sparse_ce_oob,       passed, failed);
    RUN_TEST(test_layer_norm_affine_shape_mismatch,    passed, failed);
    RUN_TEST(test_layer_norm_3d_input,                 passed, failed);
    RUN_TEST(test_layer_norm_rejects_expanded_params,  passed, failed);
    RUN_TEST(test_rng_uniform,               passed, failed);
    RUN_TEST(test_drop_path_rate_schedule,       passed, failed);
    RUN_TEST(test_drop_path_inference_noop,      passed, failed);
    RUN_TEST(test_reshape_forward_and_grad,       passed, failed);
    RUN_TEST(test_reshape_element_count_mismatch, passed, failed);
    RUN_TEST(test_reshape_to_3d,                  passed, failed);
    RUN_TEST(test_expand_dim_forward_and_grad,    passed, failed);
    RUN_TEST(test_expand_dim_bad_axis,            passed, failed);
    RUN_TEST(test_slice_axis0,                    passed, failed);
    RUN_TEST(test_slice_axis1,                    passed, failed);
    RUN_TEST(test_slice_oob,                      passed, failed);
#ifdef OVG_CUDA_ENABLED
    RUN_TEST(test_cuda_cast_bf16_roundtrip, passed, failed);
    RUN_TEST(test_cuda_bf16_matmul,         passed, failed);
    RUN_TEST(test_cuda_new_ops,        passed, failed);
    RUN_TEST(test_cuda_causal_mask_large, passed, failed);
#endif
}
