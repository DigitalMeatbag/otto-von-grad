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
    Tensor *a = tg_new(2, 2);
    Tensor *b = tg_new(2, 2);
    a->persistent = b->persistent = 1;
    a->data[0] = 1; a->data[1] = 2; a->data[2] = 3; a->data[3] = 4;
    b->data[0] = 5; b->data[1] = 6; b->data[2] = 7; b->data[3] = 8;

    Tensor *out  = tg_add(a, b);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    OVG_CHECK_NEAR(out->data[0], 6.0f,  1e-5f);
    OVG_CHECK_NEAR(out->data[3], 12.0f, 1e-5f);
    /* d(sum)/d(a_i) = 1 for all i */
    OVG_CHECK_NEAR(a->grad[0], 1.0f, 1e-5f);
    OVG_CHECK_NEAR(b->grad[2], 1.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = b->persistent = 0;
    tg_free(a); tg_free(b);
}

static void test_sub(void) {
    Tensor *a = tg_new(1, 3);
    Tensor *b = tg_new(1, 3);
    a->persistent = b->persistent = 1;
    a->data[0] = 5; a->data[1] = 3; a->data[2] = 1;
    b->data[0] = 1; b->data[1] = 1; b->data[2] = 1;

    Tensor *out  = tg_sub(a, b);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    OVG_CHECK_NEAR(out->data[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(out->data[2], 0.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[0],  1.0f, 1e-5f);
    OVG_CHECK_NEAR(b->grad[0], -1.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = b->persistent = 0;
    tg_free(a); tg_free(b);
}

static void test_mul(void) {
    Tensor *a = tg_new(1, 2);
    Tensor *b = tg_new(1, 2);
    a->persistent = b->persistent = 1;
    a->data[0] = 2.0f; a->data[1] = 3.0f;
    b->data[0] = 4.0f; b->data[1] = 5.0f;

    Tensor *out  = tg_mul(a, b);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    OVG_CHECK_NEAR(out->data[0], 8.0f,  1e-5f);
    OVG_CHECK_NEAR(out->data[1], 15.0f, 1e-5f);
    /* d(a*b)/d(a) = b */
    OVG_CHECK_NEAR(a->grad[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 5.0f, 1e-5f);
    OVG_CHECK_NEAR(b->grad[0], 2.0f, 1e-5f);
    OVG_CHECK_NEAR(b->grad[1], 3.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = b->persistent = 0;
    tg_free(a); tg_free(b);
}

static void test_pow(void) {
    Tensor *a = tg_new(1, 2);
    a->persistent = 1;
    a->data[0] = 2.0f; a->data[1] = 3.0f;

    Tensor *out  = tg_pow(a, 2.0f);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    OVG_CHECK_NEAR(out->data[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(out->data[1], 9.0f, 1e-5f);
    /* d(a^2)/d(a) = 2*a */
    OVG_CHECK_NEAR(a->grad[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 6.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_matmul(void) {
    /* a: [2x3], b: [3x2], out: [2x2] */
    Tensor *a = tg_new(2, 3);
    Tensor *b = tg_new(3, 2);
    a->persistent = b->persistent = 1;

    float a_vals[] = {1, 2, 3, 4, 5, 6};
    float b_vals[] = {7, 8, 9, 10, 11, 12};
    for (int i = 0; i < 6; i++) { a->data[i] = a_vals[i]; b->data[i] = b_vals[i]; }

    Tensor *out = tg_matmul(a, b);
    OVG_CHECK_SHAPE(out, 2, 2);
    OVG_CHECK_NEAR(out->data[0], 58.0f,  1e-4f);
    OVG_CHECK_NEAR(out->data[1], 64.0f,  1e-4f);
    OVG_CHECK_NEAR(out->data[2], 139.0f, 1e-4f);
    OVG_CHECK_NEAR(out->data[3], 154.0f, 1e-4f);

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    /* d(loss)/d(a) = ones[2x2] @ b^T = [[15,19,23],[15,19,23]] */
    OVG_CHECK_NEAR(a->grad[0], 15.0f, 1e-4f);
    OVG_CHECK_NEAR(a->grad[1], 19.0f, 1e-4f);
    OVG_CHECK_NEAR(a->grad[2], 23.0f, 1e-4f);
    /* d(loss)/d(b) = a^T @ ones[2x2] = [[5,5],[7,7],[9,9]] */
    OVG_CHECK_NEAR(b->grad[0], 5.0f, 1e-4f);
    OVG_CHECK_NEAR(b->grad[2], 7.0f, 1e-4f);
    OVG_CHECK_NEAR(b->grad[4], 9.0f, 1e-4f);

    tg_free_graph(loss);
    a->persistent = b->persistent = 0;
    tg_free(a); tg_free(b);
}

/* ── Reductions ──────────────────────────────────────────────────────────── */

static void test_mean_rows(void) {
    /* 4x3 matrix where row i has value (i+1) everywhere */
    Tensor *A = tg_new(4, 3);
    A->persistent = 1;
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            A->data[i * 3 + j] = (float)(i + 1);

    Tensor *M    = tg_mean_rows(A);
    Tensor *loss = tg_sum(M);
    tg_backward(loss);

    OVG_CHECK_SHAPE(M, 1, 3);
    /* mean over rows: col j gets (1+2+3+4)/4 = 2.5 */
    OVG_CHECK_NEAR(M->data[0], 2.5f, 1e-5f);
    OVG_CHECK_NEAR(M->data[1], 2.5f, 1e-5f);
    OVG_CHECK_NEAR(M->data[2], 2.5f, 1e-5f);
    /* d(loss)/d(A_ij) = 1/n_rows = 1/4 = 0.25 */
    for (int i = 0; i < 12; i++)
        OVG_CHECK_NEAR(A->grad[i], 0.25f, 1e-5f);

    tg_free_graph(loss);
    A->persistent = 0;
    tg_free(A);
}

/* ── Activation + normalization ──────────────────────────────────────────── */

static void test_layer_norm_forward(void) {
    /* Single row [1, 2, 3, 4] — check output mean≈0, variance≈1 */
    Tensor *a = tg_new(1, 4);
    a->data[0] = 1; a->data[1] = 2; a->data[2] = 3; a->data[3] = 4;

    Tensor *out = tg_layer_norm_rows(a, 1e-5f);
    OVG_CHECK_SHAPE(out, 1, 4);

    float sum = 0.0f, sum2 = 0.0f;
    for (int j = 0; j < 4; j++) {
        sum  += out->data[j];
        sum2 += out->data[j] * out->data[j];
    }
    OVG_CHECK_NEAR(sum,        0.0f, 1e-4f);   /* mean ≈ 0 */
    OVG_CHECK_NEAR(sum2 / 4.0f, 1.0f, 1e-4f);  /* variance ≈ 1 */

    tg_free(out);
    tg_free(a);
}

static void test_softmax_rows_sum(void) {
    /* Any input — each output row must sum to 1.0 */
    Tensor *a = tg_new(3, 5);
    tg_fill_randn(a, 1.0f);

    Tensor *out = tg_softmax_rows(a);
    OVG_CHECK_SHAPE(out, 3, 5);

    for (int r = 0; r < 3; r++) {
        float row_sum = 0.0f;
        for (int j = 0; j < 5; j++) row_sum += out->data[r * 5 + j];
        OVG_CHECK_NEAR(row_sum, 1.0f, 1e-5f);
    }

    tg_free(out);
    tg_free(a);
}

static void test_gelu_forward_and_grad(void) {
    Tensor *a = tg_new(1, 3);
    a->persistent = 1;
    a->data[0] = -1.0f;
    a->data[1] =  0.0f;
    a->data[2] =  2.0f;

    Tensor *out = tg_gelu(a);
    OVG_CHECK_NEAR(out->data[0], -0.158808f, 1e-4f);
    OVG_CHECK_NEAR(out->data[1],  0.0f,      1e-5f);
    OVG_CHECK_NEAR(out->data[2],  1.954598f, 1e-4f);

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    float saved = a->data[2];
    float h = 1e-3f;
    a->data[2] = saved + h;
    Tensor *plus = tg_sum(tg_gelu(a));
    float y_plus = plus->data[0];
    tg_free_graph(plus);
    a->data[2] = saved - h;
    Tensor *minus = tg_sum(tg_gelu(a));
    float y_minus = minus->data[0];
    tg_free_graph(minus);
    a->data[2] = saved;

    float numeric = (y_plus - y_minus) / (2.0f * h);
    OVG_CHECK_NEAR(a->grad[2], numeric, 2e-3f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_layer_norm_rows_affine(void) {
    Tensor *a = tg_new(2, 3);
    Tensor *gamma = tg_new(1, 3);
    Tensor *beta = tg_new(1, 3);
    Tensor *w = tg_new(2, 3);
    a->persistent = gamma->persistent = beta->persistent = w->persistent = 1;

    float av[] = {1, 2, 4, 2, 3, 8};
    float gv[] = {1.0f, 0.5f, -1.0f};
    float bv[] = {0.1f, -0.2f, 0.3f};
    float wv[] = {0.2f, -0.7f, 1.1f, -0.3f, 0.4f, 0.9f};
    for (int i = 0; i < 6; i++) { a->data[i] = av[i]; w->data[i] = wv[i]; }
    for (int i = 0; i < 3; i++) { gamma->data[i] = gv[i]; beta->data[i] = bv[i]; }

    Tensor *out = tg_layer_norm_rows_affine(a, gamma, beta, 1e-5f);
    OVG_CHECK_SHAPE(out, 2, 3);
    OVG_CHECK_NEAR(out->data[0], -0.969041f, 1e-4f);
    OVG_CHECK_NEAR(out->data[1], -0.333630f, 1e-4f);
    OVG_CHECK_NEAR(out->data[2], -1.036306f, 1e-4f);

    Tensor *loss = tg_sum(tg_mul(out, w));
    tg_backward(loss);

    OVG_CHECK_NEAR(beta->grad[0], -0.1f, 1e-5f);
    OVG_CHECK_NEAR(beta->grad[1], -0.3f, 1e-5f);
    OVG_CHECK_NEAR(beta->grad[2],  2.0f, 1e-5f);
    OVG_CHECK(fabsf(gamma->grad[0]) > 0.01f);
    OVG_CHECK(fabsf(a->grad[0]) > 0.01f);

    tg_free_graph(loss);
    a->persistent = gamma->persistent = beta->persistent = w->persistent = 0;
    tg_free(a); tg_free(gamma); tg_free(beta); tg_free(w);
}

/* ── Loss ────────────────────────────────────────────────────────────────── */

static void test_cross_entropy_value(void) {
    /* logits = [[1, 0, 0]], target = [[1, 0, 0]]
       softmax(1,0,0)[0] = e/(e+2),  CE = log(e+2) - 1 ≈ 0.5514 */
    Tensor *logits  = tg_new(1, 3);
    Tensor *targets = tg_new(1, 3);
    logits->data[0] = 1.0f; logits->data[1] = 0.0f; logits->data[2] = 0.0f;
    targets->data[0] = 1.0f; targets->data[1] = 0.0f; targets->data[2] = 0.0f;

    Tensor *loss = tg_cross_entropy(logits, targets);
    OVG_CHECK_SHAPE(loss, 1, 1);

    float expected = logf(expf(1.0f) + 2.0f) - 1.0f;
    OVG_CHECK_NEAR(loss->data[0], expected, 1e-4f);

    tg_free(loss);
    tg_free(logits);
    tg_free(targets);
}

static Tensor *make_dense_targets(const int *ids, int rows, int cols, float smoothing) {
    Tensor *targets = tg_new(rows, cols);
    float off = cols > 1 ? smoothing / (float)(cols - 1) : 0.0f;
    for (int r = 0; r < rows; r++)
        for (int j = 0; j < cols; j++)
            targets->data[r * cols + j] = (j == ids[r]) ? (1.0f - smoothing) : off;
    return targets;
}

static void test_cross_entropy_no_sync_cpu(void) {
    Tensor *logits = tg_new(1, 3);
    Tensor *targets = tg_new(1, 3);
    logits->data[0] = 0.2f; logits->data[1] = -0.4f; logits->data[2] = 1.3f;
    targets->data[2] = 1.0f;

    Tensor *a = tg_cross_entropy(logits, targets);
    Tensor *b = tg_cross_entropy_no_sync(logits, targets);
    OVG_CHECK_NEAR(a->data[0], b->data[0], 1e-6f);

    tg_free(a); tg_free(b); tg_free(logits); tg_free(targets);
}

static void test_cross_entropy_sparse_matches_dense(void) {
    int ids[2] = {0, 2};
    Tensor *logits = tg_new(2, 3);
    float vals[] = {1.0f, 0.0f, -0.5f, -0.2f, 0.3f, 1.7f};
    for (int i = 0; i < 6; i++) logits->data[i] = vals[i];

    Tensor *hard_targets = make_dense_targets(ids, 2, 3, 0.0f);
    Tensor *smooth_targets = make_dense_targets(ids, 2, 3, 0.2f);
    Tensor *hard_dense = tg_cross_entropy(logits, hard_targets);
    Tensor *hard_sparse = tg_cross_entropy_sparse(logits, ids, 2, 0.0f);
    Tensor *smooth_dense = tg_cross_entropy(logits, smooth_targets);
    Tensor *smooth_sparse = tg_cross_entropy_sparse(logits, ids, 2, 0.2f);

    OVG_CHECK_NEAR(hard_dense->data[0], hard_sparse->data[0], 1e-6f);
    OVG_CHECK_NEAR(smooth_dense->data[0], smooth_sparse->data[0], 1e-6f);

    tg_free(hard_dense); tg_free(hard_sparse);
    tg_free(smooth_dense); tg_free(smooth_sparse);
    tg_free(hard_targets); tg_free(smooth_targets); tg_free(logits);
}

/* ── Slicing / embedding ─────────────────────────────────────────────────── */

static void test_embed_grad_accum(void) {
    /* weight[4 x 3], ids = {0, 2, 0}:
       token 0 used twice → grad row 0 = 2.0
       token 2 used once  → grad row 2 = 1.0
       rows 1,3 unused    → grad = 0.0 */
    int V = 4, C = 3, T = 3;
    int ids[3] = {0, 2, 0};

    Tensor *W = tg_new(V, C);
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

/* ── Error-path tests (setjmp/longjmp) ───────────────────────────────────── */

static void test_add_shape_mismatch(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(2, 3);
        Tensor *b = tg_new(3, 4);
        tg_add(a, b);  /* triggers ovg_fatal → longjmp; a and b intentionally leak */
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "shape") != NULL);
}

static void test_matmul_shape_mismatch(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(2, 3);
        Tensor *b = tg_new(4, 2);
        tg_matmul(a, b);  /* 3 != 4 → shape mismatch */
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "shape") != NULL);
}

static void test_embed_oob(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *W = tg_new(4, 3);
        int ids[2] = {0, 99};  /* 99 is out of range [0, 4) */
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
        Tensor *logits = tg_new(1, 3);
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
        Tensor *a = tg_new(2, 3);
        Tensor *gamma = tg_new(1, 2);
        Tensor *beta = tg_new(1, 3);
        tg_layer_norm_rows_affine(a, gamma, beta, 1e-5f);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "gamma") != NULL);
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

/* ── tg_concat_rows ──────────────────────────────────────────────────────── */

static void test_concat_rows(void) {
    /* a[2×3], b[1×3] → out[3×3] */
    Tensor *a = tg_new(2, 3);
    Tensor *b = tg_new(1, 3);
    a->persistent = b->persistent = 1;
    float av[] = {1,2,3,4,5,6};
    float bv[] = {7,8,9};
    for (int i = 0; i < 6; i++) a->data[i] = av[i];
    for (int i = 0; i < 3; i++) b->data[i] = bv[i];

    Tensor *parts[2] = {a, b};
    Tensor *out  = tg_concat_rows(parts, 2);
    OVG_CHECK_SHAPE(out, 3, 3);

    /* forward: row 0 = a row 0, row 1 = a row 1, row 2 = b row 0 */
    OVG_CHECK_NEAR(out->data[0], 1.0f, 1e-5f);
    OVG_CHECK_NEAR(out->data[4], 5.0f, 1e-5f);
    OVG_CHECK_NEAR(out->data[6], 7.0f, 1e-5f);
    OVG_CHECK_NEAR(out->data[8], 9.0f, 1e-5f);

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    /* d(sum)/d(a_ij) = 1, d(sum)/d(b_ij) = 1 */
    for (int i = 0; i < 6; i++) OVG_CHECK_NEAR(a->grad[i], 1.0f, 1e-5f);
    for (int i = 0; i < 3; i++) OVG_CHECK_NEAR(b->grad[i], 1.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = b->persistent = 0;
    tg_free(a); tg_free(b);
}

static void test_concat_rows_col_mismatch(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(2, 3);
        Tensor *b = tg_new(1, 4);  /* col mismatch */
        Tensor *parts[2] = {a, b};
        tg_concat_rows(parts, 2);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "mismatch") != NULL);
}

/* ── tg_row_slice ────────────────────────────────────────────────────────── */

static void test_row_slice(void) {
    /* a[4×2], slice rows [1,3) → out[2×2] = rows 1 and 2 of a */
    Tensor *a = tg_new(4, 2);
    a->persistent = 1;
    float av[] = {1,2, 3,4, 5,6, 7,8};
    for (int i = 0; i < 8; i++) a->data[i] = av[i];

    Tensor *out = tg_row_slice(a, 1, 3);
    OVG_CHECK_SHAPE(out, 2, 2);

    OVG_CHECK_NEAR(out->data[0], 3.0f, 1e-5f);
    OVG_CHECK_NEAR(out->data[1], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(out->data[2], 5.0f, 1e-5f);
    OVG_CHECK_NEAR(out->data[3], 6.0f, 1e-5f);

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    /* Only rows 1 and 2 receive gradient; rows 0 and 3 get zero. */
    OVG_CHECK_NEAR(a->grad[0], 0.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 0.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[2], 1.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[3], 1.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[4], 1.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[5], 1.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[6], 0.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[7], 0.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_row_slice_oob(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(3, 2);
        tg_row_slice(a, 0, 5);  /* row_end=5 > rows=3 */
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "invalid") != NULL);
}

/* ── tg_repeat_rows ──────────────────────────────────────────────────────── */

static void test_repeat_rows(void) {
    /* a[1×3] repeated 4 times → out[4×3] */
    Tensor *a = tg_new(1, 3);
    a->persistent = 1;
    a->data[0] = 1.0f; a->data[1] = 2.0f; a->data[2] = 3.0f;

    Tensor *out = tg_repeat_rows(a, 4);
    OVG_CHECK_SHAPE(out, 4, 3);

    /* every row of out must equal a */
    for (int i = 0; i < 4; i++) {
        OVG_CHECK_NEAR(out->data[i * 3 + 0], 1.0f, 1e-5f);
        OVG_CHECK_NEAR(out->data[i * 3 + 1], 2.0f, 1e-5f);
        OVG_CHECK_NEAR(out->data[i * 3 + 2], 3.0f, 1e-5f);
    }

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    /* each a->grad[j] accumulates 4 rows × grad=1 → 4.0 */
    OVG_CHECK_NEAR(a->grad[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[2], 4.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_repeat_rows_bad_shape(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(2, 3);  /* rows != 1 */
        tg_repeat_rows(a, 4);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "1 x C") != NULL);
}

/* ── tg_repeat_cols ──────────────────────────────────────────────────────── */

static void test_repeat_cols(void) {
    /* a[3×1] repeated 4 times → out[3×4] */
    Tensor *a = tg_new(3, 1);
    a->persistent = 1;
    a->data[0] = 5.0f; a->data[1] = 6.0f; a->data[2] = 7.0f;

    Tensor *out = tg_repeat_cols(a, 4);
    OVG_CHECK_SHAPE(out, 3, 4);

    /* every col of each row must equal a->data[row] */
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            OVG_CHECK_NEAR(out->data[i * 4 + j], a->data[i], 1e-5f);

    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    /* each a->grad[i] accumulates 4 cols × grad=1 → 4.0 */
    OVG_CHECK_NEAR(a->grad[0], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[1], 4.0f, 1e-5f);
    OVG_CHECK_NEAR(a->grad[2], 4.0f, 1e-5f);

    tg_free_graph(loss);
    a->persistent = 0;
    tg_free(a);
}

static void test_repeat_cols_bad_shape(void) {
    g_last_error[0] = '\0';
    ovg_set_fatal_handler(capture_handler);

    int triggered = 0;
    if (setjmp(g_test_escape) == 0) {
        Tensor *a = tg_new(3, 2);  /* cols != 1 */
        tg_repeat_cols(a, 4);
    } else {
        triggered = 1;
    }

    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);
    OVG_CHECK(strstr(g_last_error, "R x 1") != NULL);
}

static void test_drop_path_rate_schedule(void) {
    /* Verify linear ramp: block 0 gets 0.0, block 3 gets max_rate,
     * intermediates follow rate = max_rate * i / (n_blocks - 1). */
    int n = 4;
    float max_rate = 0.2f;
    TgTransformer t = tg_transformer_create_encoder(n, 8, 16, 4, 2, max_rate);

    OVG_CHECK_NEAR(t.blocks[0].drop_path_rate, 0.0f,   1e-5f);
    OVG_CHECK_NEAR(t.blocks[1].drop_path_rate, 0.2f / 3.0f, 1e-5f);
    OVG_CHECK_NEAR(t.blocks[2].drop_path_rate, 0.4f / 3.0f, 1e-5f);
    OVG_CHECK_NEAR(t.blocks[3].drop_path_rate, 0.2f,   1e-5f);

    tg_transformer_free(&t);
}

static void test_drop_path_inference_noop(void) {
    /* Same transformer, run twice in inference mode with different RNG seeds.
     * Output must be identical because tg_training=0 disables drop-path. */
    tg_training = 0;
    int seq = 4, dim = 8, hidden = 16, heads = 2;
    TgTransformer t = tg_transformer_create_encoder(2, dim, hidden, seq, heads, 0.5f);

    Tensor *X = tg_new(seq, dim);
    for (int i = 0; i < seq * dim; i++) X->data[i] = (float)(i % 7) * 0.1f;
    X->persistent = 1;

    tg_seed(42);
    Tensor *Y0 = tg_transformer_forward(&t, X);
    float saved[32];  /* seq*dim = 4*8 */
    memcpy(saved, Y0->data, (size_t)(seq * dim) * sizeof(float));
    tg_free_graph(Y0);

    tg_seed(99);
    Tensor *Y1 = tg_transformer_forward(&t, X);
    for (int i = 0; i < seq * dim; i++)
        OVG_CHECK_NEAR(saved[i], Y1->data[i], 1e-4f);

    tg_free_graph(Y1);
    tg_transformer_free(&t);
    X->persistent = 0;
    tg_free(X);
}

#ifdef OVG_CUDA_ENABLED
static void test_cuda_new_ops(void) {
    Tensor *a = tg_new(2, 3);
    Tensor *gamma = tg_new(1, 3);
    Tensor *beta = tg_new(1, 3);
    int ids[2] = {1, 2};
    float av[] = {-1.0f, 0.0f, 2.0f, 0.4f, -0.7f, 1.1f};
    for (int i = 0; i < 6; i++) a->data[i] = av[i];
    gamma->data[0] = 1.0f; gamma->data[1] = 0.5f; gamma->data[2] = -1.0f;
    beta->data[0] = 0.1f; beta->data[1] = -0.2f; beta->data[2] = 0.3f;
    a->persistent = gamma->persistent = beta->persistent = 1;
    tg_to_cuda(a); tg_to_cuda(gamma); tg_to_cuda(beta);

    Tensor *g = tg_gelu(a);
    Tensor *ln = tg_layer_norm_rows_affine(g, gamma, beta, 1e-5f);
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
    int T = 20;  /* > 16 forces multiple CUDA thread blocks (16×16 launch) */
    Tensor *scores = tg_new(T, T);
    tg_fill(scores, 0.5f);
    scores->persistent = 1;
    tg_to_cuda(scores);

    Tensor *masked = tg_causal_mask(scores);
    tg_from_cuda(masked);

    for (int i = 0; i < T; i++)
        for (int j = 0; j < T; j++) {
            float v = masked->data[i * T + j];
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
    RUN_TEST(test_layer_norm_affine_shape_mismatch, passed, failed);
    RUN_TEST(test_rng_uniform,               passed, failed);
    RUN_TEST(test_concat_rows,               passed, failed);
    RUN_TEST(test_concat_rows_col_mismatch,  passed, failed);
    RUN_TEST(test_row_slice,                 passed, failed);
    RUN_TEST(test_row_slice_oob,             passed, failed);
    RUN_TEST(test_repeat_rows,               passed, failed);
    RUN_TEST(test_repeat_rows_bad_shape,     passed, failed);
    RUN_TEST(test_repeat_cols,               passed, failed);
    RUN_TEST(test_repeat_cols_bad_shape,     passed, failed);
    RUN_TEST(test_drop_path_rate_schedule,   passed, failed);
    RUN_TEST(test_drop_path_inference_noop,  passed, failed);
#ifdef OVG_CUDA_ENABLED
    RUN_TEST(test_cuda_new_ops,        passed, failed);
    RUN_TEST(test_cuda_causal_mask_large, passed, failed);
#endif
}
