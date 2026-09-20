#include "ovg_test.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "tg_transformer.h"
#include "tg_patch_embed.h"
#include "tg_pool.h"
#include "ovg_error.h"

#include <math.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#ifdef OVG_CUDA_ENABLED
#include "tg_cuda.h"
#endif

/* ── Error-path capture ──────────────────────────────────────────────────── */

static jmp_buf g_test_escape;

static void capture_handler(const char *msg) {
    (void)msg;
    longjmp(g_test_escape, 1);
}

/* ── Fixture: n_patches = 4, patch_size = 6, C = 8; deterministic ramp ───── */

#define NP 4
#define PS 6
#define C  8

/* patches[b][t][k] = 0.1 * (b * NP * PS + t * PS + k) - 1; persistent. */
static Tensor *make_patches(int B) {
    Tensor *p = (B > 0) ? tg_new(3, (int[]){B, NP, PS}) : tg_new(2, (int[]){NP, PS});
    int n = tg_numel(p);
    for (int i = 0; i < n; i++) TG_DATAF(p)[i] = 0.1f * (float)i - 1.0f;
    p->persistent = 1;
    return p;
}

/* enc[b][t][c] = 100 b + 10 t + c over [2, 5, C]; persistent. */
static Tensor *make_enc(void) {
    Tensor *e = tg_new(3, (int[]){2, 5, C});
    for (int b = 0; b < 2; b++)
        for (int t = 0; t < 5; t++)
            for (int c = 0; c < C; c++)
                TG_DATAF(e)[(b * 5 + t) * C + c] = 100.0f * b + 10.0f * t + (float)c;
    e->persistent = 1;
    return e;
}

/* ── Patch embedding: shapes ─────────────────────────────────────────────── */

static void test_patch_embed_shapes(void) {
    TgPatchEmbed pe = tg_patch_embed_create(NP, PS, C, 1);
    OVG_CHECK_EQ(tg_patch_embed_n_tokens(&pe), NP + 1);

    Tensor *p2 = make_patches(0);
    Tensor *o2 = tg_patch_embed_forward(&pe, p2);
    OVG_CHECK_SHAPE_ND(o2, 3, 1, NP + 1, C);
    tg_free_graph(o2);
    tg_free(p2);

    Tensor *p3 = make_patches(2);
    Tensor *o3 = tg_patch_embed_forward(&pe, p3);
    OVG_CHECK_SHAPE_ND(o3, 3, 2, NP + 1, C);
    tg_free_graph(o3);
    tg_patch_embed_free(&pe);

    TgPatchEmbed pe0 = tg_patch_embed_create(NP, PS, C, 0);
    OVG_CHECK_EQ(tg_patch_embed_n_tokens(&pe0), NP);
    OVG_CHECK(pe0.Cls == NULL);
    Tensor *o0 = tg_patch_embed_forward(&pe0, p3);
    OVG_CHECK_SHAPE_ND(o0, 3, 2, NP, C);
    tg_free_graph(o0);
    tg_free(p3);
    tg_patch_embed_free(&pe0);
}

/* ── Patch embedding: values ─────────────────────────────────────────────── */

static void test_patch_embed_values(void) {
    TgPatchEmbed pe = tg_patch_embed_create(NP, PS, C, 1);
    tg_fill(pe.Cls, 0.5f);
    for (int t = 0; t < NP; t++)
        for (int c = 0; c < C; c++)
            TG_DATAF(pe.PosEmb)[t * C + c] = (float)t + 0.01f * (float)c;

    int B = 2;
    Tensor *p   = make_patches(B);
    Tensor *out = tg_patch_embed_forward(&pe, p);
    OVG_CHECK_SHAPE_ND(out, 3, B, NP + 1, C);

    for (int b = 0; b < B; b++) {
        for (int c = 0; c < C; c++)
            OVG_CHECK_NEAR(TG_DATAF(out)[(b * (NP + 1) + 0) * C + c], 0.5f, 1e-6f);
        for (int t = 0; t < NP; t++)
            for (int c = 0; c < C; c++) {
                float ref = TG_DATAF(pe.PosEmb)[t * C + c];
                for (int k = 0; k < PS; k++)
                    ref += TG_DATAF(p)[(b * NP + t) * PS + k] * TG_DATAF(pe.Proj)[k * C + c];
                OVG_CHECK_NEAR(TG_DATAF(out)[(b * (NP + 1) + t + 1) * C + c], ref, 1e-5f);
            }
    }

    tg_free_graph(out);
    tg_free(p);
    tg_patch_embed_free(&pe);
}

/* ── Patch embedding: 3D row b equals the 2D forward of patches[b] ───────── */

static void test_patch_embed_batch_parity(void) {
    TgPatchEmbed pe = tg_patch_embed_create(NP, PS, C, 1);
    int B = 2;
    Tensor *p   = make_patches(B);
    Tensor *out = tg_patch_embed_forward(&pe, p);

    for (int b = 0; b < B; b++) {
        Tensor *pb = tg_new(2, (int[]){NP, PS});
        memcpy(TG_DATAF(pb), TG_DATAF(p) + b * NP * PS, (size_t)NP * PS * sizeof(float));
        pb->persistent = 1;
        Tensor *ob = tg_patch_embed_forward(&pe, pb);
        OVG_CHECK_SHAPE_ND(ob, 3, 1, NP + 1, C);
        for (int i = 0; i < (NP + 1) * C; i++)
            OVG_CHECK_NEAR(TG_DATAF(out)[b * (NP + 1) * C + i], TG_DATAF(ob)[i], 1e-6f);
        tg_free_graph(ob);
        tg_free(pb);
    }

    tg_free_graph(out);
    tg_free(p);
    tg_patch_embed_free(&pe);
}

/* ── Patch embedding: backward of sum(out) ───────────────────────────────── */

static void test_patch_embed_backward(void) {
    TgPatchEmbed pe = tg_patch_embed_create(NP, PS, C, 1);
    int B = 2;
    Tensor *p    = make_patches(B);
    Tensor *out  = tg_patch_embed_forward(&pe, p);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    for (int c = 0; c < C; c++) OVG_CHECK_NEAR(pe.Cls->grad[c], (float)B, 1e-6f);
    for (int i = 0; i < NP * C; i++) OVG_CHECK_NEAR(pe.PosEmb->grad[i], (float)B, 1e-6f);

    /* dProj[k][c] = sum_{b,t} patches[b][t][k] */
    for (int k = 0; k < PS; k++) {
        float s = 0.0f;
        for (int bt = 0; bt < B * NP; bt++) s += TG_DATAF(p)[bt * PS + k];
        for (int c = 0; c < C; c++) OVG_CHECK_NEAR(pe.Proj->grad[k * C + c], s, 1e-5f);
    }

    /* dpatches[b][t][k] = sum_c Proj[k][c] */
    for (int k = 0; k < PS; k++) {
        float s = 0.0f;
        for (int c = 0; c < C; c++) s += TG_DATAF(pe.Proj)[k * C + c];
        for (int bt = 0; bt < B * NP; bt++) OVG_CHECK_NEAR(p->grad[bt * PS + k], s, 1e-5f);
    }

    /* free_graph leaves the parameters and the persistent input alive */
    tg_free_graph(loss);
    Tensor *again = tg_patch_embed_forward(&pe, p);
    OVG_CHECK_SHAPE_ND(again, 3, B, NP + 1, C);
    tg_free_graph(again);

    tg_free(p);
    tg_patch_embed_free(&pe);
}

/* ── Patch embedding: collect_params ─────────────────────────────────────── */

static void test_patch_embed_collect_params(void) {
    Tensor *params[3];

    TgPatchEmbed pe1 = tg_patch_embed_create(NP, PS, C, 1);
    OVG_CHECK_EQ(tg_patch_embed_collect_params(&pe1, params, 3), 3);
    OVG_CHECK(params[0] == pe1.Cls);
    OVG_CHECK(params[1] == pe1.Proj);
    OVG_CHECK(params[2] == pe1.PosEmb);
    for (int i = 0; i < 3; i++) OVG_CHECK(params[i]->persistent == 1);

    TgPatchEmbed pe0 = tg_patch_embed_create(NP, PS, C, 0);
    OVG_CHECK_EQ(tg_patch_embed_collect_params(&pe0, params, 3), 2);
    OVG_CHECK(params[0] == pe0.Proj);
    OVG_CHECK(params[1] == pe0.PosEmb);
    for (int i = 0; i < 2; i++) OVG_CHECK(params[i]->persistent == 1);

    /* max_params too small with use_cls = 1 → fatal */
    int triggered = 0;
    ovg_set_fatal_handler(capture_handler);
    if (setjmp(g_test_escape) == 0) {
        tg_patch_embed_collect_params(&pe1, params, 2);
    } else {
        triggered = 1;
    }
    ovg_set_fatal_handler(NULL);
    OVG_CHECK(triggered);

    tg_patch_embed_free(&pe1);
    tg_patch_embed_free(&pe0);
}

/* ── Patch embedding: bad input shapes ───────────────────────────────────── */

static void test_patch_embed_bad_shape_fatal(void) {
    TgPatchEmbed pe = tg_patch_embed_create(NP, PS, C, 1);
    int triggered = 0;
    ovg_set_fatal_handler(capture_handler);
    if (setjmp(g_test_escape) == 0) {
        tg_patch_embed_forward(&pe, tg_new(2, (int[]){NP, PS + 1}));   /* wrong patch_size */
    } else {
        triggered = 1;
    }
    OVG_CHECK(triggered);

    triggered = 0;
    ovg_set_fatal_handler(capture_handler);   /* re-arm: longjmp skipped the reset */
    if (setjmp(g_test_escape) == 0) {
        tg_patch_embed_forward(&pe, tg_new(2, (int[]){NP - 1, PS}));   /* wrong n_patches */
    } else {
        triggered = 1;
    }
    OVG_CHECK(triggered);

    triggered = 0;
    ovg_set_fatal_handler(capture_handler);   /* re-arm: longjmp skipped the reset */
    if (setjmp(g_test_escape) == 0) {
        tg_patch_embed_forward(&pe, tg_new(4, (int[]){1, 1, NP, PS})); /* 4D */
    } else {
        triggered = 1;
    }
    OVG_CHECK(triggered);

    ovg_set_fatal_handler(NULL);
    tg_patch_embed_free(&pe);
}

/* ── Pooling: CLS ────────────────────────────────────────────────────────── */

static void test_pool_cls(void) {
    Tensor *enc = make_enc();
    Tensor *out = tg_pool_cls(enc);
    OVG_CHECK_SHAPE_ND(out, 2, 2, C);
    for (int b = 0; b < 2; b++)
        for (int c = 0; c < C; c++)
            OVG_CHECK_NEAR(TG_DATAF(out)[b * C + c], 100.0f * b + (float)c, 1e-6f);

    Tensor *loss = tg_sum(out);
    tg_backward(loss);
    for (int b = 0; b < 2; b++)
        for (int t = 0; t < 5; t++)
            for (int c = 0; c < C; c++)
                OVG_CHECK_NEAR(enc->grad[(b * 5 + t) * C + c], t == 0 ? 1.0f : 0.0f, 1e-7f);

    tg_free_graph(loss);
    tg_free(enc);
}

/* ── Pooling: mean over tokens ───────────────────────────────────────────── */

static void test_pool_mean_tokens(void) {
    Tensor *enc = make_enc();
    Tensor *out = tg_pool_mean_tokens(enc);
    OVG_CHECK_SHAPE_ND(out, 2, 2, C);
    for (int b = 0; b < 2; b++)
        for (int c = 0; c < C; c++)
            OVG_CHECK_NEAR(TG_DATAF(out)[b * C + c], 100.0f * b + 20.0f + (float)c, 1e-5f);

    Tensor *loss = tg_sum(out);
    tg_backward(loss);
    for (int i = 0; i < 2 * 5 * C; i++) OVG_CHECK_NEAR(enc->grad[i], 1.0f / 5.0f, 1e-7f);

    tg_free_graph(loss);
    tg_free(enc);
}

/* ── Pooling: B = 1 mean equals tg_mean_rows on the [T, C] view, bitwise ── */

static void test_pool_mean_matches_mean_rows(void) {
    int T = 5;
    Tensor *enc = tg_new(3, (int[]){1, T, C});
    tg_fill_randn(enc, 1.0f);
    enc->persistent = 1;

    Tensor *pooled = tg_pool_mean_tokens(enc);
    Tensor *flat   = tg_reshape(enc, 2, (int[]){T, C});
    Tensor *ref    = tg_mean_rows(flat);
    OVG_CHECK_SHAPE_ND(pooled, 2, 1, C);
    OVG_CHECK_SHAPE_ND(ref, 2, 1, C);
    for (int c = 0; c < C; c++) OVG_CHECK(TG_DATAF(pooled)[c] == TG_DATAF(ref)[c]);

    tg_free_graph(pooled);
    tg_free_graph(ref);
    tg_free(enc);
}

/* ── Pooling: 2D input is fatal ──────────────────────────────────────────── */

static void test_pool_bad_ndim_fatal(void) {
    Tensor *enc2 = tg_new(2, (int[]){5, C});
    int triggered = 0;
    ovg_set_fatal_handler(capture_handler);
    if (setjmp(g_test_escape) == 0) {
        tg_pool_cls(enc2);
    } else {
        triggered = 1;
    }
    OVG_CHECK(triggered);

    triggered = 0;
    ovg_set_fatal_handler(capture_handler);   /* re-arm: longjmp skipped the reset */
    if (setjmp(g_test_escape) == 0) {
        tg_pool_mean_tokens(enc2);
    } else {
        triggered = 1;
    }
    OVG_CHECK(triggered);

    ovg_set_fatal_handler(NULL);
    tg_free(enc2);
}

/* ── The ViT recipe: patch embed → encoder → pool → linear ───────────────── */

#define N_LABELS 3

typedef struct {
    TgPatchEmbed  pe;
    TgTransformer enc;
    Tensor       *Wout;
} Vit;

static Vit vit_create(void) {
    Vit v;
    v.pe   = tg_patch_embed_create(NP, PS, C, 1);
    v.enc  = tg_transformer_create_encoder(1, C, 16, NP + 1, 2, 0.0f);
    v.Wout = tg_new(2, (int[]){C, N_LABELS});
    tg_fill_xavier_uniform(v.Wout);
    v.Wout->persistent = 1;
    return v;
}

static void vit_free(Vit *v) {
    tg_patch_embed_free(&v->pe);
    tg_transformer_free(&v->enc);
    tg_free(v->Wout);
}

static Tensor *vit_forward(Vit *v, Tensor *patches) {
    Tensor *X      = tg_patch_embed_forward(&v->pe, patches);   /* [B, NP+1, C] */
    Tensor *H      = tg_transformer_forward(&v->enc, X);        /* [B, NP+1, C] */
    Tensor *pooled = tg_pool_cls(H);                            /* [B, C]       */
    return tg_matmul(pooled, v->Wout);                          /* [B, N_LABELS] */
}

static int vit_collect_params(Vit *v, Tensor **params) {
    int n = tg_patch_embed_collect_params(&v->pe, params, 3);
    for (int i = 0; i < v->enc.n_blocks; i++) {
        TgBlock *b = &v->enc.blocks[i];
        params[n++] = b->gamma1;  params[n++] = b->beta1;
        params[n++] = b->attn.Wq; params[n++] = b->attn.Wk;
        params[n++] = b->attn.Wv; params[n++] = b->attn.Wo;
        params[n++] = b->gamma2;  params[n++] = b->beta2;
        params[n++] = b->W1;      params[n++] = b->B1;
        params[n++] = b->W2;      params[n++] = b->B2;
    }
    params[n++] = v->Wout;
    return n;
}

static void test_vit_recipe_end_to_end(void) {
    Vit v = vit_create();
    int B = 2;
    Tensor *p = make_patches(B);

    /* Batch-2 logits match the batch-1 run on each sample (eval mode) */
    int prev = tg_eval_begin();
    Tensor *logits = vit_forward(&v, p);
    OVG_CHECK_SHAPE_ND(logits, 2, B, N_LABELS);
    for (int b = 0; b < B; b++) {
        Tensor *pb = tg_new(2, (int[]){NP, PS});
        memcpy(TG_DATAF(pb), TG_DATAF(p) + b * NP * PS, (size_t)NP * PS * sizeof(float));
        pb->persistent = 1;
        Tensor *lb = vit_forward(&v, pb);
        OVG_CHECK_SHAPE_ND(lb, 2, 1, N_LABELS);
        for (int j = 0; j < N_LABELS; j++)
            OVG_CHECK_NEAR(TG_DATAF(logits)[b * N_LABELS + j], TG_DATAF(lb)[j], 1e-5f);
        tg_free_graph(lb);
        tg_free(pb);
    }
    tg_eval_end(prev);

    /* Backward through the whole recipe: every parameter gets a finite grad */
    int ids[2] = {0, 2};
    Tensor *loss = tg_cross_entropy_sparse(logits, ids, 2, 0.0f);
    tg_backward(loss);

    Tensor *params[16];
    int n = vit_collect_params(&v, params);
    OVG_CHECK_EQ(n, 16);
    for (int i = 0; i < n; i++) {
        int nel = tg_numel(params[i]);
        for (int j = 0; j < nel; j++) OVG_CHECK(isfinite(params[i]->grad[j]));
    }

    tg_free_graph(loss);
    Tensor *again = vit_forward(&v, p);
    OVG_CHECK_SHAPE_ND(again, 2, B, N_LABELS);
    tg_free_graph(again);

    tg_free(p);
    vit_free(&v);
}

/* ── CUDA parity ─────────────────────────────────────────────────────────── */

#ifdef OVG_CUDA_ENABLED
static void test_patch_embed_cuda_parity(void) {
    TgPatchEmbed pe = tg_patch_embed_create(NP, PS, C, 1);
    int B = 2;
    Tensor *p = make_patches(B);

    /* Host run */
    Tensor *out_h  = tg_patch_embed_forward(&pe, p);
    Tensor *loss_h = tg_sum(out_h);
    tg_backward(loss_h);
    int n_out = tg_numel(out_h);
    float *ref_out = malloc((size_t)n_out * sizeof(float));
    float *ref_cls = malloc((size_t)C * sizeof(float));
    float *ref_prj = malloc((size_t)PS * C * sizeof(float));
    float *ref_pos = malloc((size_t)NP * C * sizeof(float));
    OVG_CHECK(ref_out && ref_cls && ref_prj && ref_pos);
    memcpy(ref_out, TG_DATAF(out_h), (size_t)n_out * sizeof(float));
    memcpy(ref_cls, pe.Cls->grad,    (size_t)C * sizeof(float));
    memcpy(ref_prj, pe.Proj->grad,   (size_t)PS * C * sizeof(float));
    memcpy(ref_pos, pe.PosEmb->grad, (size_t)NP * C * sizeof(float));
    tg_free_graph(loss_h);

    /* Device run */
    tg_to_cuda(pe.Cls); tg_to_cuda(pe.Proj); tg_to_cuda(pe.PosEmb); tg_to_cuda(p);
    Tensor *out_d  = tg_patch_embed_forward(&pe, p);
    Tensor *loss_d = tg_sum(out_d);
    tg_backward(loss_d);
    tg_from_cuda(out_d); tg_from_cuda(pe.Cls); tg_from_cuda(pe.Proj); tg_from_cuda(pe.PosEmb);

    for (int i = 0; i < n_out; i++)  OVG_CHECK_NEAR(TG_DATAF(out_d)[i], ref_out[i], 1e-5f);
    for (int i = 0; i < C; i++)      OVG_CHECK_NEAR(pe.Cls->grad[i],    ref_cls[i], 1e-5f);
    for (int i = 0; i < PS * C; i++) OVG_CHECK_NEAR(pe.Proj->grad[i],   ref_prj[i], 1e-5f);
    for (int i = 0; i < NP * C; i++) OVG_CHECK_NEAR(pe.PosEmb->grad[i], ref_pos[i], 1e-5f);

    free(ref_out); free(ref_cls); free(ref_prj); free(ref_pos);
    tg_free_graph(loss_d);
    tg_cuda_free(p); tg_cuda_free(pe.Cls); tg_cuda_free(pe.Proj); tg_cuda_free(pe.PosEmb);
    tg_free(p);
    tg_patch_embed_free(&pe);
}

static void test_pool_cuda_parity(void) {
    int n = 2 * 5 * C;
    float ref_cls[2 * C], ref_mean[2 * C];
    float *ref_gcls = malloc((size_t)n * sizeof(float));
    float *ref_gmean = malloc((size_t)n * sizeof(float));
    OVG_CHECK(ref_gcls && ref_gmean);

    /* Host runs */
    Tensor *enc = make_enc();
    Tensor *oc = tg_pool_cls(enc);
    Tensor *lc = tg_sum(oc);
    tg_backward(lc);
    memcpy(ref_cls, TG_DATAF(oc), sizeof(ref_cls));
    memcpy(ref_gcls, enc->grad, (size_t)n * sizeof(float));
    tg_free_graph(lc);

    Tensor *om = tg_pool_mean_tokens(enc);
    Tensor *lm = tg_sum(om);
    tg_backward(lm);
    memcpy(ref_mean, TG_DATAF(om), sizeof(ref_mean));
    memcpy(ref_gmean, enc->grad, (size_t)n * sizeof(float));
    tg_free_graph(lm);

    /* Device runs */
    tg_to_cuda(enc);
    Tensor *oc_d = tg_pool_cls(enc);
    Tensor *lc_d = tg_sum(oc_d);
    tg_backward(lc_d);
    tg_from_cuda(oc_d); tg_from_cuda(enc);
    for (int i = 0; i < 2 * C; i++) OVG_CHECK_NEAR(TG_DATAF(oc_d)[i], ref_cls[i], 1e-6f);
    for (int i = 0; i < n; i++)     OVG_CHECK_NEAR(enc->grad[i], ref_gcls[i], 1e-6f);
    tg_free_graph(lc_d);

    Tensor *om_d = tg_pool_mean_tokens(enc);
    Tensor *lm_d = tg_sum(om_d);
    tg_backward(lm_d);
    tg_from_cuda(om_d); tg_from_cuda(enc);
    for (int i = 0; i < 2 * C; i++) OVG_CHECK_NEAR(TG_DATAF(om_d)[i], ref_mean[i], 1e-6f);
    for (int i = 0; i < n; i++)     OVG_CHECK_NEAR(enc->grad[i], ref_gmean[i], 1e-6f);
    tg_free_graph(lm_d);

    free(ref_gcls); free(ref_gmean);
    tg_cuda_free(enc);
    tg_free(enc);
}
#endif

/* ── Suite entry point ───────────────────────────────────────────────────── */

void run_vision_tests(int *passed, int *failed) {
    RUN_TEST(test_patch_embed_shapes,          passed, failed);
    RUN_TEST(test_patch_embed_values,          passed, failed);
    RUN_TEST(test_patch_embed_batch_parity,    passed, failed);
    RUN_TEST(test_patch_embed_backward,        passed, failed);
    RUN_TEST(test_patch_embed_collect_params,  passed, failed);
    RUN_TEST(test_patch_embed_bad_shape_fatal, passed, failed);
    RUN_TEST(test_pool_cls,                    passed, failed);
    RUN_TEST(test_pool_mean_tokens,            passed, failed);
    RUN_TEST(test_pool_mean_matches_mean_rows, passed, failed);
    RUN_TEST(test_pool_bad_ndim_fatal,         passed, failed);
    RUN_TEST(test_vit_recipe_end_to_end,       passed, failed);
#ifdef OVG_CUDA_ENABLED
    RUN_TEST(test_patch_embed_cuda_parity,     passed, failed);
    RUN_TEST(test_pool_cuda_parity,            passed, failed);
#endif
}
