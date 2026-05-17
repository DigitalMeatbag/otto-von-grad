#include "smoke_tests.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "tg_gpt.h"
#include "attention.h"

#ifdef OVG_CUDA_ENABLED
#include "tg_cuda.h"
#endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static void print_grad_check(Tensor *t, const char *name) {
    int n = t->rows * t->cols, nonzero = 0;
    float abs_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float g = t->grad[i];
        if (g != 0.0f) nonzero++;
        abs_sum += g < 0.0f ? -g : g;
    }
    printf("  %s.grad: nonzero %d/%d, abs_sum %.6f\n", name, nonzero, n, abs_sum);
}

void attention_smoke_test(int T, int C, int n_heads) {
    printf("--- attention smoke test: T=%d C=%d n_heads=%d ---\n", T, C, n_heads);

    Tensor *X = tg_new(T, C);
    tg_fill_randn(X, 0.5f);

    TgSelfAttention attn = tg_attention_create(C, n_heads);
    TgAttentionForward f = tg_attention_forward_full(&attn, X);

    printf("  Q shape:    [%dx%d]\n", f.Q->rows, f.Q->cols);
    printf("  O shape:    [%dx%d]\n", f.O->rows, f.O->cols);
    printf("  proj shape: [%dx%d]\n", f.proj->rows, f.proj->cols);

    Tensor *Y    = tg_add(X, f.proj);
    Tensor *loss = tg_sum(Y);
    tg_backward(loss);

    print_grad_check(attn.Wq, "Wq");
    print_grad_check(attn.Wk, "Wk");
    print_grad_check(attn.Wv, "Wv");
    print_grad_check(attn.Wo, "Wo");
    print_grad_check(X, "X");

    tg_free(loss);
    tg_free(Y);
    tg_attention_forward_free(&f);
    tg_attention_free(&attn);
    tg_free(X);
}

void encoder_smoke_test(int T, int C) {
    printf("--- encoder smoke test: T=%d C=%d n_heads=1 (weight inspection) ---\n", T, C);

    Tensor *X = tg_new(T, C);
    tg_fill_randn(X, 0.5f);

    TgSelfAttention attn = tg_attention_create_encoder(C, 1);

    Tensor *Q       = tg_matmul(X, attn.Wq);
    Tensor *K       = tg_matmul(X, attn.Wk);
    Tensor *V       = tg_matmul(X, attn.Wv);
    Tensor *Kt      = tg_transpose(K);
    Tensor *scores  = tg_matmul(Q, Kt);
    Tensor *scaled  = tg_scale(scores, 1.0f / sqrtf((float)C));
    Tensor *weights = tg_softmax_rows(scaled);
    Tensor *O       = tg_matmul(weights, V);
    Tensor *proj    = tg_matmul(O, attn.Wo);

    tg_print(weights, "encoder weights");

    printf("  row sums:\n");
    int upper_zeros = 0;
    for (int i = 0; i < T; i++) {
        float sum = 0.0f;
        for (int j = 0; j < T; j++) {
            float w = weights->data[i * T + j];
            sum += w;
            if (j > i && w < 1e-6f) upper_zeros++;
        }
        printf("    row %d: %.6f\n", i, sum);
    }
    printf("  upper-triangle near-zeros: %d (expect 0)\n", upper_zeros);

    Tensor *Y    = tg_add(X, proj);
    Tensor *loss = tg_sum(Y);
    tg_backward(loss);

    print_grad_check(attn.Wq, "Wq");
    print_grad_check(attn.Wk, "Wk");
    print_grad_check(attn.Wv, "Wv");
    print_grad_check(attn.Wo, "Wo");
    print_grad_check(X, "X");

    tg_free(loss);
    tg_free(Y);
    tg_free(proj);
    tg_free(O);
    tg_free(weights);
    tg_free(scaled);
    tg_free(scores);
    tg_free(Kt);
    tg_free(V);
    tg_free(K);
    tg_free(Q);
    tg_attention_free(&attn);
    tg_free(X);
}

void mean_rows_smoke_test(void) {
    printf("--- tg_mean_rows smoke test ---\n");

    Tensor *A = tg_new(4, 3);
    tg_fill(A, 1.0f);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
            A->data[i * 3 + j] = (float)(i + 1);
    Tensor *M    = tg_mean_rows(A);
    Tensor *loss = tg_sum(M);
    tg_backward(loss);

    tg_print(M, "mean_rows(A)");
    printf("  A.grad (expect all 0.2500):\n");
    for (int i = 0; i < 4; i++) {
        printf("   ");
        for (int j = 0; j < 3; j++) printf(" %.4f", A->grad[i * 3 + j]);
        printf("\n");
    }
    tg_free(loss);
    tg_free(M);
    tg_free(A);

    printf("\n");
}

void transpose_grad_accum_test(void) {
    printf("--- transpose grad accumulation test ---\n");

    Tensor *A = tg_new(3, 4);
    tg_fill(A, 1.0f);
    A->persistent = 1;

    Tensor *B    = tg_transpose(A);
    Tensor *C    = tg_transpose(A);
    Tensor *loss = tg_add(tg_sum(B), tg_sum(C));
    tg_backward(loss);

    int n = A->rows * A->cols, ok = 1;
    for (int i = 0; i < n; i++)
        if (fabsf(A->grad[i] - 2.0f) > 1e-5f) { ok = 0; break; }
    printf("  CPU: A.grad all 2.0 (accumulation, not overwrite): %s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);
    tg_free_graph(loss);

#ifdef OVG_CUDA_ENABLED
    tg_to_cuda(A);

    Tensor *Bg    = tg_transpose(A);
    Tensor *Cg    = tg_transpose(A);
    Tensor *lossg = tg_add(tg_sum(Bg), tg_sum(Cg));
    tg_backward(lossg);
    tg_from_cuda(A);

    ok = 1;
    for (int i = 0; i < n; i++)
        if (fabsf(A->grad[i] - 2.0f) > 1e-5f) { ok = 0; break; }
    printf("  CUDA: A.grad all 2.0 (transpose_bwd_k accumulation): %s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);
    tg_free_graph(lossg);
    tg_cuda_free(A);
#endif

    A->persistent = 0;
    tg_free(A);
    printf("\n");
}

void collect_params_capacity_test(void) {
    printf("--- collect_params capacity test ---\n");

    // 1 block → need exactly 3 + 1*8 = 11 slots
    TgGPT gpt = tg_gpt_create(16, 4, 8, 4, 1, 1);
    Tensor *params[11];
    int n = tg_gpt_collect_params(&gpt, params, 11);
    printf("  exact-fit capacity (11 slots, 1 block): %s\n", n == 11 ? "PASS" : "FAIL");
    if (n != 11) exit(1);
    tg_gpt_free(&gpt);
    printf("\n");
}

#ifdef OVG_CUDA_ENABLED
void cuda_causal_mask_large_test(void) {
    int T = 20; // > 16 forces multiple CUDA blocks (16x16 launch)
    printf("--- CUDA causal mask T=%d (multi-block) test ---\n", T);

    Tensor *scores = tg_new(T, T);
    tg_fill(scores, 0.5f);
    scores->persistent = 1;
    tg_to_cuda(scores);

    Tensor *masked = tg_causal_mask(scores);
    tg_from_cuda(masked);

    int ok_fwd = 1;
    for (int i = 0; i < T && ok_fwd; i++)
        for (int j = 0; j < T && ok_fwd; j++) {
            float v = masked->data[i * T + j];
            if (j <= i && fabsf(v - 0.5f) > 1e-4f) ok_fwd = 0;
            if (j >  i && fabsf(v + 1e9f) > 1e4f)  ok_fwd = 0;
        }
    printf("  forward (rows 0-%d all covered): %s\n", T - 1, ok_fwd ? "PASS" : "FAIL");
    if (!ok_fwd) exit(1);

    Tensor *loss = tg_sum(masked);
    tg_backward(loss);
    tg_from_cuda(scores);

    int ok_bwd = 1;
    for (int i = 0; i < T && ok_bwd; i++)
        for (int j = 0; j < T && ok_bwd; j++) {
            float g = scores->grad[i * T + j];
            if (j <= i && fabsf(g - 1.0f) > 1e-4f) ok_bwd = 0;
            if (j >  i && fabsf(g)         > 1e-4f) ok_bwd = 0;
        }
    printf("  backward (rows 0-%d all covered): %s\n", T - 1, ok_bwd ? "PASS" : "FAIL");
    if (!ok_bwd) exit(1);

    tg_free_graph(loss);
    tg_cuda_free(scores);
    scores->persistent = 0;
    tg_free(scores);
    printf("\n");
}

/* Mixed-placement failure (one CPU parent + one CUDA parent → exit(1) in make_op)
   cannot be tested inline without subprocess infrastructure on Windows.
   To verify manually: call tg_to_cuda() on one input to an op but not the other. */
#endif

void embed_smoke_test(void) {
    printf("--- tg_embed smoke test ---\n");

    /* weight[4 x 3], ids = {0, 2, 0}:
       token 0 appears twice → weight.grad row 0 should accumulate to 2.0,
       token 2 appears once  → weight.grad row 2 should be 1.0,
       rows 1 and 3 unused   → weight.grad should stay 0.0 */
    int V = 4, C = 3, T = 3;
    int ids[3] = {0, 2, 0};

    Tensor *W = tg_new(V, C);
    W->persistent = 1;
    tg_fill_randn(W, 0.5f);

    Tensor *out  = tg_embed(W, ids, T);
    Tensor *loss = tg_sum(out);
    tg_backward(loss);

    int ok = 1;
    for (int c = 0; c < C && ok; c++) {
        if (fabsf(W->grad[0 * C + c] - 2.0f) > 1e-5f) ok = 0;  // id 0 used twice
        if (fabsf(W->grad[1 * C + c] - 0.0f) > 1e-5f) ok = 0;  // id 1 unused
        if (fabsf(W->grad[2 * C + c] - 1.0f) > 1e-5f) ok = 0;  // id 2 used once
        if (fabsf(W->grad[3 * C + c] - 0.0f) > 1e-5f) ok = 0;  // id 3 unused
    }
    printf("  CPU: weight.grad accumulation (rows 0/2/unused): %s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);

    tg_free_graph(loss);

#ifdef OVG_CUDA_ENABLED
    tg_to_cuda(W);

    Tensor *outg  = tg_embed(W, ids, T);
    Tensor *lossg = tg_sum(outg);
    tg_backward(lossg);
    tg_from_cuda(W);

    ok = 1;
    for (int c = 0; c < C && ok; c++) {
        if (fabsf(W->grad[0 * C + c] - 2.0f) > 1e-5f) ok = 0;
        if (fabsf(W->grad[1 * C + c] - 0.0f) > 1e-5f) ok = 0;
        if (fabsf(W->grad[2 * C + c] - 1.0f) > 1e-5f) ok = 0;
        if (fabsf(W->grad[3 * C + c] - 0.0f) > 1e-5f) ok = 0;
    }
    printf("  CUDA: weight.grad accumulation (atomicAdd correctness): %s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);

    tg_free_graph(lossg);
    tg_cuda_free(W);
#endif

    W->persistent = 0;
    tg_free(W);
    printf("\n");
}
