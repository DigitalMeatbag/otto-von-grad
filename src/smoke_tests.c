#include "smoke_tests.h"
#include "tg_ops.h"
#include "tg_train.h"
#include "attention.h"

#include <math.h>
#include <stdio.h>

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
