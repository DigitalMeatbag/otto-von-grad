#include "tg_tensor.h"
#include "ovg_error.h"

#include <math.h>
#include <stdlib.h>

Tensor *tg_new(int rows, int cols) {
    Tensor *t = calloc(1, sizeof(Tensor));
    if (!t) ovg_fatal("tg_new: out of memory");
    t->rows = rows;
    t->cols = cols;
    t->data = calloc((size_t)rows * cols, sizeof(float));
    if (!t->data) { free(t); ovg_fatal("tg_new: out of memory"); }
    t->grad = calloc((size_t)rows * cols, sizeof(float));
    if (!t->grad) { free(t->data); free(t); ovg_fatal("tg_new: out of memory"); }
    return t;
}

void tg_free(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t->grad);
    free(t->cache);
    free(t);
}

void tg_fill(Tensor *t, float val) {
    int n = t->rows * t->cols;
    for (int i = 0; i < n; i++) t->data[i] = val;
}

void tg_fill_randn(Tensor *t, float scale) {
    int n = t->rows * t->cols;
    for (int i = 0; i < n - 1; i += 2) {
        float u1 = (rand() + 0.5f) / (RAND_MAX + 1.0f);
        float u2 = (rand() + 0.5f) / (RAND_MAX + 1.0f);
        float r   = sqrtf(-2.0f * logf(u1));
        float th  = 2.0f * 3.14159265f * u2;
        t->data[i]   = scale * r * cosf(th);
        t->data[i+1] = scale * r * sinf(th);
    }
    if (n % 2 == 1) {
        float u1 = (rand() + 0.5f) / (RAND_MAX + 1.0f);
        float u2 = (rand() + 0.5f) / (RAND_MAX + 1.0f);
        t->data[n-1] = scale * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    }
}

static void print_matrix(const float *buf, int rows, int cols, const char *name) {
    printf("%s [%dx%d]:\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) printf("%8.4f ", buf[i*cols+j]);
        printf("\n");
    }
}

void tg_print(const Tensor *t, const char *name)      { print_matrix(t->data, t->rows, t->cols, name); }
void tg_print_grad(const Tensor *t, const char *name)  { print_matrix(t->grad, t->rows, t->cols, name); }
