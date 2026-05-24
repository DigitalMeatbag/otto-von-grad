#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "tg_ops.h"
#include "tg_train.h"
#include "tg_gpt.h"
#include "tg_rng.h"

#define MAX_VOCAB 256

typedef struct {
    char chars[MAX_VOCAB];
    int  ids[256];
    int  size;
} Vocab;

static char *read_file(const char *path, int *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Could not open file: %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    char *buf = malloc((size_t)size + 1);
    if (!buf) { fprintf(stderr, "read_file: out of memory\n"); exit(1); }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        fprintf(stderr, "read_file: read error\n"); exit(1);
    }
    buf[size] = '\0';
    fclose(f);
    *out_len = (int)size;
    return buf;
}

static Vocab vocab_build(const char *text, int len) {
    Vocab v;
    memset(v.ids, -1, sizeof(v.ids));
    v.size = 0;
    unsigned char seen[256] = {0};
    for (int i = 0; i < len; i++) seen[(unsigned char)text[i]] = 1;
    for (int c = 0; c < 256; c++) {
        if (seen[c]) { v.chars[v.size] = (char)c; v.ids[c] = v.size++; }
    }
    return v;
}

static int vocab_encode(const Vocab *v, char c) {
    int id = v->ids[(unsigned char)c];
    if (id < 0) { fprintf(stderr, "vocab_encode: unknown char\n"); exit(1); }
    return id;
}

static char vocab_decode(const Vocab *v, int id) {
    if (id < 0 || id >= v->size) { fprintf(stderr, "vocab_decode: bad id\n"); exit(1); }
    return v->chars[id];
}

static int *tokenize(const char *text, int len, const Vocab *v) {
    int *tokens = malloc((size_t)len * sizeof(int));
    if (!tokens) { fprintf(stderr, "tokenize: out of memory\n"); exit(1); }
    for (int i = 0; i < len; i++) tokens[i] = vocab_encode(v, text[i]);
    return tokens;
}

static Tensor *make_one_hot(const int *ids, int n, int n_classes) {
    Tensor *out = tg_new(n, n_classes);
    tg_fill(out, 0.0f);
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= n_classes) {
            fprintf(stderr, "make_one_hot: id %d out of range\n", ids[i]); exit(1);
        }
        out->data[i * n_classes + ids[i]] = 1.0f;
    }
    return out;
}

static int argmax_row(Tensor *t, int row) {
    int best = 0;
    float best_val = t->data[row * t->cols];
    for (int j = 1; j < t->cols; j++) {
        float v = t->data[row * t->cols + j];
        if (v > best_val) { best = j; best_val = v; }
    }
    return best;
}

static void generate(TgGPT *g, const Vocab *v, const int *seed, int steps) {
    int *ctx = malloc((size_t)g->seq_len * sizeof(int));
    if (!ctx) { fprintf(stderr, "generate: out of memory\n"); exit(1); }
    memcpy(ctx, seed, (size_t)g->seq_len * sizeof(int));

    printf("generated:\n");
    for (int i = 0; i < g->seq_len; i++) putchar(vocab_decode(v, ctx[i]));

    for (int s = 0; s < steps; s++) {
        Tensor *logits = tg_gpt_forward(g, ctx);
        int next = argmax_row(logits, g->seq_len - 1);
        putchar(vocab_decode(v, next));
        for (int i = 0; i < g->seq_len - 1; i++) ctx[i] = ctx[i + 1];
        ctx[g->seq_len - 1] = next;
        tg_free_graph(logits);
    }
    printf("\n");
    free(ctx);
}

int main(void) {
    tg_seed_from_entropy();

    int text_len;
    char *text        = read_file("data/text/candide.txt", &text_len);
    Vocab vocab       = vocab_build(text, text_len);
    int *all_tokens   = tokenize(text, text_len, &vocab);

    int T        = 8;
    int C        = 16;
    int H        = 32;
    int n_blocks = 2;
    int n_heads  = 1;
    int steps    = 1200;
    float lr     = 0.05f;

    if (text_len <= T + 1) {
        fprintf(stderr, "text too short for context length %d\n", T); exit(1);
    }

    TgGPT gpt = tg_gpt_create(vocab.size, C, H, T, n_blocks, n_heads);

    Tensor *params[64];
    int n_params = tg_gpt_collect_params(&gpt, params, 64);

    printf("loaded data/text/candide.txt: %d chars\n", text_len);
    printf("vocab size: %d\n", vocab.size);
    printf("params: %d tensors\n", n_params);
    printf("baseline ln(%d) ~= %.6f\n", vocab.size, logf((float)vocab.size));

    int *inputs  = malloc((size_t)T * sizeof(int));
    int *targets = malloc((size_t)T * sizeof(int));
    if (!inputs || !targets) { fprintf(stderr, "out of memory\n"); exit(1); }

    int max_start = text_len - T - 1;
    for (int step = 1; step <= steps; step++) {
        int start = (step * 17) % max_start;
        for (int i = 0; i < T; i++) {
            inputs[i]  = all_tokens[start + i];
            targets[i] = all_tokens[start + i + 1];
        }

        Tensor *tgt_hot = make_one_hot(targets, T, vocab.size);
        Tensor *logits  = tg_gpt_forward(&gpt, inputs);
        Tensor *loss    = tg_cross_entropy(logits, tgt_hot);

        tg_backward(loss);
        tg_sgd_step(params, n_params, lr);

        if (step == 1 || step % 200 == 0)
            printf("step %4d/%d  loss: %.6f\n", step, steps, loss->data[0]);

        tg_free_graph(loss);
    }

    int *eval_in = inputs, *eval_tgt = targets;
    for (int i = 0; i < T; i++) {
        eval_in[i]  = all_tokens[i];
        eval_tgt[i] = all_tokens[i + 1];
    }
    Tensor *eval_tgt_hot = make_one_hot(eval_tgt, T, vocab.size);
    Tensor *eval_logits  = tg_gpt_forward(&gpt, eval_in);
    Tensor *eval_loss    = tg_cross_entropy(eval_logits, eval_tgt_hot);
    printf("final eval loss: %.6f\n", eval_loss->data[0]);

    generate(&gpt, &vocab, eval_in, 200);

    tg_free_graph(eval_loss);
    tg_gpt_free(&gpt);
    free(inputs);
    free(targets);
    free(all_tokens);
    free(text);

    return 0;
}
