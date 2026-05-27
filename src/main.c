#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  include <direct.h>
static void ensure_dir(const char *path) { _mkdir(path); }
#else
#  include <sys/stat.h>
static void ensure_dir(const char *path) { mkdir(path, 0755); }
#endif

#include "tg_ops.h"
#include "tg_train.h"
#include "tg_gpt.h"
#include "tg_tokenizer.h"
#include "tg_sample.h"
#include "tg_checkpoint.h"
#include "tg_rng.h"

#define CHECKPOINT_PATH "data/checkpoints/model.bin"

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

static void print_char(char c, void *ud) { (void)ud; putchar(c); }

int main(void) {
    tg_seed_from_entropy();

    TgGPTConfig cfg = {
        .vocab_size = 0,   /* filled after vocab build */
        .embed_dim  = 16,
        .hidden_dim = 32,
        .seq_len    = 8,
        .n_blocks   = 2,
        .n_heads    = 1,
    };
    const int   steps     = 1200;
    const float lr        = 3e-4f;
    const float beta1     = 0.9f;
    const float beta2     = 0.999f;
    const float adam_eps  = 1e-8f;
    const float gen_temp  = 0.8f;
    const int   gen_topk  = 10;
    const int   gen_steps = 200;

    int   text_len;
    char *text      = tg_read_file("data/text/candide.txt", &text_len);
    TgVocab vocab   = tg_vocab_build(text, text_len);
    int *all_tokens = tg_tokenize(text, text_len, &vocab);

    cfg.vocab_size = vocab.size;
    const int T    = cfg.seq_len;

    if (text_len <= T + 1) {
        fprintf(stderr, "text too short for context length %d\n", T); exit(1);
    }

    /* Train/val split: last 10% of tokens held out for validation */
    int val_start    = text_len - text_len / 10;
    int val_len      = text_len - val_start;
    int max_train    = val_start - T - 1;
    int val_max_step = val_len - T - 1;
    if (max_train <= 0 || val_max_step <= 0) {
        fprintf(stderr, "corpus too small for train/val split\n"); exit(1);
    }

    TgGPT gpt = tg_gpt_create_from_config(&cfg);

    int max_params  = 3 + cfg.n_blocks * 12 + 4;
    Tensor **params = malloc((size_t)max_params * sizeof(Tensor *));
    if (!params) { fprintf(stderr, "out of memory\n"); exit(1); }
    int n_params = tg_gpt_collect_params(&gpt, params, max_params);

    printf("loaded data/text/candide.txt: %d chars\n", text_len);
    printf("vocab size: %d\n", vocab.size);
    printf("params: %d tensors\n", n_params);
    printf("train tokens: %d  val tokens: %d\n", val_start, val_len);
    printf("baseline ln(%d) ~= %.6f\n", vocab.size, logf((float)vocab.size));

    /* Resume from checkpoint if available */
    if (tg_checkpoint_load(CHECKPOINT_PATH, params, n_params) == 0)
        printf("[ovg] resumed from %s\n", CHECKPOINT_PATH);
    else
        printf("[ovg] no checkpoint found — training from scratch\n");

    /* Adam moment buffers — always zero-initialized; optimizer state is not
       checkpointed, so a resumed run restarts momentum from scratch. */
    float **m_buf = calloc((size_t)n_params, sizeof(float *));
    float **v_buf = calloc((size_t)n_params, sizeof(float *));
    if (!m_buf || !v_buf) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int i = 0; i < n_params; i++) {
        size_t nel = (size_t)params[i]->rows * params[i]->cols;
        m_buf[i]   = calloc(nel, sizeof(float));
        v_buf[i]   = calloc(nel, sizeof(float));
        if (!m_buf[i] || !v_buf[i]) { fprintf(stderr, "out of memory\n"); exit(1); }
    }

    int *inputs  = malloc((size_t)T * sizeof(int));
    int *targets = malloc((size_t)T * sizeof(int));
    if (!inputs || !targets) { fprintf(stderr, "out of memory\n"); exit(1); }

    tg_training = 1;
    for (int step = 1; step <= steps; step++) {
        int start = (step * 17) % max_train;
        for (int i = 0; i < T; i++) {
            inputs[i]  = all_tokens[start + i];
            targets[i] = all_tokens[start + i + 1];
        }

        Tensor *tgt_hot = make_one_hot(targets, T, vocab.size);
        Tensor *logits  = tg_gpt_forward(&gpt, inputs);
        Tensor *loss    = tg_cross_entropy(logits, tgt_hot);

        tg_backward(loss);
        tg_adam_step(params, m_buf, v_buf, n_params, lr, step, beta1, beta2, adam_eps);

        if (step == 1 || step % 200 == 0) {
            float train_loss = loss->data[0];

            /* Val loss: one window sampled from the held-out set */
            tg_training = 0;
            int vstart = val_start + (step % val_max_step);
            for (int i = 0; i < T; i++) {
                inputs[i]  = all_tokens[vstart + i];
                targets[i] = all_tokens[vstart + i + 1];
            }
            Tensor *vhot  = make_one_hot(targets, T, vocab.size);
            Tensor *vlog  = tg_gpt_forward(&gpt, inputs);
            Tensor *vloss = tg_cross_entropy(vlog, vhot);
            printf("step %4d/%d  train: %.6f  val: %.6f\n",
                   step, steps, train_loss, vloss->data[0]);
            tg_free_graph(vloss);
            tg_training = 1;
        }

        tg_free_graph(loss);
    }

    /* Final eval on the first T tokens */
    tg_training = 0;
    for (int i = 0; i < T; i++) {
        inputs[i]  = all_tokens[i];
        targets[i] = all_tokens[i + 1];
    }
    Tensor *eval_tgt_hot = make_one_hot(targets, T, vocab.size);
    Tensor *eval_logits  = tg_gpt_forward(&gpt, inputs);
    Tensor *eval_loss    = tg_cross_entropy(eval_logits, eval_tgt_hot);
    printf("final eval loss: %.6f\n", eval_loss->data[0]);

    printf("generated: ");
    for (int i = 0; i < T; i++) putchar(tg_vocab_decode(&vocab, inputs[i]));
    tg_generate(&gpt, &vocab, inputs, T, gen_steps, gen_temp, gen_topk, print_char, NULL);
    printf("\n");

    tg_free_graph(eval_loss);

    /* Save checkpoint */
    ensure_dir("data/checkpoints");
    if (tg_checkpoint_save(CHECKPOINT_PATH, params, n_params) == 0)
        printf("[ovg] checkpoint saved to %s\n", CHECKPOINT_PATH);

    for (int i = 0; i < n_params; i++) { free(m_buf[i]); free(v_buf[i]); }
    free(m_buf);
    free(v_buf);
    tg_gpt_free(&gpt);
    free(params);
    free(inputs);
    free(targets);
    free(all_tokens);
    free(text);

    return 0;
}
