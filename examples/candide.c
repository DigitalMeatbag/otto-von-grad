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
#include "tg_optim.h"
#include "tg_gpt.h"
#include "tg_tokenizer.h"
#include "tg_sample.h"
#include "tg_checkpoint.h"
#include "tg_rng.h"

/* OVG_EXAMPLES_DIR is set by CMake to this directory, so the demo runs from any CWD. */
#ifndef OVG_EXAMPLES_DIR
#  define OVG_EXAMPLES_DIR "examples"
#endif
#define CORPUS_PATH      OVG_EXAMPLES_DIR "/data/candide.txt"
#define CHECKPOINT_DIR   OVG_EXAMPLES_DIR "/data/checkpoints"
#define CHECKPOINT_PATH  CHECKPOINT_DIR "/model.bin"

static Tensor *make_one_hot(const int *ids, int n, int n_classes) {
    int shape[2] = {n, n_classes};
    Tensor *out = tg_new(2, shape);
    tg_fill(out, 0.0f);
    float *d = TG_DATAF(out);
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= n_classes) {
            fprintf(stderr, "make_one_hot: id %d out of range\n", ids[i]); exit(1);
        }
        d[i * n_classes + ids[i]] = 1.0f;
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
    char *text      = tg_read_file(CORPUS_PATH, &text_len);
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

    printf("loaded %s: %d chars\n", CORPUS_PATH, text_len);
    printf("vocab size: %d\n", vocab.size);
    printf("params: %d tensors\n", n_params);
    printf("train tokens: %d  val tokens: %d\n", val_start, val_len);
    printf("baseline ln(%d) ~= %.6f\n", vocab.size, logf((float)vocab.size));

    /* Optimizer first, then resume: load_run restores params, Adam moments, the
       cumulative step, and the RNG state. tg_seed_from_entropy() above must
       precede this call or it would clobber the restored RNG state. */
    TgAdam opt = tg_adam_create(params, n_params, beta1, beta2, adam_eps);
    if (tg_checkpoint_load_run(CHECKPOINT_PATH, &opt, TG_LOAD_RESUME, NULL) == 0)
        printf("[ovg] resumed from %s at step %d\n", CHECKPOINT_PATH, opt.step);
    else
        printf("[ovg] no checkpoint found — training from scratch\n");
    ensure_dir(CHECKPOINT_DIR);

    int *inputs  = malloc((size_t)T * sizeof(int));
    int *targets = malloc((size_t)T * sizeof(int));
    if (!inputs || !targets) { fprintf(stderr, "out of memory\n"); exit(1); }

    /* Each run trains `steps` more steps: `step` is this run's counter and drives
       the data order; opt.step is the cumulative count and drives Adam's bias
       correction. */
    tg_training = 1;
    for (int step = 1; step <= steps; step++) {
        int start = (step * 17) % max_train;
        for (int i = 0; i < T; i++) {
            inputs[i]  = all_tokens[start + i];
            targets[i] = all_tokens[start + i + 1];
        }

        Tensor *tgt_hot = make_one_hot(targets, T, vocab.size);
        Tensor *logits  = tg_gpt_forward(&gpt, inputs, 1);
        Tensor *loss    = tg_cross_entropy(logits, tgt_hot);

        int   log_now    = (step == 1 || step % 200 == 0);
        float train_loss = log_now ? tg_scalar_value(loss) : 0.0f;  /* before accumulate frees the graph */

        tg_adam_zero_grads(&opt);
        tg_adam_accumulate(&opt, loss, 1);
        tg_adam_update(&opt, lr, 0.0f);

        if (log_now) {
            /* Val loss: one window sampled from the held-out set */
            int prev = tg_eval_begin();
            int vstart = val_start + (step % val_max_step);
            for (int i = 0; i < T; i++) {
                inputs[i]  = all_tokens[vstart + i];
                targets[i] = all_tokens[vstart + i + 1];
            }
            Tensor *vhot  = make_one_hot(targets, T, vocab.size);
            Tensor *vlog  = tg_gpt_forward(&gpt, inputs, 1);
            Tensor *vloss = tg_cross_entropy(vlog, vhot);
            printf("step %4d/%d (total %d)  train: %.6f  val: %.6f\n",
                   step, steps, opt.step, train_loss, tg_scalar_value(vloss));
            tg_free_graph(vloss);
            tg_eval_end(prev);
        }

        /* Periodic save so a Ctrl-C loses at most 200 steps */
        if (step % 200 == 0)
            tg_checkpoint_save_run(CHECKPOINT_PATH, &opt, &cfg, (int)sizeof cfg);
    }

    /* Final eval on the first T tokens */
    int prev_training = tg_eval_begin();
    for (int i = 0; i < T; i++) {
        inputs[i]  = all_tokens[i];
        targets[i] = all_tokens[i + 1];
    }
    Tensor *eval_tgt_hot = make_one_hot(targets, T, vocab.size);
    Tensor *eval_logits  = tg_gpt_forward(&gpt, inputs, 1);
    Tensor *eval_loss    = tg_cross_entropy(eval_logits, eval_tgt_hot);
    printf("final eval loss: %.6f\n", tg_scalar_value(eval_loss));

    printf("generated: ");
    for (int i = 0; i < T; i++) putchar(tg_vocab_decode(&vocab, inputs[i]));
    tg_generate(&gpt, &vocab, inputs, T, gen_steps, gen_temp, gen_topk, print_char, NULL);
    printf("\n");

    tg_free_graph(eval_loss);
    tg_eval_end(prev_training);

    /* Save checkpoint (params + optimizer state + step + RNG state) */
    if (tg_checkpoint_save_run(CHECKPOINT_PATH, &opt, &cfg, (int)sizeof cfg) == 0)
        printf("[ovg] checkpoint saved to %s (step %d)\n", CHECKPOINT_PATH, opt.step);

    tg_adam_free(&opt);
    tg_gpt_free(&gpt);
    free(params);
    free(inputs);
    free(targets);
    free(all_tokens);
    free(text);

    return 0;
}
