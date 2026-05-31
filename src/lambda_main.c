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

#ifdef OVG_CUDA_ENABLED
#  include "tg_cuda.h"
#endif

/* Canonical character set for all lambda curriculum phases.
   \n  space  ( ) - . = > \  a-h
   Phase 1 uses 14 of these; phases 2+ use all 17. Pinning the vocab
   here keeps TokEmb/Wout dims stable across staged fine-tuning. */
#define LAMBDA_VOCAB "\n ()-.=>\\abcdefgh"

#define MAX_PHASES 9
static const char *CORPUS_PATHS[MAX_PHASES + 1] = {
    NULL,
    "data/text/lambda_phase1.txt",
    "data/text/lambda_phase2.txt",
    "data/text/lambda_phase3.txt",
    "data/text/lambda_phase4.txt",
    "data/text/lambda_phase5.txt",
    "data/text/lambda_phase6.txt",
    "data/text/lambda_phase7.txt",
    "data/text/lambda_phase8.txt",
    "data/text/lambda_phase9.txt",
};
static const char *CKPT_PATHS[MAX_PHASES + 1] = {
    NULL,
    "data/checkpoints/lambda_phase1.bin",
    "data/checkpoints/lambda_phase2.bin",
    "data/checkpoints/lambda_phase3.bin",
    "data/checkpoints/lambda_phase4.bin",
    "data/checkpoints/lambda_phase5.bin",
    "data/checkpoints/lambda_phase6.bin",
    "data/checkpoints/lambda_phase7.bin",
    "data/checkpoints/lambda_phase8.bin",
    "data/checkpoints/lambda_phase9.bin",
};

static void print_char(char c, void *ud) { (void)ud; putchar(c); }

int main(int argc, char **argv) {
    int phase = 1;
    const char *prompt = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--prompt requires an argument\n");
                return 1;
            }
            prompt = argv[++i];
        } else {
            int p = atoi(argv[i]);
            if (p < 1 || p > MAX_PHASES) {
                fprintf(stderr,
                    "usage: otto_lambda [phase] [--prompt TEXT]  (phase 1-%d)\n",
                    MAX_PHASES);
                return 1;
            }
            phase = p;
        }
    }

    const char *corpus_path = CORPUS_PATHS[phase];
    const char *save_path   = CKPT_PATHS[phase];
    /* inference loads the phase's own checkpoint;
       training for phase > 1 starts from the previous phase */
    const char *load_path   = (prompt || phase == 1) ? CKPT_PATHS[phase]
                                                      : CKPT_PATHS[phase - 1];

    /* diagnostics go to stderr in inference mode so stdout stays clean */
    FILE *diag = prompt ? stderr : stdout;

    TgGPTConfig cfg = {
        .vocab_size = 0,   /* filled after vocab build */
        .embed_dim  = 64,
        .hidden_dim = 128,
        .seq_len    = 512,
        .n_blocks   = 4,
        .n_heads    = 4,
    };
    const int   steps      = 50000;
    const int   batch_size = 16;
    const float lr         = 3e-4f;
    const float beta1      = 0.9f;
    const float beta2      = 0.999f;
    const float adam_eps   = 1e-8f;
    const float gen_temp   = 0.8f;
    const int   gen_topk   = 10;
    const int   gen_steps  = 200;

    TgVocab vocab   = tg_vocab_from_chars(LAMBDA_VOCAB);
    cfg.vocab_size  = vocab.size;
    const int T     = cfg.seq_len;

    TgGPT gpt = tg_gpt_create_from_config(&cfg);

    int max_params  = 3 + cfg.n_blocks * 12 + 4;
    Tensor **params = malloc((size_t)max_params * sizeof(Tensor *));
    if (!params) { fprintf(stderr, "out of memory\n"); exit(1); }
    int n_params = tg_gpt_collect_params(&gpt, params, max_params);

    fprintf(diag, "phase %d  vocab size: %d  params: %d tensors\n",
            phase, vocab.size, n_params);

    /* Load checkpoint into CPU buffers */
    if (tg_checkpoint_load(load_path, params, n_params) == 0)
        fprintf(diag, "[lambda] loaded %s\n", load_path);
    else if (prompt) {
        fprintf(stderr, "[lambda] no checkpoint at %s\n", load_path);
        tg_gpt_free(&gpt);
        free(params);
        return 1;
    } else
        fprintf(diag, "[lambda] no checkpoint — training from scratch\n");

    /* ------------------------------------------------------------------ */
    /* Inference mode                                                       */
    /* ------------------------------------------------------------------ */
    if (prompt) {
#ifdef OVG_CUDA_ENABLED
        for (int i = 0; i < n_params; i++)
            tg_to_cuda(params[i]);
        fprintf(stderr, "[lambda] inference on GPU\n");
#endif
        int plen = (int)strlen(prompt);
        int *ctx = calloc((size_t)T, sizeof(int));
        if (!ctx) { fprintf(stderr, "out of memory\n"); return 1; }

        int cur; /* next free position in ctx */
        if (plen <= T) {
            for (int i = 0; i < plen; i++)
                ctx[i] = tg_vocab_encode(&vocab, prompt[i]);
            cur = plen;
        } else {
            /* prompt longer than window: use the last T chars */
            int skip = plen - T;
            for (int i = 0; i < T; i++)
                ctx[i] = tg_vocab_encode(&vocab, prompt[skip + i]);
            cur = T;
        }

        fputs(prompt, stdout);
        fflush(stdout);

        tg_training = 0;
        for (int s = 0; s < T; s++) {
            Tensor *logits = tg_gpt_forward(&gpt, ctx, 1);
#ifdef OVG_CUDA_ENABLED
            tg_from_cuda(logits);
#endif
            /* read from the last real token while the window is still growing;
               once full, always read from T-1 */
            int row  = (cur < T) ? cur - 1 : T - 1;
            int next = tg_sample_argmax(logits, row);
            char c   = tg_vocab_decode(&vocab, next);
            putchar(c);
            fflush(stdout);
            if (cur < T) {
                ctx[cur++] = next;
            } else {
                for (int i = 0; i < T - 1; i++) ctx[i] = ctx[i + 1];
                ctx[T - 1] = next;
            }
            tg_free_graph(logits);
            if (c == '\n') break;
        }

        free(ctx);
#ifdef OVG_CUDA_ENABLED
        for (int i = 0; i < n_params; i++)
            tg_cuda_free(params[i]);
#endif
        tg_gpt_free(&gpt);
        free(params);
        return 0;
    }

    /* ------------------------------------------------------------------ */
    /* Training mode                                                        */
    /* ------------------------------------------------------------------ */
    tg_seed_from_entropy();

    int   text_len;
    char *text      = tg_read_file(corpus_path, &text_len);
    int *all_tokens = tg_tokenize(text, text_len, &vocab);

    if (text_len <= T + 1) {
        fprintf(stderr, "text too short for context length %d\n", T); exit(1);
    }

    int val_start    = text_len - text_len / 10;
    int val_len      = text_len - val_start;
    int max_train    = val_start - T - 1;
    int val_max_step = val_len - T - 1;
    if (max_train <= 0 || val_max_step <= 0) {
        fprintf(stderr, "corpus too small for train/val split\n"); exit(1);
    }

    printf("corpus: %s  (%d chars)\n", corpus_path, text_len);
    printf("train tokens: %d  val tokens: %d\n", val_start, val_len);
    printf("baseline ln(%d) ~= %.6f\n", vocab.size, logf((float)vocab.size));

    /* Move all parameters to GPU and allocate device moment buffers */
#ifdef OVG_CUDA_ENABLED
    for (int i = 0; i < n_params; i++)
        tg_to_cuda(params[i]);

    float **m_buf = malloc((size_t)n_params * sizeof(float *));
    float **v_buf = malloc((size_t)n_params * sizeof(float *));
    if (!m_buf || !v_buf) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int i = 0; i < n_params; i++) {
        int nel  = tg_numel(params[i]);
        m_buf[i] = tg_cuda_malloc_floats(nel);   /* zeroed on device */
        v_buf[i] = tg_cuda_malloc_floats(nel);
        if (!m_buf[i] || !v_buf[i]) { fprintf(stderr, "out of memory\n"); exit(1); }
    }
    printf("[lambda] training on GPU\n");
#else
    float **m_buf = calloc((size_t)n_params, sizeof(float *));
    float **v_buf = calloc((size_t)n_params, sizeof(float *));
    if (!m_buf || !v_buf) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int i = 0; i < n_params; i++) {
        size_t nel = (size_t)tg_numel(params[i]);
        m_buf[i]   = calloc(nel, sizeof(float));
        v_buf[i]   = calloc(nel, sizeof(float));
        if (!m_buf[i] || !v_buf[i]) { fprintf(stderr, "out of memory\n"); exit(1); }
    }
    printf("[lambda] training on CPU\n");
#endif

    int *inputs  = malloc((size_t)(batch_size * T) * sizeof(int));
    int *targets = malloc((size_t)(batch_size * T) * sizeof(int));
    if (!inputs || !targets) { fprintf(stderr, "out of memory\n"); exit(1); }

    tg_training = 1;
    for (int step = 1; step <= steps; step++) {
        int stride = max_train / batch_size;
        for (int b = 0; b < batch_size; b++) {
            int start = ((step * 17) + b * stride) % max_train;
            for (int i = 0; i < T; i++) {
                inputs [b * T + i] = all_tokens[start + i];
                targets[b * T + i] = all_tokens[start + i + 1];
            }
        }

        /* tg_cross_entropy_sparse: takes host int[] targets, no one-hot needed.
           On GPU logits it uploads ids internally and syncs the loss scalar back. */
        Tensor *logits = tg_gpt_forward(&gpt, inputs, batch_size);
        Tensor *loss   = tg_cross_entropy_sparse(logits, targets, batch_size * T, 0.0f);

        tg_backward(loss);
#ifdef OVG_CUDA_ENABLED
        tg_adam_step_gpu(params, m_buf, v_buf, n_params, lr, step, beta1, beta2, adam_eps);
#else
        tg_adam_step(params, m_buf, v_buf, n_params, lr, step, beta1, beta2, adam_eps);
#endif

        if (step == 1 || step % 500 == 0) {
            float train_loss = TG_DATAF(loss)[0];

            tg_training = 0;
            int vstart = val_start + (step % val_max_step);
            for (int i = 0; i < T; i++) {
                inputs[i]  = all_tokens[vstart + i];
                targets[i] = all_tokens[vstart + i + 1];
            }
            Tensor *vlog  = tg_gpt_forward(&gpt, inputs, 1);
            Tensor *vloss = tg_cross_entropy_sparse(vlog, targets, T, 0.0f);
            printf("step %5d/%d  train: %.6f  val: %.6f\n",
                   step, steps, train_loss, TG_DATAF(vloss)[0]);
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
    Tensor *eval_logits = tg_gpt_forward(&gpt, inputs, 1);
    Tensor *eval_loss   = tg_cross_entropy_sparse(eval_logits, targets, T, 0.0f);
    printf("final eval loss: %.6f\n", TG_DATAF(eval_loss)[0]);
    tg_free_graph(eval_loss);

    /* Generation — inline so we can sync logits from GPU before sampling */
    printf("generated:\n");
    for (int i = 0; i < T; i++) putchar(tg_vocab_decode(&vocab, inputs[i]));
    {
        int *ctx = malloc((size_t)T * sizeof(int));
        if (!ctx) { fprintf(stderr, "out of memory\n"); exit(1); }
        memcpy(ctx, inputs, (size_t)T * sizeof(int));
        for (int s = 0; s < gen_steps; s++) {
            Tensor *logits = tg_gpt_forward(&gpt, ctx, 1);
#ifdef OVG_CUDA_ENABLED
            tg_from_cuda(logits);
#endif
            int next = tg_sample_topk(logits, T - 1, gen_temp, gen_topk);
            print_char(tg_vocab_decode(&vocab, next), NULL);
            for (int i = 0; i < T - 1; i++) ctx[i] = ctx[i + 1];
            ctx[T - 1] = next;
            tg_free_graph(logits);
        }
        putchar('\n');
        free(ctx);
    }

    /* Pull GPU params back to CPU before saving checkpoint */
#ifdef OVG_CUDA_ENABLED
    for (int i = 0; i < n_params; i++)
        tg_from_cuda(params[i]);
#endif

    ensure_dir("data/checkpoints");
    if (tg_checkpoint_save(save_path, params, n_params) == 0)
        printf("[lambda] checkpoint saved to %s\n", save_path);

    /* Cleanup */
#ifdef OVG_CUDA_ENABLED
    for (int i = 0; i < n_params; i++) {
        tg_cuda_free_floats(m_buf[i]);
        tg_cuda_free_floats(v_buf[i]);
        tg_cuda_free(params[i]);
    }
#else
    for (int i = 0; i < n_params; i++) { free(m_buf[i]); free(v_buf[i]); }
#endif
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
