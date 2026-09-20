/* tg_checkpoint.c — checkpoint I/O.
 *
 * Writes format v3 (see tg_checkpoint.h). Reads v3 and the legacy v2
 * parameter-only layout; v1 is rejected. Both writers go through a
 * <path>.tmp + rename so a process killed mid-save leaves the previous
 * checkpoint intact.
 */

#include "tg_checkpoint.h"
#include "tg_rng.h"
#include "ovg_error.h"

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef OVG_CUDA_ENABLED
#  include "tg_cuda_internal.h"
#endif

#define CKPT_FLAG_OPTIMIZER 0x1u
#define CKPT_FLAGS_KNOWN    (CKPT_FLAG_OPTIMIZER)
#define CKPT_OPT_KIND_ADAM  1

/* ── Header ──────────────────────────────────────────────────────────────── */

/* Reads magic and the v3 header fields (or the v2 count) into *info. On return
   the stream is positioned at the config bytes (v3) or at the first param (v2;
   n_params already consumed). Prints to stderr and returns -1 on any failure. */
static int read_header(FILE *f, const char *path, const char *fn, TgCheckpointInfo *info) {
    memset(info, 0, sizeof *info);
    uint32_t magic = 0;
    if (fread(&magic, sizeof magic, 1, f) != 1) {
        fprintf(stderr, "%s: short read in %s\n", fn, path);
        return -1;
    }
    if (magic == TG_CHECKPOINT_MAGIC_V2) {
        int32_t cnt = 0;
        if (fread(&cnt, sizeof cnt, 1, f) != 1) {
            fprintf(stderr, "%s: short read in %s\n", fn, path);
            return -1;
        }
        info->version  = 2;
        info->n_params = (int)cnt;
        return 0;
    }
    if (magic != TG_CHECKPOINT_MAGIC_V3) {
        fprintf(stderr, "%s: bad magic in %s\n", fn, path);
        return -1;
    }
    uint32_t flags = 0, rng = 0;
    int32_t  step = 0, config_len = 0;
    if (fread(&flags,      sizeof flags,      1, f) != 1 ||
        fread(&step,       sizeof step,       1, f) != 1 ||
        fread(&rng,        sizeof rng,        1, f) != 1 ||
        fread(&config_len, sizeof config_len, 1, f) != 1) {
        fprintf(stderr, "%s: short read in %s\n", fn, path);
        return -1;
    }
    if (flags & ~CKPT_FLAGS_KNOWN) {
        fprintf(stderr, "%s: unknown flag bits 0x%08x in %s\n", fn, (unsigned)flags, path);
        return -1;
    }
    if (config_len < 0) {
        fprintf(stderr, "%s: negative config_len in %s\n", fn, path);
        return -1;
    }
    info->version       = 3;
    info->has_optimizer = (flags & CKPT_FLAG_OPTIMIZER) ? 1 : 0;
    info->step          = (int)step;
    info->rng_state     = rng;
    info->config_len    = (int)config_len;
    return 0;
}

/* v3 only: skip the config bytes and read n_params into *info. */
static int read_config_skip_and_count(FILE *f, const char *path, const char *fn,
                                      TgCheckpointInfo *info) {
    if (info->version != 3) return 0;
    if (info->config_len > 0 && fseek(f, (long)info->config_len, SEEK_CUR) != 0) {
        fprintf(stderr, "%s: short read in %s\n", fn, path);
        return -1;
    }
    int32_t cnt = 0;
    if (fread(&cnt, sizeof cnt, 1, f) != 1) {
        fprintf(stderr, "%s: short read in %s\n", fn, path);
        return -1;
    }
    info->n_params = (int)cnt;
    return 0;
}

/* ── Params ──────────────────────────────────────────────────────────────── */

static int write_params(FILE *f, Tensor **params, int n) {
    int32_t cnt = (int32_t)n;
    if (fwrite(&cnt, sizeof cnt, 1, f) != 1) return -1;
    for (int i = 0; i < n; i++) {
#ifdef OVG_CUDA_ENABLED
        if (params[i]->on_cuda) tg_from_cuda(params[i]);
#endif
        int32_t nd = (int32_t)params[i]->ndim;
        int32_t sh[TG_MAX_DIMS];
        memset(sh, 0, sizeof sh);
        for (int d = 0; d < params[i]->ndim; d++) sh[d] = (int32_t)params[i]->shape[d];
        size_t nel = (size_t)tg_numel(params[i]);
        if (fwrite(&nd, sizeof nd, 1, f) != 1 ||
            fwrite(sh,  sizeof sh, 1, f) != 1 ||
            fwrite(TG_DATAF(params[i]), sizeof(float), nel, f) != nel) return -1;
    }
    return 0;
}

/* Reads params after the count has been consumed and validated. Loaded params
   are re-uploaded with tg_to_cuda when they live on the device (this also
   copies the stale host grad buffer, which is harmless: grads are zeroed
   before their next use). */
static int read_params(FILE *f, const char *path, const char *fn,
                       const TgCheckpointInfo *info, Tensor **params, int n) {
    if (info->n_params != n) {
        fprintf(stderr, "%s: param count mismatch in %s (file=%d expected=%d)\n",
                fn, path, info->n_params, n);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        int32_t nd = 0;
        int32_t sh[TG_MAX_DIMS];
        memset(sh, 0, sizeof sh);
        if (fread(&nd, sizeof nd, 1, f) != 1 || fread(sh, sizeof sh, 1, f) != 1) {
            fprintf(stderr, "%s: read error: %s\n", fn, path);
            return -1;
        }
        if (nd != (int32_t)params[i]->ndim) {
            fprintf(stderr, "%s: ndim mismatch at param %d: file=%d expected=%d\n",
                    fn, i, (int)nd, params[i]->ndim);
            return -1;
        }
        for (int d = 0; d < params[i]->ndim; d++) {
            if (sh[d] != (int32_t)params[i]->shape[d]) {
                fprintf(stderr, "%s: shape mismatch at param %d dim %d: file=%d expected=%d\n",
                        fn, i, d, (int)sh[d], params[i]->shape[d]);
                return -1;
            }
        }
        size_t nel = (size_t)tg_numel(params[i]);
        if (fread(TG_DATAF(params[i]), sizeof(float), nel, f) != nel) {
            fprintf(stderr, "%s: read error: %s\n", fn, path);
            return -1;
        }
#ifdef OVG_CUDA_ENABLED
        if (params[i]->on_cuda) tg_to_cuda(params[i]);
#endif
    }
    return 0;
}

/* ── Atomic-ish write ────────────────────────────────────────────────────── */

static char *tmp_path_for(const char *path) {
    size_t len = strlen(path);
    char *tmp = malloc(len + 5);
    if (!tmp) ovg_fatal("tg_checkpoint: out of memory");
    memcpy(tmp, path, len);
    memcpy(tmp + len, ".tmp", 5);
    return tmp;
}

/* Writes the file body via `body`, then replaces <path> with <path>.tmp.
   Windows rename() fails when the target exists, so remove first. */
typedef int (*CkptBodyFn)(FILE *f, void *ctx);

static int write_atomic(const char *path, const char *fn, CkptBodyFn body, void *ctx) {
    char *tmp = tmp_path_for(path);
    FILE *f = fopen(tmp, "wb");
    if (!f) {
        fprintf(stderr, "%s: cannot open: %s\n", fn, tmp);
        free(tmp);
        return -1;
    }
    int rc = body(f, ctx);
    if (fclose(f) != 0) rc = -1;
    if (rc != 0) {
        fprintf(stderr, "%s: write error: %s\n", fn, path);
        remove(tmp);
        free(tmp);
        return -1;
    }
    remove(path);
    if (rename(tmp, path) != 0) {
        fprintf(stderr, "%s: cannot rename %s -> %s\n", fn, tmp, path);
        remove(tmp);
        free(tmp);
        return -1;
    }
    free(tmp);
    return 0;
}

/* ── Weights-only pair ───────────────────────────────────────────────────── */

typedef struct { Tensor **params; int n; } WeightsCtx;

static int write_weights_body(FILE *f, void *vctx) {
    WeightsCtx *c = vctx;
    uint32_t magic = TG_CHECKPOINT_MAGIC_V3, flags = 0, rng = 0;
    int32_t  step = 0, config_len = 0;
    if (fwrite(&magic,      sizeof magic,      1, f) != 1 ||
        fwrite(&flags,      sizeof flags,      1, f) != 1 ||
        fwrite(&step,       sizeof step,       1, f) != 1 ||
        fwrite(&rng,        sizeof rng,        1, f) != 1 ||
        fwrite(&config_len, sizeof config_len, 1, f) != 1) return -1;
    return write_params(f, c->params, c->n);
}

int tg_checkpoint_save(const char *path, Tensor **params, int n) {
    WeightsCtx c = { params, n };
    return write_atomic(path, "tg_checkpoint_save", write_weights_body, &c);
}

int tg_checkpoint_load(const char *path, Tensor **params, int n) {
    static const char *fn = "tg_checkpoint_load";
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    TgCheckpointInfo info;
    int rc = read_header(f, path, fn, &info);
    if (rc == 0) rc = read_config_skip_and_count(f, path, fn, &info);
    if (rc == 0) rc = read_params(f, path, fn, &info, params, n);
    /* The optimizer section, if any, is left unread. */
    fclose(f);
    return rc;
}

/* ── Info ────────────────────────────────────────────────────────────────── */

int tg_checkpoint_info(const char *path, TgCheckpointInfo *info, void *config_out, int config_cap) {
    static const char *fn = "tg_checkpoint_info";
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    TgCheckpointInfo hdr;
    int rc = read_header(f, path, fn, &hdr);
    if (rc == 0 && hdr.version == 3) {
        int ncopy = hdr.config_len < config_cap ? hdr.config_len : config_cap;
        if (ncopy < 0) ncopy = 0;
        if (ncopy > 0 && fread(config_out, 1, (size_t)ncopy, f) != (size_t)ncopy) {
            fprintf(stderr, "%s: short read in %s\n", fn, path);
            rc = -1;
        }
        if (rc == 0 && hdr.config_len > ncopy &&
            fseek(f, (long)(hdr.config_len - ncopy), SEEK_CUR) != 0) {
            fprintf(stderr, "%s: short read in %s\n", fn, path);
            rc = -1;
        }
        int32_t cnt = 0;
        if (rc == 0 && fread(&cnt, sizeof cnt, 1, f) != 1) {
            fprintf(stderr, "%s: short read in %s\n", fn, path);
            rc = -1;
        }
        hdr.n_params = (int)cnt;
    }
    fclose(f);
    if (rc == 0 && info) *info = hdr;
    return rc;
}

/* ── Run state pair ──────────────────────────────────────────────────────── */

typedef struct { const TgAdam *opt; const void *config; int config_len; } RunCtx;

/* Host scratch sized to the largest param, for staging device moments. */
static float *moment_scratch(const TgAdam *opt, const char *fn) {
#ifdef OVG_CUDA_ENABLED
    if (opt->on_cuda) {
        int max_nel = 0;
        for (int i = 0; i < opt->n_params; i++) {
            int nel = tg_numel(opt->params[i]);
            if (nel > max_nel) max_nel = nel;
        }
        float *scratch = malloc((size_t)max_nel * sizeof(float));
        if (!scratch) ovg_fatal("%s: out of memory", fn);
        return scratch;
    }
#endif
    (void)opt; (void)fn;
    return NULL;
}

/* Writes one moment buffer, downloading from the device through host scratch
   when the optimizer lives there. */
static int write_moment(FILE *f, const TgAdam *opt, float *buf, int nel, float *scratch) {
    const float *src = buf;
#ifdef OVG_CUDA_ENABLED
    if (opt->on_cuda) {
        tg_cuda_download_floats(scratch, buf, nel);
        src = scratch;
    }
#else
    (void)opt; (void)scratch;
#endif
    return fwrite(src, sizeof(float), (size_t)nel, f) == (size_t)nel ? 0 : -1;
}

static int write_run_body(FILE *f, void *vctx) {
    RunCtx *c = vctx;
    const TgAdam *opt = c->opt;
    uint32_t magic = TG_CHECKPOINT_MAGIC_V3, flags = CKPT_FLAG_OPTIMIZER;
    uint32_t rng   = tg_rng_get_state();
    int32_t  step  = (int32_t)opt->step, config_len = (int32_t)c->config_len;
    if (fwrite(&magic,      sizeof magic,      1, f) != 1 ||
        fwrite(&flags,      sizeof flags,      1, f) != 1 ||
        fwrite(&step,       sizeof step,       1, f) != 1 ||
        fwrite(&rng,        sizeof rng,        1, f) != 1 ||
        fwrite(&config_len, sizeof config_len, 1, f) != 1) return -1;
    if (config_len > 0 &&
        fwrite(c->config, 1, (size_t)config_len, f) != (size_t)config_len) return -1;
    if (write_params(f, opt->params, opt->n_params) != 0) return -1;

    int32_t kind = CKPT_OPT_KIND_ADAM;
    if (fwrite(&kind,       sizeof kind,       1, f) != 1 ||
        fwrite(&opt->beta1, sizeof opt->beta1, 1, f) != 1 ||
        fwrite(&opt->beta2, sizeof opt->beta2, 1, f) != 1 ||
        fwrite(&opt->eps,   sizeof opt->eps,   1, f) != 1) return -1;

    float *scratch = moment_scratch(opt, "tg_checkpoint_save_run");
    int rc = 0;
    for (int i = 0; i < opt->n_params && rc == 0; i++) {
        int nel = tg_numel(opt->params[i]);
        if (write_moment(f, opt, opt->m[i], nel, scratch) != 0 ||
            write_moment(f, opt, opt->v[i], nel, scratch) != 0) rc = -1;
    }
    free(scratch);
    return rc;
}

int tg_checkpoint_save_run(const char *path, const TgAdam *opt, const void *config, int config_len) {
    if (!opt || !opt->params || opt->n_params <= 0 || !opt->m || !opt->v)
        ovg_fatal("tg_checkpoint_save_run: optimizer is not initialised");
    if (config_len < 0 || (config_len > 0 && !config))
        ovg_fatal("tg_checkpoint_save_run: bad config (len=%d)", config_len);
    RunCtx c = { opt, config, config_len };
    return write_atomic(path, "tg_checkpoint_save_run", write_run_body, &c);
}

/* Reads one moment buffer into buf (host), or via host scratch onto the device. */
static int read_moment(FILE *f, const TgAdam *opt, float *buf, int nel, float *scratch) {
    float *dst = buf;
#ifdef OVG_CUDA_ENABLED
    if (opt->on_cuda) dst = scratch;
#else
    (void)opt; (void)scratch;
#endif
    if (fread(dst, sizeof(float), (size_t)nel, f) != (size_t)nel) return -1;
#ifdef OVG_CUDA_ENABLED
    if (opt->on_cuda) tg_cuda_upload_floats(buf, scratch, nel);
#endif
    return 0;
}

static int read_optimizer_section(FILE *f, const char *path, const char *fn, TgAdam *opt) {
    int32_t kind = 0;
    float b1 = 0.0f, b2 = 0.0f, eps = 0.0f;
    if (fread(&kind, sizeof kind, 1, f) != 1 ||
        fread(&b1,   sizeof b1,   1, f) != 1 ||
        fread(&b2,   sizeof b2,   1, f) != 1 ||
        fread(&eps,  sizeof eps,  1, f) != 1) {
        fprintf(stderr, "%s: short read in %s\n", fn, path);
        return -1;
    }
    if (kind != CKPT_OPT_KIND_ADAM) {
        fprintf(stderr, "%s: unknown optimizer kind %d in %s\n", fn, (int)kind, path);
        return -1;
    }
    if (b1 != opt->beta1 || b2 != opt->beta2 || eps != opt->eps) {
        fprintf(stderr, "%s: optimizer hyperparameters in %s (beta1=%g beta2=%g eps=%g) "
                "differ from the target (beta1=%g beta2=%g eps=%g); "
                "use TG_LOAD_INIT_FROM_WEIGHTS to start fresh moments\n",
                fn, path, b1, b2, eps, opt->beta1, opt->beta2, opt->eps);
        return -1;
    }

    float *scratch = moment_scratch(opt, fn);
    int rc = 0;
    for (int i = 0; i < opt->n_params && rc == 0; i++) {
        int nel = tg_numel(opt->params[i]);
        if (read_moment(f, opt, opt->m[i], nel, scratch) != 0 ||
            read_moment(f, opt, opt->v[i], nel, scratch) != 0) {
            fprintf(stderr, "%s: short read in %s\n", fn, path);
            rc = -1;
        }
    }
    free(scratch);
    return rc;
}

int tg_checkpoint_load_run(const char *path, TgAdam *opt, TgLoadMode mode, TgCheckpointInfo *info) {
    static const char *fn = "tg_checkpoint_load_run";
    if (!opt || !opt->params || opt->n_params <= 0 || !opt->m || !opt->v)
        ovg_fatal("tg_checkpoint_load_run: optimizer is not initialised");
    if (mode != TG_LOAD_RESUME && mode != TG_LOAD_INIT_FROM_WEIGHTS)
        ovg_fatal("tg_checkpoint_load_run: bad load mode %d", (int)mode);

    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    TgCheckpointInfo hdr;
    int rc = read_header(f, path, fn, &hdr);
    int restore = (rc == 0 && mode == TG_LOAD_RESUME && hdr.has_optimizer);
    if (restore && hdr.rng_state == 0) {
        fprintf(stderr, "%s: optimizer section with rng_state == 0 in %s (corrupt)\n", fn, path);
        rc = -1;
    }
    if (rc == 0) rc = read_config_skip_and_count(f, path, fn, &hdr);
    if (rc == 0) rc = read_params(f, path, fn, &hdr, opt->params, opt->n_params);

    if (rc == 0 && restore) {
        rc = read_optimizer_section(f, path, fn, opt);
        if (rc == 0) {
            opt->step = hdr.step;
            tg_rng_set_state(hdr.rng_state);
        }
    } else if (rc == 0) {
        tg_adam_reset(opt);
        if (mode == TG_LOAD_RESUME) {
            printf("[ovg] checkpoint %s: no optimizer state; moments zeroed, step reset to 0\n", path);
            fflush(stdout);
        }
    }
    fclose(f);
    if (rc == 0 && info) *info = hdr;
    return rc;
}
