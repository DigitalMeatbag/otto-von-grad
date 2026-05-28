#include "tg_checkpoint.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* Binary format v2 (little-endian):
   [uint32] magic = 0x00475632 ("OVG2", version 2)
   [int32]  n     (param count)
   per param: [int32] ndim, [int32 * TG_MAX_DIMS] shape, then numel floats */

#define OVG_CHECKPOINT_MAGIC 0x00475632u

int tg_checkpoint_save(const char *path, Tensor **params, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "tg_checkpoint_save: cannot open: %s\n", path);
        return -1;
    }
    uint32_t magic = OVG_CHECKPOINT_MAGIC;
    int32_t  cnt   = (int32_t)n;
    if (fwrite(&magic, sizeof magic, 1, f) != 1 ||
        fwrite(&cnt,   sizeof cnt,   1, f) != 1) goto io_err;

    for (int i = 0; i < n; i++) {
        int32_t nd = (int32_t)params[i]->ndim;
        int32_t sh[TG_MAX_DIMS];
        memset(sh, 0, sizeof sh);
        for (int d = 0; d < params[i]->ndim; d++) sh[d] = (int32_t)params[i]->shape[d];
        size_t nel = (size_t)tg_numel(params[i]);
        if (fwrite(&nd, sizeof nd, 1, f) != 1 ||
            fwrite(sh,  sizeof sh, 1, f) != 1 ||
            fwrite(TG_DATAF(params[i]), sizeof(float), nel, f) != nel) goto io_err;
    }
    fclose(f);
    return 0;

io_err:
    fprintf(stderr, "tg_checkpoint_save: write error: %s\n", path);
    fclose(f);
    remove(path);
    return -1;
}

int tg_checkpoint_load(const char *path, Tensor **params, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    uint32_t magic = 0;
    int32_t  cnt   = 0;
    if (fread(&magic, sizeof magic, 1, f) != 1 || magic != OVG_CHECKPOINT_MAGIC) {
        fprintf(stderr, "tg_checkpoint_load: bad magic in %s\n", path);
        fclose(f); return -1;
    }
    if (fread(&cnt, sizeof cnt, 1, f) != 1 || cnt != (int32_t)n) {
        fprintf(stderr, "tg_checkpoint_load: param count mismatch in %s "
                "(file=%d expected=%d)\n", path, (int)cnt, n);
        fclose(f); return -1;
    }
    for (int i = 0; i < n; i++) {
        int32_t nd = 0;
        int32_t sh[TG_MAX_DIMS];
        memset(sh, 0, sizeof sh);
        if (fread(&nd, sizeof nd, 1, f) != 1 || fread(sh, sizeof sh, 1, f) != 1)
            goto io_err;
        if (nd != (int32_t)params[i]->ndim) {
            fprintf(stderr, "tg_checkpoint_load: ndim mismatch at param %d: "
                    "file=%d expected=%d\n", i, (int)nd, params[i]->ndim);
            fclose(f); return -1;
        }
        for (int d = 0; d < params[i]->ndim; d++) {
            if (sh[d] != (int32_t)params[i]->shape[d]) {
                fprintf(stderr, "tg_checkpoint_load: shape mismatch at param %d dim %d: "
                        "file=%d expected=%d\n", i, d, (int)sh[d], params[i]->shape[d]);
                fclose(f); return -1;
            }
        }
        size_t nel = (size_t)tg_numel(params[i]);
        if (fread(TG_DATAF(params[i]), sizeof(float), nel, f) != nel) goto io_err;
    }
    fclose(f);
    return 0;

io_err:
    fprintf(stderr, "tg_checkpoint_load: read error: %s\n", path);
    fclose(f);
    return -1;
}
