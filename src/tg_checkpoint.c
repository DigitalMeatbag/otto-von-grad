#include "tg_checkpoint.h"
#include <stdio.h>
#include <stdint.h>

/* Binary format (little-endian):
   [uint32] magic = 0x00475643 ("OVG\0", version 1)
   [int32]  n     (param count)
   per param: [int32] rows, [int32] cols, then rows*cols floats */

#define OVG_CHECKPOINT_MAGIC 0x00475643u

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
        int32_t r   = (int32_t)params[i]->rows;
        int32_t c   = (int32_t)params[i]->cols;
        size_t  nel = (size_t)r * (size_t)c;
        if (fwrite(&r, sizeof r, 1, f) != 1 ||
            fwrite(&c, sizeof c, 1, f) != 1 ||
            fwrite(params[i]->data, sizeof(float), nel, f) != nel) goto io_err;
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
    if (!f) return -1;  /* missing file is expected on first run — no message */

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
        int32_t r, c;
        if (fread(&r, sizeof r, 1, f) != 1 || fread(&c, sizeof c, 1, f) != 1)
            goto io_err;
        if (r != (int32_t)params[i]->rows || c != (int32_t)params[i]->cols) {
            fprintf(stderr, "tg_checkpoint_load: shape mismatch at param %d: "
                    "file=[%dx%d] expected=[%dx%d]\n",
                    i, (int)r, (int)c, params[i]->rows, params[i]->cols);
            fclose(f); return -1;
        }
        size_t nel = (size_t)r * (size_t)c;
        if (fread(params[i]->data, sizeof(float), nel, f) != nel) goto io_err;
    }
    fclose(f);
    return 0;

io_err:
    fprintf(stderr, "tg_checkpoint_load: read error: %s\n", path);
    fclose(f);
    return -1;
}
