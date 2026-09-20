#ifndef TG_CHECKPOINT_H
#define TG_CHECKPOINT_H

#include <stdint.h>
#include "tg_tensor.h"
#include "tg_optim.h"

/* Binary format v3 (little-endian, fixed-width):
     uint32  magic        = TG_CHECKPOINT_MAGIC_V3
     uint32  flags        bit 0: optimizer section present; other bits must be 0
     int32   step         completed updates at save (0 for weights-only)
     uint32  rng_state    xorshift32 word at save when flags bit 0 is set; 0 otherwise
     int32   config_len   >= 0
     uint8   config[config_len]
     int32   n_params
     per param: int32 ndim, int32 shape[TG_MAX_DIMS], float data[numel]
     if flags & 1:
       int32 optimizer_kind = 1 (Adam); float beta1, beta2, eps
       per param: float m[numel], float v[numel]
   v2 files (magic TG_CHECKPOINT_MAGIC_V2: uint32 magic, int32 n, params) load as
   weights-only; v1 is rejected. Both writers write <path>.tmp then rename. */

#define TG_CHECKPOINT_MAGIC_V2 0x00475632u
#define TG_CHECKPOINT_MAGIC_V3 0x00475633u

typedef struct {
    int      version;        /* 2 or 3 */
    int      has_optimizer;
    int      step;
    uint32_t rng_state;
    int      config_len;
    int      n_params;
} TgCheckpointInfo;

typedef enum {
    TG_LOAD_RESUME = 0,             /* params + optimizer state + step + RNG state (default) */
    TG_LOAD_INIT_FROM_WEIGHTS = 1   /* params only; optimizer reset to step 0; RNG untouched */
} TgLoadMode;

/* Weights-only pair. save writes v3 with flags = 0, step = 0, rng_state = 0,
   config_len = 0. load accepts v2 and v3 and reads params only.
   Validates magic, count, and per-tensor shapes against the target tensors.
   Return 0 on success, -1 on any error (message on stderr; a missing file is
   silent on load). */
int tg_checkpoint_save(const char *path, Tensor **params, int n);
int tg_checkpoint_load(const char *path, Tensor **params, int n);

/* Reads the header only. Copies min(config_len, config_cap) bytes into config_out
   (config_out may be NULL when config_cap == 0). Returns 0, or -1 on error / missing file.
   v2 files report version = 2, has_optimizer = 0, step = 0, rng_state = 0, config_len = 0. */
int tg_checkpoint_info(const char *path, TgCheckpointInfo *info, void *config_out, int config_cap);

/* Full run state: opt->params, opt's m/v/step/betas/eps, the current RNG state, and config.
   Device params and moments are synced to host by this call. */
int tg_checkpoint_save_run(const char *path, const TgAdam *opt, const void *config, int config_len);

/* Loads into opt->params (uploading to the device if they are on one).
   RESUME on a file with an optimizer section: restores m/v (uploaded if on_cuda), step, RNG state.
   RESUME on a v2 file or a v3 file without an optimizer section: params loaded, tg_adam_reset(opt),
     RNG untouched (the header's rng_state is ignored), and a notice on stdout:
     "[ovg] checkpoint <path>: no optimizer state; moments zeroed, step reset to 0".
   INIT_FROM_WEIGHTS: params loaded, tg_adam_reset(opt), RNG untouched, optimizer section skipped.
   info may be NULL. Returns 0, or -1 on any error (params may be partially overwritten on -1;
   callers treat -1 as fatal for the run).
   Ordering contract: call AFTER tg_seed / tg_seed_from_entropy, which overwrite the RNG word. */
int tg_checkpoint_load_run(const char *path, TgAdam *opt, TgLoadMode mode, TgCheckpointInfo *info);

#endif /* TG_CHECKPOINT_H */
