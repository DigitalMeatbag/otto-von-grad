#ifndef TG_CHECKPOINT_H
#define TG_CHECKPOINT_H

#include "tg_tensor.h"

/* Save params[0..n) to a binary file.
   Returns 0 on success, -1 on I/O error (message written to stderr). */
int tg_checkpoint_save(const char *path, Tensor **params, int n);

/* Load params[0..n) from a binary file previously written by tg_checkpoint_save.
   Validates magic, count, and per-tensor shapes against the target tensors.
   Returns 0 on success, -1 on any error (shape mismatch, bad magic, etc.).
   A missing file returns -1 silently; other errors write to stderr. */
int tg_checkpoint_load(const char *path, Tensor **params, int n);

#endif /* TG_CHECKPOINT_H */
