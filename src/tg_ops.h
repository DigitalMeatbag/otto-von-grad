#ifndef TG_OPS_H
#define TG_OPS_H

#include "tg_tensor.h"

Tensor *tg_add(Tensor *a, Tensor *b);
Tensor *tg_sub(Tensor *a, Tensor *b);
Tensor *tg_mul(Tensor *a, Tensor *b);
Tensor *tg_pow(Tensor *a, float p);
Tensor *tg_scale(Tensor *a, float s);
Tensor *tg_matmul(Tensor *a, Tensor *b);
Tensor *tg_transpose(Tensor *a);
Tensor *tg_tanh(Tensor *a);
Tensor *tg_relu(Tensor *a);
Tensor *tg_gelu(Tensor *a);
Tensor *tg_sum(Tensor *a);
Tensor *tg_mean(Tensor *a);
Tensor *tg_mean_rows(Tensor *a);
Tensor *tg_causal_mask(Tensor *scores);
Tensor *tg_layer_norm_rows(Tensor *a, float eps);
Tensor *tg_layer_norm_rows_affine(Tensor *a, Tensor *gamma, Tensor *beta, float eps);
Tensor *tg_softmax_rows(Tensor *a);
Tensor *tg_cross_entropy(Tensor *logits, Tensor *targets);
Tensor *tg_cross_entropy_no_sync(Tensor *logits, Tensor *targets);
Tensor *tg_cross_entropy_sparse(Tensor *logits, const int *class_ids, int n_ids, float label_smoothing);
Tensor *tg_cross_entropy_sparse_no_sync(Tensor *logits, const int *class_ids, int n_ids, float label_smoothing);
Tensor *tg_slice_cols(Tensor *a, int col_start, int col_end);
Tensor *tg_concat_cols(Tensor **parts, int n_parts);
Tensor *tg_dropout(Tensor *a, float p);
Tensor *tg_embed(Tensor *weight, const int *token_ids, int seq_len);

#endif
